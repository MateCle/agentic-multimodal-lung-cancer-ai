"""
Generator node for the LangGraph orchestrator (AFM2-aligned).

Reconstructs missing modalities via LLM-guided k-NN retrieval.
Following AFM2's Generation Agent pattern with N>1 best-of-N support:

  1. LLM interprets refined guidance (from pre-Verifier) to produce
     modality weights for k-NN similarity computation.
  2. Top k*N neighbours retrieved; split into N candidate arrays per modality.
  3. N candidates stored in generation_candidates for the post-Verifier to
     score and rank (best-of-N selection).
  4. On self-refinement retries, correction hints from the Verifier are
     included in the LLM prompt.

Usage in graph.py:
    from src.orchestrator.nodes.generator import make_generator_node, build_pool_index
    pool = build_pool_index(all_data, train_ids)
    generator = make_generator_node(pool, llm, metadata, n_candidates=3)
    builder.add_node("generator", generator)
"""

import logging

import numpy as np

from src.data_loader import MODALITY_DIMS, MODALITY_KEYS, load_patient
from src.orchestrator.llm import BaseLLMClient
from src.orchestrator.state import PatientState

logger = logging.getLogger(__name__)

BASE_K = 5
K_INCREMENT = 3
DEFAULT_N_CANDIDATES = 3

GENERATOR_SYSTEM_PROMPT = (
    "You are the Generator agent in a multimodal lung cancer survival "
    "prediction system following the AFM2 framework. Your role is to "
    "interpret mining rules and decide how to weight features for "
    "patient similarity search (k-NN retrieval).\n\n"
    "You receive guidance describing biological relationships "
    "between available and missing modalities. Based on this guidance, "
    "identify which available modalities and feature ranges are most "
    "important for finding similar patients.\n\n"
    "Respond ONLY in JSON:\n"
    '{"modality_weights": {"modality_name": <float 0-1>, ...}, '
    '"reasoning": "...", "k_suggestion": <int>}'
)


# ---------------------------------------------------------------------------
# Pool index construction
# ---------------------------------------------------------------------------


def _build_pool_entry(pid: str, patient: dict) -> dict:
    """Build a single retrieval pool entry with normalised feature vectors."""
    entry: dict = {
        "patient_id": pid,
        "available": patient["available_modalities"],
        "features": {},
        "features_norm": {},
    }
    for mod in MODALITY_KEYS:
        if patient[mod] is None:
            continue
        arr = np.array(patient[mod]).flatten().astype(np.float32)
        if arr.size != MODALITY_DIMS[mod]:
            continue
        entry["features"][mod] = arr
        norm = np.linalg.norm(arr)
        entry["features_norm"][mod] = arr / norm if norm > 0 else arr
    return entry


def build_pool_index(raw_data: dict, patient_ids: list[str]) -> list[dict]:
    """
    Precompute a retrieval index from training patient IDs.
    Called once at graph build time.
    """
    pool = []
    for pid in patient_ids:
        patient = load_patient(pid, raw_data)
        if patient is None:
            continue
        pool.append(_build_pool_entry(pid, patient))
    return pool


# ---------------------------------------------------------------------------
# k-NN retrieval
# ---------------------------------------------------------------------------


def _normalize(arr: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(arr)
    return arr / norm if norm > 0 else arr


def _compute_similarity(query_norm, candidate, modality_weights=None):
    """
    Compute weighted cosine similarity across shared modalities.
    If modality_weights is provided, each modality's similarity
    is multiplied by its weight before averaging.
    """
    shared = [m for m in query_norm if m in candidate["features_norm"]]
    if not shared:
        return None

    # Precompute weight getter to avoid a conditional inside the loop
    get_w = (
        (lambda m: modality_weights.get(m, 0.0))
        if modality_weights
        else (lambda _: 1.0)
    )

    weighted_sims: list[float] = []
    total_weight = 0.0
    for m in shared:
        sim = float(np.dot(query_norm[m], candidate["features_norm"][m]))
        w = get_w(m)
        weighted_sims.append(sim * w)
        total_weight += w

    if total_weight == 0:
        return None
    return sum(weighted_sims) / total_weight


def _knn_retrieve(
    query_features, target_modality, pool, k=5, exclude_pid=None, modality_weights=None
):
    """
    Find k nearest neighbors who have target_modality.
    Returns (weighted_average_features, neighbor_info_list).
    """
    dim = MODALITY_DIMS[target_modality]
    query_norm = {mod: _normalize(arr) for mod, arr in query_features.items()}

    candidates = []
    for entry in pool:
        if entry["patient_id"] == exclude_pid:
            continue
        if target_modality not in entry["features"]:
            continue
        sim = _compute_similarity(query_norm, entry, modality_weights)
        if sim is not None:
            candidates.append((sim, entry))

    if not candidates:
        return np.zeros(dim, dtype=np.float32), []

    candidates.sort(key=lambda x: x[0], reverse=True)
    top_k = candidates[:k]

    weights = np.array([max(sim, 0.0) for sim, _ in top_k], dtype=np.float32)
    weight_sum = weights.sum()
    if weight_sum == 0:
        weights = np.ones(len(top_k), dtype=np.float32) / len(top_k)
    else:
        weights /= weight_sum

    result = np.zeros(dim, dtype=np.float32)
    neighbor_info = []

    for w, (sim, entry) in zip(weights, top_k):
        result += w * entry["features"][target_modality]
        neighbor_info.append(
            {
                "patient_id": entry["patient_id"],
                "similarity": round(float(sim), 4),
                "weight": round(float(w), 4),
            }
        )

    return result, neighbor_info


def _average_chunk(
    chunk: list, target_modality: str, dim: int
) -> tuple[np.ndarray, list[dict]]:
    """Weighted average of one k-neighbour chunk → (reconstruction, info)."""
    weights = np.array([max(sim, 0.0) for sim, _ in chunk], dtype=np.float32)
    weight_sum = weights.sum()
    if weight_sum == 0:
        weights = np.ones(len(chunk), dtype=np.float32) / len(chunk)
    else:
        weights /= weight_sum

    result = np.zeros(dim, dtype=np.float32)
    info: list[dict] = []
    for w, (sim, entry) in zip(weights, chunk):
        result += w * entry["features"][target_modality]
        info.append(
            {
                "patient_id": entry["patient_id"],
                "similarity": round(float(sim), 4),
                "weight": round(float(w), 4),
            }
        )
    return result, info


def _knn_retrieve_candidates(
    query_features,
    target_modality,
    pool,
    k: int = 5,
    n_candidates: int = 1,
    exclude_pid=None,
    modality_weights=None,
) -> tuple[list[np.ndarray], list[dict]]:
    """
    Retrieve N candidate reconstructions for target_modality.

    Fetches top k*N neighbours, splits them into N consecutive chunks of k,
    and computes a similarity-weighted average within each chunk. This gives
    N diverse candidates spanning from the best-matched to lower-matched
    neighbours in the pool.

    Returns:
        candidates:    list of N np.ndarray of shape (MODALITY_DIMS[target],)
        neighbor_info: neighbour metadata for the first (best) candidate
    """
    dim = MODALITY_DIMS[target_modality]
    query_norm = {mod: _normalize(arr) for mod, arr in query_features.items()}

    pool_entries = []
    for entry in pool:
        if entry["patient_id"] == exclude_pid:
            continue
        if target_modality not in entry["features"]:
            continue
        sim = _compute_similarity(query_norm, entry, modality_weights)
        if sim is not None:
            pool_entries.append((sim, entry))

    if not pool_entries:
        return [np.zeros(dim, dtype=np.float32)] * n_candidates, []

    pool_entries.sort(key=lambda x: x[0], reverse=True)
    top_kn = pool_entries[: k * n_candidates]

    # Stride sampling: candidate i picks every N-th entry starting at offset i.
    # e.g. N=3, k=5 → candidate 0: [0,3,6,9,12], candidate 1: [1,4,7,10,13], ...
    # Each candidate therefore spans the full similarity range of the top-kN
    # neighbourhood rather than getting progressively worse tiers, which makes
    # the reconstructions genuinely diverse while keeping average similarity
    # comparable across candidates — a requirement for the Verifier to
    # discriminate on biological criteria rather than similarity rank alone.
    candidates: list[np.ndarray] = []
    neighbor_info: list[dict] = []

    for i in range(n_candidates):
        chunk = top_kn[i::n_candidates][:k]
        if not chunk:
            candidates.append(np.zeros(dim, dtype=np.float32))
            continue
        result, info = _average_chunk(chunk, target_modality, dim)
        candidates.append(result)
        if i == 0:
            neighbor_info = info

    return candidates, neighbor_info


# ---------------------------------------------------------------------------
# LLM guidance for retrieval
# ---------------------------------------------------------------------------


def _build_guidance_prompt(
    guidance_text: str,
    available_modalities: list[str],
    target_modality: str,
    correction_hint: str = "",
) -> str:
    """Build prompt for the Generator's LLM call."""
    prompt = (
        f"Guidance for reconstructing '{target_modality}':\n"
        f"  {guidance_text}\n\n"
        f"Available modalities for similarity search: {available_modalities}\n\n"
        f"Based on this guidance, assign a weight (0.0 to 1.0) to each available "
        f"modality indicating how important it is for finding similar patients "
        f"who can help reconstruct '{target_modality}'.\n"
        f"Also suggest an appropriate k for k-NN retrieval (3-20).\n"
    )
    if correction_hint:
        prompt += (
            f"\nPREVIOUS ATTEMPT FAILED. Verifier feedback:\n"
            f"  {correction_hint}\n"
            f"Adjust your weights and k accordingly. Make a concrete change; "
            f"do not repeat the same weights and k.\n"
        )
    prompt += (
        f"\nRespond ONLY in JSON:\n"
        f'{{"modality_weights": {{"{available_modalities[0]}": 0.8, ...}}, '
        f'"reasoning": "...", "k_suggestion": 5}}'
    )
    return prompt


# ---------------------------------------------------------------------------
# Node factory
# ---------------------------------------------------------------------------


def _get_attempt_number(execution_log):
    return sum(1 for line in execution_log if "[Generator] Completed" in line)


def _collect_query_features(state: PatientState) -> dict:
    query_features = {}
    for mod in MODALITY_KEYS:
        data = state.get(mod)
        if data is None:
            continue

        arr = np.array(data).flatten().astype(np.float32)
        if arr.size == MODALITY_DIMS[mod]:
            query_features[mod] = arr

    return query_features


def _get_llm_guidance(
    llm: BaseLLMClient,
    modality: str,
    guidance_text: str,
    available_mods: list[str],
    correction_hint: str,
    base_k: int,
    log_lines: list[str],
) -> tuple[dict | None, int]:
    modality_weights = None
    k = base_k

    if llm is None:
        return modality_weights, k

    guidance_prompt = _build_guidance_prompt(
        guidance_text=guidance_text,
        available_modalities=available_mods,
        target_modality=modality,
        correction_hint=correction_hint,
    )
    try:
        response = llm.invoke_json(guidance_prompt, system=GENERATOR_SYSTEM_PROMPT)
        modality_weights = _sanitize_modality_weights(
            response.get("modality_weights"), available_mods
        )
        k_suggestion = response.get("k_suggestion")
        if k_suggestion and isinstance(k_suggestion, (int, float)):
            k = max(3, min(20, int(k_suggestion)))
        if correction_hint and k < base_k:
            k = base_k
        reasoning = response.get("reasoning", "")
        log_lines.append(
            f"[Generator] LLM guidance for '{modality}': "
            f"weights={modality_weights}, k={k}. "
            f"{reasoning[:80]}"
        )
    except Exception as e:
        log_lines.append(
            f"[Generator] LLM guidance failed for '{modality}': "
            f"{e}. Using uniform weights."
        )

    return modality_weights, k


def _sanitize_modality_weights(
    raw_weights: dict | None, available_mods: list[str]
) -> dict | None:
    if not isinstance(raw_weights, dict):
        return None

    weights: dict[str, float] = {}
    for mod in available_mods:
        val = raw_weights.get(mod, 0.0)
        try:
            val = float(val)
        except (TypeError, ValueError):
            val = 0.0
        weights[mod] = max(0.0, val)

    if sum(weights.values()) <= 0:
        return None
    return weights


def _append_neighbor_log(
    log_lines: list[str],
    modality: str,
    k: int,
    n_candidates: int,
    attempt: int,
    guidance_text: str,
    hint: str,
    neighbors: list[dict],
) -> None:
    if neighbors:
        log_lines.append(
            f"[Generator] '{modality}' k-NN (k={k}, N={n_candidates}, "
            f"attempt={attempt + 1}). "
            f"Top: {neighbors[0]['patient_id']} "
            f"(sim={neighbors[0]['similarity']:.4f}). "
            f"Guidance: '{guidance_text[:50]}'. Hint: '{hint[:40]}'."
        )
    else:
        log_lines.append(
            f"[Generator] '{modality}': no neighbors (N={n_candidates}). Zero fallback."
        )


def make_generator_node(
    pool: list[dict],
    llm: BaseLLMClient = None,
    metadata: dict = None,
    n_candidates: int = DEFAULT_N_CANDIDATES,
):
    """
    Returns a Generator closure following AFM2's Generation Agent.

    For each missing modality:
        1. Calls LLM to interpret refined guidance into modality weights
        2. Computes weighted cosine similarity on shared modalities
        3. Retrieves top-k*N neighbors split into N candidate reconstructions
        4. Stores N candidates in generation_candidates for the Verifier

    On self-refinement retries, correction hints from the Verifier are
    included in the LLM prompt to adjust the retrieval strategy.

    Args:
        pool:         Precomputed pool index from build_pool_index().
        llm:          LLM client (optional — falls back to uniform weights).
        metadata:     Feature name metadata (optional).
        n_candidates: Number of candidates to produce per modality (default 3).
    """

    def generator_node(state: PatientState) -> dict:
        pid = state["patient_id"]
        missing = state["missing_modalities"]
        # Use refined guidance from pre-Verifier; fall back to raw mining rules
        guidance = state.get("guidance") or state.get("mining_rules") or {}
        hints = state.get("correction_hints") or {}
        log_lines = []

        attempt = _get_attempt_number(state.get("execution_log") or [])
        base_k = BASE_K + (attempt * K_INCREMENT)

        query_features = _collect_query_features(state)

        if not query_features:
            zero_candidates = {
                mod: [np.zeros(MODALITY_DIMS[mod], dtype=np.float32)] * n_candidates
                for mod in missing
            }
            log_lines.append(
                f"[Generator] No available modalities for {pid}. "
                f"Zero fallback for: {missing}."
            )
            log_lines.append(f"[Generator] Completed attempt {attempt + 1} for {pid}.")
            return {
                "generation_candidates": zero_candidates,
                "execution_log": log_lines,
            }

        available_mods = list(query_features.keys())
        generation_candidates: dict[str, list[np.ndarray]] = {}

        for modality in missing:
            guidance_text = guidance.get(modality, "Use k-NN with uniform weighting.")
            hint = hints.get(modality, "")

            modality_weights, k = _get_llm_guidance(
                llm=llm,
                modality=modality,
                guidance_text=guidance_text,
                available_mods=available_mods,
                correction_hint=hint,
                base_k=base_k,
                log_lines=log_lines,
            )

            candidates, neighbors = _knn_retrieve_candidates(
                query_features=query_features,
                target_modality=modality,
                pool=pool,
                k=k,
                n_candidates=n_candidates,
                exclude_pid=pid,
                modality_weights=modality_weights,
            )
            generation_candidates[modality] = candidates
            _append_neighbor_log(
                log_lines=log_lines,
                modality=modality,
                k=k,
                n_candidates=n_candidates,
                attempt=attempt,
                guidance_text=guidance_text,
                hint=hint,
                neighbors=neighbors,
            )

        log_lines.append(f"[Generator] Completed attempt {attempt + 1} for {pid}.")
        return {
            "generation_candidates": generation_candidates,
            "execution_log": log_lines,
        }

    return generator_node


# ---------------------------------------------------------------------------
# Mock (no LLM, no pool)
# ---------------------------------------------------------------------------


def generator_node(state: PatientState) -> dict:
    """MOCK fallback: single zero-array candidate per modality."""
    missing = state["missing_modalities"]
    guidance = state.get("guidance") or state.get("mining_rules") or {}
    log_lines = []

    generation_candidates: dict[str, list] = {}
    for modality in missing:
        dim = MODALITY_DIMS[modality]
        generation_candidates[modality] = [np.zeros(dim)]
        rule = guidance.get(modality, "no rule")
        log_lines.append(
            f"[Generator] MOCK '{modality}' zeros, shape=({dim},). Rule: '{rule[:60]}'."
        )

    return {"generation_candidates": generation_candidates, "execution_log": log_lines}
