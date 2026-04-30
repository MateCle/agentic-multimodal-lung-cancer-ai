"""
Generator node for the LangGraph orchestrator (AFM2-aligned).

Reconstructs missing modalities via LLM-guided k-NN retrieval.
Following AFM2's Generation Agent pattern:
    1. LLM interprets mining rules to identify which features to prioritize
    2. k-NN retrieval uses those weights for similarity computation
    3. On self-refinement, correction hints from Verifier refine the LLM prompt

Usage in graph.py:
    from src.orchestrator.nodes.generator import make_generator_node, build_pool_index
    pool = build_pool_index(all_data, train_ids)
    generator = make_generator_node(pool, llm, metadata)
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

GENERATOR_SYSTEM_PROMPT = (
    "You are the Generator agent in a multimodal lung cancer survival "
    "prediction system following the AFM2 framework. Your role is to "
    "interpret mining rules and decide how to weight features for "
    "patient similarity search (k-NN retrieval).\n\n"
    "You receive a mining rule describing biological relationships "
    "between available and missing modalities. Based on this rule, "
    "identify which available modalities and feature ranges are most "
    "important for finding similar patients.\n\n"
    "Respond ONLY in JSON:\n"
    '{"modality_weights": {"modality_name": <float 0-1>, ...}, '
    '"reasoning": "...", "k_suggestion": <int>}'
)


# ---------------------------------------------------------------------------
# Pool index construction
# ---------------------------------------------------------------------------


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

        entry = {
            "patient_id": pid,
            "available": patient["available_modalities"],
            "features": {},
            "features_norm": {},
        }

        for mod in MODALITY_KEYS:
            if patient[mod] is not None:
                arr = np.array(patient[mod]).flatten().astype(np.float32)
                if arr.size != MODALITY_DIMS[mod]:
                    continue
                entry["features"][mod] = arr
                norm = np.linalg.norm(arr)
                entry["features_norm"][mod] = arr / norm if norm > 0 else arr

        pool.append(entry)

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

    sims = []
    weights = []
    for m in shared:
        sim = float(np.dot(query_norm[m], candidate["features_norm"][m]))
        w = modality_weights.get(m, 1.0) if modality_weights else 1.0
        sims.append(sim * w)
        weights.append(w)

    total_weight = sum(weights)
    if total_weight == 0:
        return np.mean(sims)
    return sum(sims) / total_weight


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


# ---------------------------------------------------------------------------
# LLM guidance for retrieval
# ---------------------------------------------------------------------------


def _build_guidance_prompt(
    mining_rule: str,
    available_modalities: list[str],
    target_modality: str,
    correction_hint: str = "",
) -> str:
    """Build prompt for the Generator's LLM call."""
    prompt = (
        f"Mining rule for reconstructing '{target_modality}':\n"
        f"  {mining_rule}\n\n"
        f"Available modalities for similarity search: {available_modalities}\n\n"
        f"Based on this rule, assign a weight (0.0 to 1.0) to each available "
        f"modality indicating how important it is for finding similar patients "
        f"who can help reconstruct '{target_modality}'.\n"
        f"Also suggest an appropriate k for k-NN retrieval (3-20).\n"
    )
    if correction_hint:
        prompt += (
            f"\nPREVIOUS ATTEMPT FAILED. Verifier feedback:\n"
            f"  {correction_hint}\n"
            f"Adjust your weights and k accordingly.\n"
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
    rule: str,
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
        mining_rule=rule,
        available_modalities=available_mods,
        target_modality=modality,
        correction_hint=correction_hint,
    )
    try:
        response = llm.invoke_json(guidance_prompt, system=GENERATOR_SYSTEM_PROMPT)
        modality_weights = response.get("modality_weights")
        k_suggestion = response.get("k_suggestion")
        if k_suggestion and isinstance(k_suggestion, (int, float)):
            k = max(3, min(20, int(k_suggestion)))
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


def _append_neighbor_log(
    log_lines: list[str],
    modality: str,
    k: int,
    attempt: int,
    rule: str,
    hint: str,
    neighbors: list[dict],
) -> None:
    if neighbors:
        log_lines.append(
            f"[Generator] '{modality}' k-NN (k={k}, "
            f"attempt={attempt + 1}). "
            f"Top: {neighbors[0]['patient_id']} "
            f"(sim={neighbors[0]['similarity']:.4f}). "
            f"Rule: '{rule[:50]}'. Hint: '{hint[:40]}'."
        )
    else:
        log_lines.append(f"[Generator] '{modality}': no neighbors. Zero fallback.")


def make_generator_node(
    pool: list[dict], llm: BaseLLMClient = None, metadata: dict = None
):
    """
    Returns a Generator closure following AFM2's Generation Agent.

    For each missing modality:
        1. Calls LLM to interpret mining rules into modality weights
        2. Computes weighted cosine similarity on shared modalities
        3. Retrieves top-k neighbors who have the target modality
        4. Returns similarity-weighted average of their features

    On self-refinement retries, correction hints from the Verifier
    are included in the LLM prompt to adjust the retrieval strategy.

    Args:
        pool:     Precomputed pool index from build_pool_index().
        llm:      LLM client (optional — falls back to uniform weights).
        metadata: Feature name metadata (optional).
    """

    def generator_node(state: PatientState) -> dict:
        pid = state["patient_id"]
        missing = state["missing_modalities"]
        rules = state.get("mining_rules") or {}
        hints = state.get("correction_hints") or {}
        generated = dict(state.get("generated_modalities") or {})
        log_lines = []

        attempt = _get_attempt_number(state.get("execution_log") or [])
        base_k = BASE_K + (attempt * K_INCREMENT)

        query_features = _collect_query_features(state)

        if not query_features:
            for modality in missing:
                generated[modality] = np.zeros(
                    MODALITY_DIMS[modality], dtype=np.float32
                )
            log_lines.append(
                f"[Generator] No available modalities for {pid}. "
                f"Zero fallback for: {missing}."
            )
            log_lines.append(f"[Generator] Completed attempt {attempt + 1} for {pid}.")
            return {"generated_modalities": generated, "execution_log": log_lines}

        available_mods = list(query_features.keys())

        # Generate each missing modality
        for modality in missing:
            rule = rules.get(modality, "no rule available")
            hint = hints.get(modality, "")

            modality_weights, k = _get_llm_guidance(
                llm=llm,
                modality=modality,
                rule=rule,
                available_mods=available_mods,
                correction_hint=hint,
                base_k=base_k,
                log_lines=log_lines,
            )

            result, neighbors = _knn_retrieve(
                query_features=query_features,
                target_modality=modality,
                pool=pool,
                k=k,
                exclude_pid=pid,
                modality_weights=modality_weights,
            )
            generated[modality] = result
            _append_neighbor_log(
                log_lines=log_lines,
                modality=modality,
                k=k,
                attempt=attempt,
                rule=rule,
                hint=hint,
                neighbors=neighbors,
            )

        log_lines.append(f"[Generator] Completed attempt {attempt + 1} for {pid}.")
        return {"generated_modalities": generated, "execution_log": log_lines}

    return generator_node


# ---------------------------------------------------------------------------
# Mock (no LLM, no pool)
# ---------------------------------------------------------------------------


def generator_node(state: PatientState) -> dict:
    """MOCK fallback: zero arrays."""
    missing = state["missing_modalities"]
    rules = state.get("mining_rules") or {}
    generated = dict(state.get("generated_modalities") or {})
    log_lines = []

    for modality in missing:
        dim = MODALITY_DIMS[modality]
        generated[modality] = np.zeros(dim)
        rule = rules.get(modality, "no rule")
        log_lines.append(
            f"[Generator] MOCK '{modality}' zeros, shape=({dim},). Rule: '{rule[:60]}'."
        )

    return {"generated_modalities": generated, "execution_log": log_lines}
