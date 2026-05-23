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

FAISS backend:
  IndexFlatIP on fused normalised modality vectors, built once at init.
  Falls back to FAISS-CPU if no GPU, then to the brute-force sklearn loop.

Usage in graph.py:
    from src.orchestrator.nodes.generator import make_generator_node, build_pool_index
    pool = build_pool_index(all_data, train_ids)
    generator = make_generator_node(pool, llm, metadata, n_candidates=3)
    builder.add_node("generator", generator)
"""

import logging
from collections import Counter, defaultdict

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
# FAISS availability
# ---------------------------------------------------------------------------

try:
    import faiss as _faiss_lib

    _FAISS_AVAILABLE = True
except ImportError:
    _faiss_lib = None
    _FAISS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Pool index construction
# ---------------------------------------------------------------------------


def _build_pool_entry(pid: str, patient: dict) -> dict:
    """Build a single retrieval pool entry with normalised feature vectors.

    Accepts any non-empty array regardless of size so that cohort-specific
    dims (e.g. LUSC methylation = 16206 vs MODALITY_DIMS = 16166) are not
    silently dropped.
    """
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
        if arr.size == 0:
            continue
        entry["features"][mod] = arr
        norm = np.linalg.norm(arr)
        entry["features_norm"][mod] = arr / norm if norm > 0 else arr
    return entry


def build_pool_index(
    raw_data: dict,
    patient_ids: list[str],
    cohort_map: dict[str, str] | None = None,
) -> list[dict]:
    """
    Precompute a retrieval index from training patient IDs.
    Called once at graph build time.

    cohort_map: optional dict[patient_id → cohort_label].  When provided,
    dim sanitization is performed independently per cohort so that cohort-
    specific feature sizes (e.g. LUSC clinical=63 vs LUAD clinical=56) are
    not incorrectly stripped by the dominant-dim logic.
    """
    pool = []
    for pid in patient_ids:
        patient = load_patient(pid, raw_data)
        if patient is None:
            continue
        pool.append(_build_pool_entry(pid, patient))

    _sanitize_pool_feature_dims(pool, cohort_map=cohort_map)
    return pool


def _sanitize_pool_feature_dims(
    pool: list[dict],
    cohort_map: dict[str, str] | None = None,
) -> None:
    """Enforce per-modality dimensional consistency within the retrieval pool.

    When cohort_map is provided, sanitization runs independently per cohort so
    that cohort-specific dims (e.g. LUSC methylation=16206 vs LUAD=16166) are
    not removed by the dominant-dim logic applied across the full pool.
    """
    if cohort_map is not None:
        by_cohort: dict[str, list[dict]] = defaultdict(list)
        for entry in pool:
            cohort = cohort_map.get(entry["patient_id"], "unknown")
            by_cohort[cohort].append(entry)
        for cohort, entries in by_cohort.items():
            logger.info("[Generator] Sanitizing pool dims for cohort '%s' (%d entries)", cohort, len(entries))
            _sanitize_pool_feature_dims(entries, cohort_map=None)
        return

    dim_counts: dict[str, Counter] = {mod: Counter() for mod in MODALITY_KEYS}
    for entry in pool:
        for mod, arr in entry["features"].items():
            dim_counts[mod][int(np.asarray(arr).size)] += 1

    expected_dims: dict[str, int] = {}
    for mod in MODALITY_KEYS:
        if dim_counts[mod]:
            expected_dims[mod] = dim_counts[mod].most_common(1)[0][0]

    if expected_dims:
        logger.info("[Generator] Pool expected dims (dominant): %s", expected_dims)

    mismatches: dict[str, list[tuple[str, tuple[int, ...], int]]] = defaultdict(list)
    for entry in pool:
        pid = entry["patient_id"]
        for mod in list(entry["features"].keys()):
            expected = expected_dims.get(mod)
            if expected is None:
                continue
            feat = np.asarray(entry["features"][mod], dtype=np.float32).flatten()
            if feat.shape != (expected,):
                mismatches[mod].append((pid, feat.shape, expected))
                entry["features"].pop(mod, None)
                entry["features_norm"].pop(mod, None)

    for mod, rows in mismatches.items():
        logger.warning(
            "[Generator] Pool dim mismatch for modality '%s': removed %d malformed "
            "entries (showing up to 10): %s",
            mod,
            len(rows),
            rows[:10],
        )


# ---------------------------------------------------------------------------
# FAISS index construction
# ---------------------------------------------------------------------------


def _detect_pool_dims(pool: list[dict]) -> dict[str, int]:
    """Infer actual modality dims from pool entries (handles cohort-specific sizes)."""
    dims = dict(MODALITY_DIMS)
    for entry in pool:
        for mod, arr in entry["features"].items():
            dims[mod] = int(arr.size)
    return dims


def _pool_fused_offsets(
    pool_dims: dict[str, int],
) -> tuple[dict[str, tuple[int, int]], int]:
    """Compute (start, end) byte offsets and total dim for the fused feature vector."""
    offsets: dict[str, tuple[int, int]] = {}
    offset = 0
    for mod in MODALITY_KEYS:
        dim = pool_dims.get(mod, 0)
        if dim > 0:
            offsets[mod] = (offset, offset + dim)
            offset += dim
    return offsets, offset


def _make_fused_vec(entry: dict, offsets: dict, total_dim: int) -> np.ndarray:
    """Concatenate all normalised modality vectors for a pool entry (zeros for missing)."""
    vec = np.zeros(total_dim, dtype=np.float32)
    for mod, (start, end) in offsets.items():
        if mod in entry["features_norm"]:
            src = entry["features_norm"][mod]
            vec[start : start + src.size] = src
    return vec


def _try_move_to_gpu(index):
    """Attempt to move a FAISS CPU index to GPU; return (index, on_gpu)."""
    try:
        res = _faiss_lib.StandardGpuResources()
        gpu_index = _faiss_lib.index_cpu_to_gpu(res, 0, index)
        return gpu_index, True
    except Exception:
        return index, False


def _build_faiss_index(
    pool: list[dict],
    pool_dims: dict[str, int],
) -> dict:
    """
    Build per-target-modality FAISS IndexFlatIP from the pool.

    Each index operates in the fused normalised feature space
    (all modalities concatenated, zeros where absent). At query time
    the modality weights are embedded in the query vector so that the
    inner product equals the weighted cosine similarity used by the
    brute-force fallback.

    Returns a dict with keys:
        indices       – {target_mod: faiss_index | None}
        pool_by_mod   – {target_mod: [pool_entry, ...]}
        offsets       – {mod: (start, end)} in fused vector
        total_dim     – int
        on_gpu        – bool
    """
    offsets, total_dim = _pool_fused_offsets(pool_dims)
    indices: dict = {}
    pool_by_mod: dict = {}
    on_gpu = False

    for mod in MODALITY_KEYS:
        entries = [e for e in pool if mod in e["features"]]
        if not entries:
            indices[mod] = None
            pool_by_mod[mod] = []
            continue

        if _FAISS_AVAILABLE and total_dim > 0:
            vecs = np.stack(
                [_make_fused_vec(e, offsets, total_dim) for e in entries],
            ).astype(np.float32)
            index = _faiss_lib.IndexFlatIP(total_dim)
            index, gpu = _try_move_to_gpu(index)
            if gpu:
                on_gpu = True
            index.add(vecs)
            indices[mod] = index
        else:
            indices[mod] = None

        pool_by_mod[mod] = entries

    backend = "GPU" if on_gpu else ("CPU" if _FAISS_AVAILABLE else "sklearn")
    logger.info(
        "[Generator] FAISS index built — backend=%s, total_dim=%d", backend, total_dim
    )
    return {
        "indices": indices,
        "pool_by_mod": pool_by_mod,
        "offsets": offsets,
        "total_dim": total_dim,
        "on_gpu": on_gpu,
    }


# ---------------------------------------------------------------------------
# k-NN retrieval — brute-force (sklearn fallback)
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
    entries_with_mod = [e for e in pool if target_modality in e["features"]]
    dim = (
        entries_with_mod[0]["features"][target_modality].size
        if entries_with_mod
        else MODALITY_DIMS[target_modality]
    )
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
    """Weighted average of one k-neighbour chunk → (reconstruction, info).
    
    Defensive: skips pool entries whose target_modality vector does not match
    the expected dimensionality. Logs a warning for each skipped entry so the
    incident is auditable from the orchestrator execution log.
    """
    # First pass: filter out entries with wrong dimensionality
    valid_chunk = []
    skipped = []
    for sim, entry in chunk:
        feat = entry["features"].get(target_modality)
        if feat is None:
            skipped.append((entry["patient_id"], "missing"))
            continue
        feat_arr = np.asarray(feat)
        if feat_arr.shape != (dim,):
            skipped.append((entry["patient_id"], f"shape={feat_arr.shape}"))
            continue
        valid_chunk.append((sim, entry))
 
    if skipped:
        logger.warning(
            "[Generator] _average_chunk: skipped %d/%d neighbours for "
            "modality '%s' (expected dim=%d). Skipped: %s",
            len(skipped), len(chunk), target_modality, dim, skipped[:5],
        )
 
    if not valid_chunk:
        # All neighbours malformed: return zero-fill rather than crash
        logger.error(
            "[Generator] _average_chunk: no valid neighbours for modality "
            "'%s' (expected dim=%d). Returning zero-fill candidate.",
            target_modality, dim,
        )
        return np.zeros(dim, dtype=np.float32), []
 
    # Standard weighted average on the valid subset
    weights = np.array([max(sim, 0.0) for sim, _ in valid_chunk], dtype=np.float32)
    weight_sum = weights.sum()
    if weight_sum == 0:
        weights = np.ones(len(valid_chunk), dtype=np.float32) / len(valid_chunk)
    else:
        weights /= weight_sum
 
    result = np.zeros(dim, dtype=np.float32)
    info: list[dict] = []
    for w, (sim, entry) in zip(weights, valid_chunk):
        result += w * np.asarray(entry["features"][target_modality], dtype=np.float32)
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
    Retrieve N candidate reconstructions for target_modality (brute-force).

    Fetches top k*N neighbours, splits them into N consecutive chunks of k,
    and computes a similarity-weighted average within each chunk. This gives
    N diverse candidates spanning from the best-matched to lower-matched
    neighbours in the pool.

    Returns:
        candidates:    list of N np.ndarray of shape (actual dim,)
        neighbor_info: neighbour metadata for the first (best) candidate
    """
    entries_with_mod = [e for e in pool if target_modality in e["features"]]
    dim = (
        entries_with_mod[0]["features"][target_modality].size
        if entries_with_mod
        else MODALITY_DIMS[target_modality]
    )
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
# k-NN retrieval — FAISS path
# ---------------------------------------------------------------------------


def _build_faiss_query(
    query_features: dict,
    offsets: dict,
    total_dim: int,
    modality_weights: dict | None,
) -> np.ndarray:
    """
    Build the fused query vector for FAISS IndexFlatIP.

    For each available modality m, writes w_m * normalised_vec into the
    corresponding slice of a zero-padded vector of length total_dim.
    Inner product of this query with an unweighted pool entry equals:
        sum_m  w_m * cosine_sim(query_m, pool_m)
    which is exactly _compute_similarity with the same weights.
    """
    vec = np.zeros(total_dim, dtype=np.float32)
    for mod, arr in query_features.items():
        if mod not in offsets:
            continue
        start, end = offsets[mod]
        w = modality_weights.get(mod, 1.0) if modality_weights else 1.0
        norm_arr = _normalize(arr)
        slice_len = end - start
        vec[start : start + min(norm_arr.size, slice_len)] = norm_arr[:slice_len] * w
    return vec


def _knn_retrieve_candidates_faiss(
    query_features: dict,
    target_modality: str,
    faiss_data: dict,
    k: int = 5,
    n_candidates: int = 1,
    exclude_pid: str | None = None,
    modality_weights: dict | None = None,
) -> tuple[list[np.ndarray], list[dict]]:
    """
    FAISS-based N-candidate retrieval replacing the brute-force inner loop.

    Uses IndexFlatIP (exact inner product) on fused normalised vectors, so
    results are mathematically identical to the brute-force path within FP
    rounding. Falls back gracefully when the index for this modality is None.
    """
    index = faiss_data["indices"].get(target_modality)
    entries = faiss_data["pool_by_mod"].get(target_modality, [])
    offsets = faiss_data["offsets"]
    total_dim = faiss_data["total_dim"]

    if index is None or not entries:
        # Fallback: brute-force for this modality
        return _knn_retrieve_candidates(
            query_features=query_features,
            target_modality=target_modality,
            pool=list(entries),
            k=k,
            n_candidates=n_candidates,
            exclude_pid=exclude_pid,
            modality_weights=modality_weights,
        )

    dim = entries[0]["features"][target_modality].size

    query_vec = _build_faiss_query(query_features, offsets, total_dim, modality_weights)
    # Search full index to guarantee correctness. The per-entry normalization step
    # (normalising by each entry's available modality weights) can significantly
    # reorder the raw FAISS ranking, so truncating candidates upfront risks excluding
    # the correct top-k. For production pools with millions of entries, full search
    # is still much faster than sklearn brute-force; for smaller pools, correctness
    # is the priority and the cost is negligible.
    n_search = len(entries)

    distances, idxs = index.search(query_vec.reshape(1, -1), n_search)
    distances = distances[0]
    idxs = idxs[0]

    results: list[tuple[float, dict]] = []
    for sim, idx in zip(distances, idxs):
        if idx < 0 or int(idx) >= len(entries):
            continue
        entry = entries[int(idx)]
        if exclude_pid and entry["patient_id"] == exclude_pid:
            continue

        # Normalize FAISS distance to match _compute_similarity: average instead of sum.
        # FAISS IndexFlatIP returns sum(w_m * sim_m) over shared modalities, but
        # _compute_similarity returns sum(w_m * sim_m) / sum(w_m). We must normalize
        # per-entry because each entry has different available modalities.
        shared_modalities = [m for m in query_features if m in entry["features_norm"]]
        total_weight = sum(
            (modality_weights.get(m, 1.0) if modality_weights else 1.0)
            for m in shared_modalities
        )
        normalized_sim = float(sim) / total_weight if total_weight > 0 else float(sim)

        results.append((normalized_sim, entry))

    if not results:
        return [np.zeros(dim, dtype=np.float32)] * n_candidates, []

    results.sort(key=lambda x: x[0], reverse=True)
    top_kn = results[: k * n_candidates]

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
        if arr.size > 0:
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
    cohort_map: dict[str, str] | None = None,
):
    """
    Returns a Generator closure following AFM2's Generation Agent.

    For each missing modality:
        1. Calls LLM to interpret refined guidance into modality weights
        2. Computes weighted cosine similarity on shared modalities via FAISS
           IndexFlatIP (GPU→CPU→sklearn fallback chain)
        3. Retrieves top-k*N neighbors split into N candidate reconstructions
        4. Stores N candidates in generation_candidates for the Verifier

    On self-refinement retries, correction hints from the Verifier are
    included in the LLM prompt to adjust the retrieval strategy.

    Args:
        pool:         Precomputed pool index from build_pool_index().
        llm:          LLM client (optional — falls back to uniform weights).
        metadata:     Feature name metadata (optional).
        n_candidates: Number of candidates to produce per modality (default 3).
        cohort_map:   Optional dict[patient_id → cohort] to filter pool by cohort.
    """
    # Detect cohort-specific dims from pool (e.g. LUSC methylation=16206 vs 16166)
    pool_dims = _detect_pool_dims(pool)

    # Build FAISS index once at init (GPU if available, else CPU, else None)
    faiss_data = _build_faiss_index(pool, pool_dims)
    _use_faiss = _FAISS_AVAILABLE and any(
        v is not None for v in faiss_data["indices"].values()
    )

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

        # Filter pool by cohort if cohort_map is available
        cohort = state.get("cohort")
        pool_filtered = pool
        if cohort_map and cohort:
            pool_filtered = [
                e for e in pool if cohort_map.get(e["patient_id"]) == cohort
            ]
            if not pool_filtered:
                pool_filtered = pool

        if not query_features:
            zero_candidates = {
                mod: [
                    np.zeros(pool_dims.get(mod, MODALITY_DIMS[mod]), dtype=np.float32)
                ]
                * n_candidates
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

            if _use_faiss:
                candidates, neighbors = _knn_retrieve_candidates_faiss(
                    query_features=query_features,
                    target_modality=modality,
                    faiss_data=faiss_data,
                    k=k,
                    n_candidates=n_candidates,
                    exclude_pid=pid,
                    modality_weights=modality_weights,
                )
            else:
                candidates, neighbors = _knn_retrieve_candidates(
                    query_features=query_features,
                    target_modality=modality,
                    pool=pool_filtered,
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
