"""
Profile FAISS vs sklearn retrieval to identify performance bottlenecks.

Usage:
    python -m cProfile -s cumulative src/evaluation/profile_faiss.py > profile.txt
    # or with line_profiler:
    kernprof -l -v src/evaluation/profile_faiss.py
"""

from pathlib import Path

import numpy as np

DATA_DIR = Path("data/extracted/cache_data")
SPLITS_DIR = DATA_DIR / "splits"


def profile_retrieval():
    """Profile a single retrieval cycle on LUSC with full search."""
    from src.data_loader import load_raw_data, load_split, load_split_patients
    from src.orchestrator.nodes.generator import (
        _build_faiss_index,
        _detect_pool_dims,
        _knn_retrieve_candidates,
        _knn_retrieve_candidates_faiss,
        build_pool_index,
    )

    cohort = "lusc"
    split_file = "tcga_lusc_DSS_k5_r1_test0.2_val0.2_seed42.json"

    print("[Profile] Loading data...")
    raw_data, _ = load_raw_data(DATA_DIR, cohort)
    train_ids, val_ids, _ = load_split(SPLITS_DIR, split_file)
    train_patients = load_split_patients(train_ids, raw_data)
    query_patients = load_split_patients(val_ids, raw_data)

    print(f"[Profile] Building pool from {len(train_ids)} patients...")
    pool = build_pool_index(raw_data, train_ids)
    pool_dims = _detect_pool_dims(pool)

    print(f"[Profile] Pool dims: {pool_dims}")
    print("[Profile] Building FAISS index...")
    faiss_data = _build_faiss_index(pool, pool_dims)

    # Get a test query
    test_patient = query_patients[0]
    pid = test_patient["patient_id"]
    missing = test_patient["missing_modalities"]

    from src.data_loader import MODALITY_KEYS

    query_features = {}
    for mod in MODALITY_KEYS:
        val = test_patient.get(mod)
        if val is not None:
            arr = np.array(val).flatten().astype(np.float32)
            if arr.size > 0:
                query_features[mod] = arr

    target_mod = missing[0] if missing else "transcriptomics"
    k, n_candidates = 5, 3

    print(
        f"\n[Profile] Query: pid={pid}, target_mod={target_mod}, k={k}, N={n_candidates}"
    )
    print(f"[Profile] Query has {len(query_features)} modalities")
    print(
        f"[Profile] Pool size for {target_mod}: {len(faiss_data['pool_by_mod'].get(target_mod, []))}"
    )

    print("\n=== SKLEARN BRUTE-FORCE ===")
    sklearn_result = _knn_retrieve_candidates(
        query_features=query_features,
        target_modality=target_mod,
        pool=pool,
        k=k,
        n_candidates=n_candidates,
        exclude_pid=pid,
    )
    print(
        f"✓ sklearn result: {len(sklearn_result[0])} candidates, {len(sklearn_result[1])} neighbors"
    )

    print("\n=== FAISS FULL SEARCH ===")
    faiss_result = _knn_retrieve_candidates_faiss(
        query_features=query_features,
        target_modality=target_mod,
        faiss_data=faiss_data,
        k=k,
        n_candidates=n_candidates,
        exclude_pid=pid,
    )
    print(
        f"✓ FAISS result: {len(faiss_result[0])} candidates, {len(faiss_result[1])} neighbors"
    )

    # Validate
    s_cands = np.array(sklearn_result[0])
    f_cands = np.array(faiss_result[0])
    diff = np.max(np.abs(s_cands - f_cands))
    print(f"\n✓ Correctness: max_diff = {diff:.2e}")


if __name__ == "__main__":
    profile_retrieval()
