"""
FAISS k-NN benchmark: sklearn-CPU vs FAISS-CPU vs FAISS-GPU.

Loads LUAD data (stable reference cohort), builds a pool from training
patients, then times each retrieval backend on up to 100 query patients.
Correctness is verified by comparing FAISS results to the brute-force
sklearn baseline within FP tolerance.

Output:
    results/benchmarks/faiss_comparison.json

Usage:
    python -m src.evaluation.benchmark_faiss
    python -m src.evaluation.benchmark_faiss --cohort lusc --n-queries 100
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("data/extracted/cache_data")
SPLITS_DIR = DATA_DIR / "splits"
RESULTS_DIR = Path("results/benchmarks")

_SPLIT_FILES = {
    "luad": "tcga_luad_DSS_k3_r1_test0.2_val0.2_seed42.json",
    "lusc": "tcga_lusc_DSS_k5_r1_test0.2_val0.2_seed42.json",
}

_FP_TOL = 1e-4  # relative tolerance for correctness check


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_data(cohort: str):
    from src.data_loader import load_raw_data, load_split, load_split_patients

    raw_data, _ = load_raw_data(DATA_DIR, cohort)
    train_ids, val_ids, _ = load_split(SPLITS_DIR, _SPLIT_FILES[cohort])
    train_patients = load_split_patients(train_ids, raw_data)
    query_patients = load_split_patients(val_ids, raw_data)
    return raw_data, train_ids, query_patients


def _build_query_features(patient: dict) -> dict:
    """Extract available modality arrays from a patient dict."""
    from src.data_loader import MODALITY_KEYS

    features = {}
    for mod in MODALITY_KEYS:
        val = patient.get(mod)
        if val is not None:
            arr = np.array(val).flatten().astype(np.float32)
            if arr.size > 0:
                features[mod] = arr
    return features


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def _time_sklearn(pool, query_patients, n_queries: int, k: int, n_candidates: int):
    from src.orchestrator.nodes.generator import _knn_retrieve_candidates
    from src.data_loader import MODALITY_KEYS

    times = []
    results = {}

    for patient in query_patients[:n_queries]:
        pid = patient["patient_id"]
        missing = patient["missing_modalities"]
        query_features = _build_query_features(patient)
        if not query_features or not missing:
            continue

        t0 = time.perf_counter()
        patient_results = {}
        for mod in missing:
            cands, _ = _knn_retrieve_candidates(
                query_features=query_features,
                target_modality=mod,
                pool=pool,
                k=k,
                n_candidates=n_candidates,
                exclude_pid=pid,
            )
            patient_results[mod] = [c.tolist() for c in cands]
        elapsed = time.perf_counter() - t0

        times.append(elapsed)
        results[pid] = patient_results

    return times, results


def _time_faiss(
    pool,
    pool_dims,
    faiss_data,
    query_patients,
    n_queries: int,
    k: int,
    n_candidates: int,
):
    from src.orchestrator.nodes.generator import _knn_retrieve_candidates_faiss

    times = []
    results = {}

    for patient in query_patients[:n_queries]:
        pid = patient["patient_id"]
        missing = patient["missing_modalities"]
        query_features = _build_query_features(patient)
        if not query_features or not missing:
            continue

        t0 = time.perf_counter()
        patient_results = {}
        for mod in missing:
            cands, _ = _knn_retrieve_candidates_faiss(
                query_features=query_features,
                target_modality=mod,
                faiss_data=faiss_data,
                k=k,
                n_candidates=n_candidates,
                exclude_pid=pid,
            )
            patient_results[mod] = [c.tolist() for c in cands]
        elapsed = time.perf_counter() - t0

        times.append(elapsed)
        results[pid] = patient_results

    return times, results


# ---------------------------------------------------------------------------
# Correctness check
# ---------------------------------------------------------------------------


def _check_correctness(
    sklearn_results: dict, faiss_results: dict, label: str
) -> dict:
    n_checked = 0
    n_mismatched = 0
    max_diff = 0.0

    for pid in sklearn_results:
        if pid not in faiss_results:
            continue
        for mod in sklearn_results[pid]:
            if mod not in faiss_results[pid]:
                continue
            s_cands = np.array(sklearn_results[pid][mod])
            f_cands = np.array(faiss_results[pid][mod])
            if s_cands.shape != f_cands.shape:
                n_mismatched += 1
                continue
            diff = np.max(np.abs(s_cands - f_cands))
            max_diff = max(max_diff, float(diff))
            if diff > _FP_TOL:
                n_mismatched += 1
            n_checked += 1

    status = "PASS" if n_mismatched == 0 else "FAIL"
    logger.info(
        "[%s] correctness: %s | checked=%d mismatched=%d max_diff=%.2e",
        label,
        status,
        n_checked,
        n_mismatched,
        max_diff,
    )
    return {
        "status": status,
        "n_checked": n_checked,
        "n_mismatched": n_mismatched,
        "max_abs_diff": round(max_diff, 8),
    }


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------


def run_benchmark(cohort: str = "luad", n_queries: int = 100, k: int = 5, n_candidates: int = 3):
    try:
        import faiss as _faiss_lib
        faiss_available = True
    except ImportError:
        faiss_available = False
        logger.warning("FAISS not installed — only sklearn-CPU will be timed.")

    logger.info("Loading %s data...", cohort.upper())
    raw_data, train_ids, query_patients = _load_data(cohort)

    from src.orchestrator.nodes.generator import (
        build_pool_index,
        _detect_pool_dims,
        _build_faiss_index,
    )

    logger.info("Building pool from %d training patients...", len(train_ids))
    pool = build_pool_index(raw_data, train_ids)
    pool_dims = _detect_pool_dims(pool)
    logger.info("Pool dims: %s", pool_dims)

    # ---- sklearn-CPU ----
    logger.info("Timing sklearn-CPU (%d queries, k=%d, N=%d)...", n_queries, k, n_candidates)
    sk_times, sk_results = _time_sklearn(pool, query_patients, n_queries, k, n_candidates)

    # ---- FAISS ----
    faiss_cpu_times, faiss_cpu_results = [], {}
    faiss_gpu_times, faiss_gpu_results = [], {}
    faiss_cpu_correct = {"status": "N/A"}
    faiss_gpu_correct = {"status": "N/A"}
    on_gpu = False

    if faiss_available:
        logger.info("Building FAISS-CPU index...")
        # Force CPU by patching _try_move_to_gpu
        from src.orchestrator.nodes import generator as gen_mod
        _orig_try_gpu = gen_mod._try_move_to_gpu

        def _no_gpu(index):
            return index, False

        gen_mod._try_move_to_gpu = _no_gpu
        faiss_data_cpu = _build_faiss_index(pool, pool_dims)
        gen_mod._try_move_to_gpu = _orig_try_gpu

        logger.info("Timing FAISS-CPU (%d queries)...", n_queries)
        faiss_cpu_times, faiss_cpu_results = _time_faiss(
            pool, pool_dims, faiss_data_cpu, query_patients, n_queries, k, n_candidates
        )
        faiss_cpu_correct = _check_correctness(sk_results, faiss_cpu_results, "FAISS-CPU")

        logger.info("Building FAISS-GPU index (falls back to CPU if no GPU)...")
        faiss_data_gpu = _build_faiss_index(pool, pool_dims)
        on_gpu = faiss_data_gpu["on_gpu"]

        logger.info("Timing FAISS-%s (%d queries)...", "GPU" if on_gpu else "CPU-fallback", n_queries)
        faiss_gpu_times, faiss_gpu_results = _time_faiss(
            pool, pool_dims, faiss_data_gpu, query_patients, n_queries, k, n_candidates
        )
        faiss_gpu_correct = _check_correctness(sk_results, faiss_gpu_results, "FAISS-GPU")

    # ---- Summarise ----
    def _stats(times):
        if not times:
            return {}
        return {
            "n_queries": len(times),
            "mean_s": round(float(np.mean(times)), 6),
            "median_s": round(float(np.median(times)), 6),
            "p95_s": round(float(np.percentile(times, 95)), 6),
            "total_s": round(float(np.sum(times)), 4),
        }

    report = {
        "cohort": cohort.upper(),
        "k": k,
        "n_candidates": n_candidates,
        "pool_size": len(pool),
        "pool_dims": pool_dims,
        "faiss_available": faiss_available,
        "faiss_on_gpu": on_gpu,
        "sklearn_cpu": _stats(sk_times),
        "faiss_cpu": {**_stats(faiss_cpu_times), "correctness": faiss_cpu_correct},
        "faiss_gpu": {**_stats(faiss_gpu_times), "correctness": faiss_gpu_correct},
    }

    if sk_times and faiss_cpu_times:
        speedup = np.mean(sk_times) / max(np.mean(faiss_cpu_times), 1e-9)
        report["faiss_cpu"]["speedup_vs_sklearn"] = round(float(speedup), 2)
    if sk_times and faiss_gpu_times:
        speedup = np.mean(sk_times) / max(np.mean(faiss_gpu_times), 1e-9)
        report["faiss_gpu"]["speedup_vs_sklearn"] = round(float(speedup), 2)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "faiss_comparison.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Results saved to %s", out_path)

    # Print summary
    print("\n=== FAISS Benchmark Summary ===")
    print(f"Cohort: {report['cohort']}  |  Pool: {report['pool_size']} patients")
    print(f"k={k}  |  N={n_candidates}  |  Queries: {len(sk_times)}")
    print(f"sklearn-CPU   mean={report['sklearn_cpu'].get('mean_s', 'N/A'):.4f}s")
    if faiss_available:
        fc = report["faiss_cpu"]
        fg = report["faiss_gpu"]
        print(
            f"FAISS-CPU     mean={fc.get('mean_s', 'N/A'):.4f}s  "
            f"speedup={fc.get('speedup_vs_sklearn', 'N/A')}x  "
            f"correct={fc['correctness']['status']}"
        )
        label = "FAISS-GPU" if on_gpu else "FAISS-CPU(2)"
        print(
            f"{label:<14}mean={fg.get('mean_s', 'N/A'):.4f}s  "
            f"speedup={fg.get('speedup_vs_sklearn', 'N/A')}x  "
            f"correct={fg['correctness']['status']}"
        )
    print(f"\nJSON: {out_path}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark FAISS vs sklearn k-NN retrieval.")
    parser.add_argument("--cohort", default="luad", choices=["luad", "lusc"])
    parser.add_argument("--n-queries", type=int, default=100)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--n-candidates", type=int, default=3)
    args = parser.parse_args()

    run_benchmark(
        cohort=args.cohort,
        n_queries=args.n_queries,
        k=args.k,
        n_candidates=args.n_candidates,
    )
