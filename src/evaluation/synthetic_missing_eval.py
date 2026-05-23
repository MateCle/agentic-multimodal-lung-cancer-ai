"""
src/evaluation/synthetic_missing_eval.py

Synthetic Missing-Modality Reconstruction Evaluation
====================================================

Purpose
-------
The end-to-end evaluation in `evaluate_orchestrator.py` measures the
downstream C-index after the orchestrator imputes a missing modality.
This is the right outcome metric, but it conflates two things:

  1. Whether the orchestrator reconstructs a plausible feature vector
     for the missing modality (reconstruction quality), and
  2. Whether the downstream predictor can use that reconstruction
     (predictor compatibility / train-test alignment).

When the downstream metric disappoints, we cannot tell which of the
two failed. This script isolates (1) by giving the orchestrator a
controllable, ground-truth-known reconstruction task:

  • Take patients from the test split who have ALL four modalities
    present (i.e. for whom ground truth exists for every modality).
  • For each such patient, for each modality m:
      - Mask m (set to None) and label the patient as "missing m".
      - Run the orchestrator. The Generator reconstructs m.
      - Compare the reconstructed vector to the held-out ground truth
        with cosine similarity, MSE, and (optional) per-modality
        block-level diagnostics.
  • Also collect the Verifier post-hoc overall_score for each
    reconstruction, so the script can report the correlation between
    the Verifier's internal quality score and the *actual* recon
    quality measured against ground truth. This validates (or refutes)
    the Verifier as a useful signal.

Outputs
-------
For each cohort:

  results/evaluation/synthetic_missing/<run_id>/
    per_patient_<cohort>.csv         # one row per (patient, modality)
    summary_<cohort>.json            # aggregate stats
    verifier_correlation_<cohort>.png  # scatter, optional
    verifier_correlation_<cohort>.json # Pearson + Spearman + slope

Aggregate stats reported in summary_*.json:

  reconstruction:
    per_modality: {clinical, transcriptomics, wsi, methylation}:
      n_samples
      cosine_similarity_mean / std / median
      mse_mean / std / median
      normalized_mse_mean    # MSE / variance of ground-truth modality
    overall:
      same stats, pooled

  verifier_validation:
    pearson_r between overall_score and cosine_similarity
    spearman_r between overall_score and cosine_similarity
    pearson_r between overall_score and -MSE
    n_paired

Usage
-----
    python -m src.evaluation.synthetic_missing_eval \\
        --cohort luad --output-dir results/evaluation/synthetic_missing/

    # Limit to N patients per cohort for a quick run:
    python -m src.evaluation.synthetic_missing_eval --cohort lusc --max-patients 20

Notes
-----
This script does not call the Predictor — it only invokes the orchestrator
up to (and including) the Verifier. The Predictor's risk score is not
relevant for reconstruction quality. We extract the reconstructed
modality and the Verifier score from the final PatientState dict.

The script is deliberately tolerant of partial failures: if a single
patient/modality combination fails (LLM JSON parse error, k-NN pool
miss, etc.) it is logged and skipped.
"""

import argparse
import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

DATA_DIR = Path("data/extracted/cache_data")
SPLITS_DIR = DATA_DIR / "splits"
DEFAULT_OUTPUT_DIR = Path("results/evaluation/synthetic_missing")

_SPLIT_FILES = {
    "luad": "tcga_luad_DSS_k3_r1_test0.2_val0.2_seed42.json",
    "lusc": "tcga_lusc_DSS_k5_r1_test0.2_val0.2_seed42.json",
}

_MODALITY_KEYS = ["clinical", "transcriptomics", "wsi", "methylation"]


# ---------------------------------------------------------------------------
# State builder (mirrors evaluate_orchestrator._build_initial_state, but with
# a forced availability vector — we manually mask the target modality)
# ---------------------------------------------------------------------------


def _build_state_with_modality_masked(
    patient_id: str,
    patient: dict,
    cohort: str,
    masked_modality: str,
) -> dict:
    """Return an initial PatientState in which `masked_modality` is set to
    None and removed from available_modalities. All other modalities keep
    their ground-truth values."""
    state = {
        "user_query": (
            f"Predict survival for patient {patient_id} from the "
            f"{cohort.upper()} cohort."
        ),
        "parsed_query": {},
        "patient_id": patient_id,
        "cohort": cohort,
        "clinical": None,
        "transcriptomics": None,
        "wsi": None,
        "methylation": None,
        "available_modalities": [],
        "missing_modalities": [],
        "agent_summaries": {},
        "mining_rules": {},
        "guidance": {},
        "generation_candidates": {},
        "generated_modalities": {},
        "verification_scores": {},
        "verification_passed": False,
        "survival_prediction": None,
        "risk_class": "",
        "top_shap_features": [],
        "shap_feature_details": [],
        "source_map": {},
        "prediction_reliability": {},
        "clinical_report": "",
        "routing_decision": "",
        "execution_log": [],
        "correction_hints": {},
        "forced_missing_modalities": [masked_modality],
    }
    for m in _MODALITY_KEYS:
        if m == masked_modality:
            state[m] = None
            state["missing_modalities"].append(m)
        else:
            state[m] = patient[m]
            state["available_modalities"].append(m)
    return state


# ---------------------------------------------------------------------------
# Reconstruction metrics
# ---------------------------------------------------------------------------


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(f"cosine: shape mismatch {a.shape} vs {b.shape}")
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(f"mse: shape mismatch {a.shape} vs {b.shape}")
    return float(np.mean((a - b) ** 2))


def _normalized_mse(
    reconstructed: np.ndarray,
    ground_truth: np.ndarray,
    modality_variance: float,
) -> float:
    """MSE divided by the variance of the ground-truth modality across
    the test set. Removes the effect of differing modality scales
    (methylation values are bounded, transcriptomics aren't, etc.).
    """
    raw = _mse(reconstructed, ground_truth)
    if modality_variance == 0.0:
        return float("nan")
    return raw / modality_variance


# ---------------------------------------------------------------------------
# Verifier score extraction
# ---------------------------------------------------------------------------


def _extract_verifier_overall_score(
    final_state: dict, modality: str
) -> Optional[float]:
    """Pull the post-Verifier overall_score for the given modality from
    the final state. Tolerates several layout variants of
    verification_scores."""
    scores = final_state.get("verification_scores") or {}
    payload = scores.get(modality)
    if payload is None:
        return None
    if isinstance(payload, (int, float)):
        return float(payload)
    if isinstance(payload, dict):
        # Common layouts: {"overall_score": ...} or {"overall": ...}
        for key in ("overall_score", "overall", "score"):
            v = payload.get(key)
            if isinstance(v, (int, float)):
                return float(v)
    return None


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def _aggregate_modality_stats(rows: list[dict], modality: str) -> dict:
    """Compute per-modality stats from per-(patient, modality) rows."""
    rows_m = [r for r in rows if r["modality"] == modality]
    if not rows_m:
        return {"n_samples": 0}
    cos = np.array([r["cosine_similarity"] for r in rows_m], dtype=np.float64)
    mse = np.array([r["mse"] for r in rows_m], dtype=np.float64)
    nmse = np.array([r["normalized_mse"] for r in rows_m], dtype=np.float64)

    def stats(arr: np.ndarray) -> dict:
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return {"mean": float("nan"), "std": float("nan"), "median": float("nan")}
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
            "median": float(np.median(arr)),
        }

    return {
        "n_samples": len(rows_m),
        "cosine_similarity": stats(cos),
        "mse": stats(mse),
        "normalized_mse": stats(nmse),
    }


def _verifier_correlation(rows: list[dict]) -> dict:
    """Pearson + Spearman between Verifier overall_score and reconstruction
    quality. Reported on the pooled set across modalities."""
    pairs = [
        (r["verifier_overall_score"], r["cosine_similarity"], r["mse"])
        for r in rows
        if r["verifier_overall_score"] is not None
        and np.isfinite(r["cosine_similarity"])
        and np.isfinite(r["mse"])
    ]
    if len(pairs) < 5:
        return {
            "n_paired": len(pairs),
            "pearson_r_cos": None,
            "spearman_r_cos": None,
            "pearson_r_neg_mse": None,
            "note": "too few paired samples",
        }
    ver = np.array([p[0] for p in pairs], dtype=np.float64)
    cos = np.array([p[1] for p in pairs], dtype=np.float64)
    mse = np.array([p[2] for p in pairs], dtype=np.float64)

    def pearson(a, b):
        if np.std(a) == 0 or np.std(b) == 0:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    def spearman(a, b):
        ra = np.argsort(np.argsort(a))
        rb = np.argsort(np.argsort(b))
        return pearson(ra.astype(float), rb.astype(float))

    return {
        "n_paired": len(pairs),
        "pearson_r_cos": pearson(ver, cos),
        "spearman_r_cos": spearman(ver, cos),
        "pearson_r_neg_mse": pearson(ver, -mse),
    }


def run_synthetic_missing_eval(
    cohort: str,
    output_dir: Path,
    max_patients: Optional[int] = None,
    seed: int = 42,
) -> dict:
    """Main entry point. Returns the summary dict that is also written
    to disk as summary_<cohort>.json."""
    # Imports deferred so that --help works without a TCGA / LLM setup.
    from src.data_loader import (
        MODALITY_KEYS,
        load_raw_data,
        load_split,
        load_split_patients,
    )
    from src.orchestrator.graph import build_graph

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load data and splits ---
    raw_data, _meta = load_raw_data(DATA_DIR, cohort)
    train_ids, _val_ids, test_ids = load_split(SPLITS_DIR, _SPLIT_FILES[cohort])
    test_patients = load_split_patients(test_ids, raw_data)

    # --- Keep only test patients with ALL 4 modalities present ---
    complete = [
        p for p in test_patients if set(p["available_modalities"]) == set(MODALITY_KEYS)
    ]
    logger.info(
        "%s: %d test patients total, %d with all 4 modalities present.",
        cohort.upper(),
        len(test_patients),
        len(complete),
    )

    if max_patients is not None:
        rng = np.random.default_rng(seed)
        if len(complete) > max_patients:
            idx = rng.choice(len(complete), size=max_patients, replace=False)
            complete = [complete[i] for i in sorted(idx.tolist())]
            logger.info(
                "%s: subsampled to %d patients for this run.",
                cohort.upper(),
                len(complete),
            )

    if not complete:
        raise RuntimeError(
            f"No fully-complete test patients found for cohort {cohort}; "
            "synthetic missing experiment cannot run."
        )

    # --- Precompute per-modality variance for normalized_mse ---
    modality_variance: dict[str, float] = {}
    for m in MODALITY_KEYS:
        # Use the training-split ground truth for variance, to avoid
        # leakage from the test set being evaluated.
        train_patients = load_split_patients(train_ids, raw_data)
        vals = [
            np.asarray(p[m], dtype=np.float64).ravel()
            for p in train_patients
            if m in p["available_modalities"] and p[m] is not None
        ]
        if vals:
            stacked = np.concatenate(vals)
            modality_variance[m] = float(np.var(stacked))
        else:
            modality_variance[m] = 0.0
        logger.info(
            "%s/%s: training-set variance = %.4g",
            cohort.upper(),
            m,
            modality_variance[m],
        )

    # --- Build the orchestrator graph in deterministic mode ---
    # generator_temperature=0.0 ensures that for a given (patient, masked
    # modality) the reconstruction is reproducible across runs. This makes
    # the reconstruction quality numbers meaningful as a *system property*
    # rather than as a draw from a stochastic distribution.
    logger.info("Building orchestrator graph (Generator T=0)...")
    graph = build_graph(
        data_dir=DATA_DIR,
        model_name="coxnet",
        imputation="mice",
        train_patient_ids=train_ids,
        n_candidates=3,
        generator_temperature=0.0,
    )

    # --- Main loop: for each patient, mask each modality, reconstruct ---
    per_patient_rows: list[dict] = []
    for patient_idx, patient in enumerate(complete):
        pid = patient["patient_id"]
        logger.info(
            "[%d/%d] Patient %s — masking each modality",
            patient_idx + 1,
            len(complete),
            pid,
        )

        for masked_m in MODALITY_KEYS:
            ground_truth = patient[masked_m]
            if ground_truth is None:
                # Shouldn't happen since we filtered on complete, but skip
                # defensively.
                continue

            state = _build_state_with_modality_masked(
                patient_id=pid,
                patient=patient,
                cohort=cohort,
                masked_modality=masked_m,
            )

            try:
                final_state = graph.invoke(state)
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "  %s/mask=%s: orchestrator failed (%s) — skipping",
                    pid,
                    masked_m,
                    type(e).__name__,
                )
                continue

            # DEBUG: capture the exact routing and outputs for each masked run.
            logger.warning("  DEBUG final state keys: %s", list(final_state.keys()))
            logger.warning(
                "  DEBUG generated_modalities = %s",
                final_state.get("generated_modalities"),
            )
            logger.warning(
                "  DEBUG routing_decision = %s",
                final_state.get("routing_decision"),
            )
            logger.warning("  DEBUG execution_log:")
            for line in (final_state.get("execution_log") or []):
                logger.warning("    - %s", line)
            logger.warning(
                "  DEBUG missing_modalities (final) = %s",
                final_state.get("missing_modalities"),
            )
            logger.warning(
                "  DEBUG verification_passed = %s",
                final_state.get("verification_passed"),
            )

            gen = (final_state.get("generated_modalities") or {}).get(masked_m)
            if gen is None:
                logger.warning(
                    "  %s/mask=%s: no reconstructed array in final state — skipping",
                    pid,
                    masked_m,
                )
                continue

            try:
                cos = _cosine_similarity(gen, ground_truth)
                mse = _mse(gen, ground_truth)
                nmse = _normalized_mse(gen, ground_truth, modality_variance[masked_m])
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "  %s/mask=%s: metric failure (%s) — skipping", pid, masked_m, e
                )
                continue

            ver_score = _extract_verifier_overall_score(final_state, masked_m)
            verification_passed = bool(final_state.get("verification_passed", False))
            n_retries = sum(
                1
                for line in (final_state.get("execution_log") or [])
                if "retry" in str(line).lower()
            )

            row = {
                "patient_id": pid,
                "cohort": cohort,
                "modality": masked_m,
                "cosine_similarity": cos,
                "mse": mse,
                "normalized_mse": nmse,
                "verifier_overall_score": ver_score,
                "verification_passed": verification_passed,
                "n_retries": n_retries,
            }
            per_patient_rows.append(row)
            logger.info(
                "  %s/mask=%s: cos=%.3f  mse=%.4g  nmse=%.3f  ver=%s",
                pid,
                masked_m,
                cos,
                mse,
                nmse,
                f"{ver_score:.2f}" if ver_score is not None else "N/A",
            )

        if patient_idx + 1 == 3 and len(per_patient_rows) == 0:
            logger.error(
                "First 3 patients produced zero reconstructions. "
                "Aborting to save compute time. Check graph wiring and "
                "initial state."
            )
            raise RuntimeError("All early reconstructions failed -- aborting")

    # --- Persist per-patient CSV ---
    csv_path = output_dir / f"per_patient_{cohort}.csv"
    if per_patient_rows:
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(per_patient_rows[0].keys()))
            writer.writeheader()
            writer.writerows(per_patient_rows)
        logger.info("Wrote per-patient rows → %s", csv_path)

    # --- Aggregate stats ---
    per_modality = {
        m: _aggregate_modality_stats(per_patient_rows, m) for m in _MODALITY_KEYS
    }
    overall = _aggregate_modality_stats(
        [
            {**r, "modality": "_overall"}
            for r in [dict(r, modality="_overall") for r in per_patient_rows]
        ],
        "_overall",
    )
    verifier_corr = _verifier_correlation(per_patient_rows)

    summary = {
        "cohort": cohort,
        "n_patients_evaluated": len(complete),
        "n_reconstructions": len(per_patient_rows),
        "generator_temperature": 0.0,
        "modality_variance_train": modality_variance,
        "reconstruction": {
            "per_modality": per_modality,
            "overall": overall,
        },
        "verifier_validation": verifier_corr,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    summary_path = output_dir / f"summary_{cohort}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Wrote summary → %s", summary_path)

    # --- Brief stdout report ---
    print(f"\n=== Synthetic missing-modality reconstruction — {cohort.upper()} ===")
    print(f"  Patients evaluated: {len(complete)}")
    print(f"  Reconstructions:    {len(per_patient_rows)}")
    print("  Generator T:        0.0 (deterministic)")
    print()
    print(f"  {'Modality':<16}  {'N':>4}  {'cos.sim.':>10}  {'norm.MSE':>10}")
    print("  " + "-" * 50)
    for m in _MODALITY_KEYS:
        stats = per_modality[m]
        n = stats.get("n_samples", 0)
        if n == 0:
            print(f"  {m:<16}  {0:>4}  {'N/A':>10}  {'N/A':>10}")
            continue
        cos_mean = stats["cosine_similarity"]["mean"]
        cos_std = stats["cosine_similarity"]["std"]
        nmse_mean = stats["normalized_mse"]["mean"]
        print(f"  {m:<16}  {n:>4}  {cos_mean:>6.3f}±{cos_std:.3f}  {nmse_mean:>10.3f}")
    print()
    print("  Verifier ↔ reconstruction-quality correlation:")
    print(
        f"    Pearson  r(overall_score, cosine_sim) = {verifier_corr.get('pearson_r_cos')}"
    )
    print(
        f"    Spearman r(overall_score, cosine_sim) = {verifier_corr.get('spearman_r_cos')}"
    )
    print(
        f"    Pearson  r(overall_score, -MSE)       = {verifier_corr.get('pearson_r_neg_mse')}"
    )
    print()

    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Synthetic missing-modality reconstruction evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--cohort",
        required=True,
        choices=("luad", "lusc"),
        help="TCGA cohort to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for per_patient_<cohort>.csv and summary_<cohort>.json. "
        "Defaults to results/evaluation/synthetic_missing/<timestamp>/",
    )
    parser.add_argument(
        "--max-patients",
        type=int,
        default=None,
        help="Cap the number of complete patients evaluated. None = all.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for max-patients subsampling.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.output_dir is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_dir = DEFAULT_OUTPUT_DIR / run_id
    else:
        output_dir = args.output_dir

    run_synthetic_missing_eval(
        cohort=args.cohort,
        output_dir=output_dir,
        max_patients=args.max_patients,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
