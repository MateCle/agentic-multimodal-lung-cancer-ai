"""
Offline re-analysis of evaluation results.

Reads existing per_patient_*.csv files and recalculates C-index metrics
with FIX 1 (outlier filtering) and FIX 2 (subgroup C-index) applied,
without re-running the orchestrator graph.

Usage:
    python -m src.evaluation.reanalyze_results --cohort luad
    python -m src.evaluation.reanalyze_results --cohort lusc
    python -m src.evaluation.reanalyze_results --all
"""

import argparse
import csv
import json
import logging
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results/evaluation")
RISK_SCORE_OUTLIER_THRESHOLD = 100.0


def _safe_cindex(
    scores: list[float],
    events: list[bool],
    times: list[float],
    label: str = "",
) -> float:
    """Compute concordance_index_censored; returns nan on any failure."""
    try:
        from sksurv.metrics import concordance_index_censored  # type: ignore

        y_event = np.asarray(events, dtype=bool)
        y_time = np.asarray(times, dtype=float)
        risk = np.asarray(scores, dtype=float)
        if len(scores) < 2 or not any(y_event):
            logger.warning("C-index (%s): insufficient data.", label)
            return float("nan")
        result = concordance_index_censored(y_event, y_time, risk)
        return float(result[0])
    except Exception as exc:
        logger.warning("C-index (%s) failed: %s", label, exc)
        return float("nan")


def _round_or_null(value: float, decimals: int = 4):
    if np.isnan(value):
        return None
    return round(float(value), decimals)


def _compute_baseline_risk_offline(patient_id: str, patient_data: dict, pipeline, cohort: str) -> float:
    """
    Compute baseline risk (zero-fill missing modalities) for offline reanalysis.
    Uses same logic as evaluate_orchestrator._compute_baseline_risk but inlined.
    """
    try:
        from src.orchestrator.nodes.predictor import _apply_pipeline, _assemble_features

        state: dict = {
            "patient_id": patient_id,
            "cohort": cohort,
            "clinical": patient_data.get("clinical"),
            "transcriptomics": patient_data.get("transcriptomics"),
            "wsi": patient_data.get("wsi"),
            "methylation": patient_data.get("methylation"),
            "available_modalities": patient_data.get("available_modalities", []),
            "missing_modalities": patient_data.get("missing_modalities", []),
            "generated_modalities": {},
            "verification_scores": {},
            "verification_passed": False,
            "agent_summaries": {},
            "mining_rules": {},
            "guidance": {},
            "generation_candidates": {},
            "prediction_reliability": {},
            "user_query": "",
            "parsed_query": {},
            "clinical_report": "",
            "routing_decision": "",
            "execution_log": [],
            "correction_hints": {},
            "survival_prediction": None,
            "risk_class": "",
            "top_shap_features": [],
            "shap_feature_details": [],
            "source_map": {},
        }

        actual_dims = getattr(pipeline, "actual_dims", None)
        x_raw, _ = _assemble_features(state, actual_dims)
        x_pca = _apply_pipeline(x_raw, pipeline)
        return float(np.asarray(pipeline.model.predict_risk(x_pca)).flatten()[0])
    except Exception as exc:
        logger.warning("Baseline failed for %s: %s", patient_id, exc)
        return float("nan")


def reanalyze_cohort(cohort: str, imputation: str = "mice") -> dict:
    """
    Read per_patient_{cohort}.csv, apply fixes, recalculate C-index metrics.
    Also recompute baseline scores from original data.
    
    Args:
        cohort: 'luad' or 'lusc'
        imputation: 'mice' or 'knn_tuned' (affects which pipeline is loaded)
    
    Returns:
        summary_dict with recalculated metrics.
    """
    from src.baseline.pipeline import load_pipeline
    from src.data_loader import load_raw_data, load_split, load_split_patients
    
    # Load data and pipeline
    data_dir = Path("data/extracted/cache_data")
    splits_dir = data_dir / "splits"
    
    _split_files = {
        "luad": "tcga_luad_DSS_k3_r1_test0.2_val0.2_seed42.json",
        "lusc": "tcga_lusc_DSS_k5_r1_test0.2_val0.2_seed42.json",
    }
    
    split_file = _split_files.get(cohort)
    if split_file is None:
        logger.error("Unknown cohort: %s", cohort)
        return {}
    
    logger.info("Loading raw data for %s...", cohort.upper())
    raw_data, _ = load_raw_data(data_dir, cohort)
    
    logger.info("Loading split and test patients...")
    train_ids, _, test_ids = load_split(splits_dir, split_file)
    test_patients_dict = {p["patient_id"]: p for p in load_split_patients(test_ids, raw_data)}
    
    logger.info("Loading pipeline for %s with imputation=%s...", cohort.upper(), imputation)
    try:
        pipeline = load_pipeline(cohort, "coxnet", imputation)
    except Exception as exc:
        logger.warning("Could not load pipeline: %s", exc)
        pipeline = None
    
    csv_path = RESULTS_DIR / f"per_patient_{cohort}.csv"
    if not csv_path.exists():
        logger.error("CSV not found: %s", csv_path)
        return {}
    
    logger.info("Reading %s...", csv_path)
    
    # Parse CSV
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for record in reader:
            rows.append({
                "patient_id": record["patient_id"],
                "risk_score": float(record["risk_score"]),
                "event": int(record["event"]),
                "time": float(record["time"]),
                "n_missing": int(record["n_missing"]),
                "provenance": float(record["provenance"]),
                "mahal_pct": float(record["mahal_pct"]),
                "ci_width": float(record["ci_width"]),
            })
    
    n_test = len(rows)
    logger.info("Loaded %d patients.", n_test)
    
    # Compute baseline scores for all patients
    logger.info("Computing baseline (zero-fill) scores for all patients...")
    for row in rows:
        pid = row["patient_id"]
        if pid in test_patients_dict and pipeline is not None:
            baseline_score = _compute_baseline_risk_offline(
                pid, test_patients_dict[pid], pipeline, cohort
            )
            row["baseline_score"] = baseline_score
        else:
            row["baseline_score"] = float("nan")
    
    # --- FIX 1: Outlier filtering ---
    orchestrator_scores = [r["risk_score"] for r in rows]
    events = [bool(r["event"]) for r in rows]
    times = [r["time"] for r in rows]
    
    outlier_mask = [abs(score) <= RISK_SCORE_OUTLIER_THRESHOLD for score in orchestrator_scores]
    n_outliers_excluded = sum(1 for m in outlier_mask if not m)
    if n_outliers_excluded > 0:
        logger.warning("Excluding %d outlier patients (|risk_score| > %.1f).", 
                      n_outliers_excluded, RISK_SCORE_OUTLIER_THRESHOLD)
        for i, row in enumerate(rows):
            if not outlier_mask[i]:
                logger.warning("  Outlier: %s (score=%.4f)", row["patient_id"], row["risk_score"])
    
    # Apply mask
    orchestrator_scores_filtered = [s for s, m in zip(orchestrator_scores, outlier_mask) if m]
    events_filtered = [e for e, m in zip(events, outlier_mask) if m]
    times_filtered = [t for t, m in zip(times, outlier_mask) if m]
    rows_filtered = [r for r, m in zip(rows, outlier_mask) if m]
    
    # --- Recalculate overall C-index (filtered) ---
    orchestrator_cindex = _safe_cindex(orchestrator_scores_filtered, events_filtered, times_filtered, 
                                      f"orchestrator_reanalyzed_{cohort}")
    
    # --- FIX 2: Subgroup C-index (complete vs. missing) ---
    rows_complete = [r for r in rows_filtered if r["n_missing"] == 0]
    rows_missing = [r for r in rows_filtered if r["n_missing"] > 0]
    
    n_complete_filtered = len(rows_complete)
    n_missing_filtered = len(rows_missing)
    
    if rows_complete:
        orch_scores_complete = [r["risk_score"] for r in rows_complete]
        events_complete = [r["event"] for r in rows_complete]
        times_complete = [r["time"] for r in rows_complete]
        cindex_complete = _safe_cindex(orch_scores_complete, events_complete, times_complete,
                                      f"orchestrator_complete_{cohort}")
    else:
        cindex_complete = float("nan")
    
    if rows_missing:
        orch_scores_missing = [r["risk_score"] for r in rows_missing]
        events_missing = [r["event"] for r in rows_missing]
        times_missing = [r["time"] for r in rows_missing]
        cindex_missing = _safe_cindex(orch_scores_missing, events_missing, times_missing,
                                     f"orchestrator_missing_{cohort}")
    else:
        cindex_missing = float("nan")
    
    # --- Baseline subgroup C-index (complete vs. missing) ---
    if rows_complete:
        baseline_scores_complete = [r.get("baseline_score", float("nan")) for r in rows_complete]
        events_complete = [r["event"] for r in rows_complete]
        times_complete = [r["time"] for r in rows_complete]
        all_valid = all(not np.isnan(b) for b in baseline_scores_complete)
        baseline_cindex_complete = (
            _safe_cindex(baseline_scores_complete, events_complete, times_complete,
                        f"baseline_complete_{cohort}")
            if all_valid else float("nan")
        )
    else:
        baseline_cindex_complete = float("nan")
    
    if rows_missing:
        baseline_scores_missing = [r.get("baseline_score", float("nan")) for r in rows_missing]
        events_missing = [r["event"] for r in rows_missing]
        times_missing = [r["time"] for r in rows_missing]
        all_valid = all(not np.isnan(b) for b in baseline_scores_missing)
        baseline_cindex_missing = (
            _safe_cindex(baseline_scores_missing, events_missing, times_missing,
                        f"baseline_missing_{cohort}")
            if all_valid else float("nan")
        )
    else:
        baseline_cindex_missing = float("nan")
    
    # --- Mean reliability metrics ---
    def _mean_valid(vals: list) -> float | None:
        clean = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
        return round(float(np.mean(clean)), 4) if clean else None
    
    prov_vals = [r["provenance"] for r in rows_filtered]
    mahal_vals = [r["mahal_pct"] for r in rows_filtered]
    ci_vals = [r["ci_width"] for r in rows_filtered]
    
    # --- Build summary ---
    summary = {
        "cohort": cohort,
        "n_test": n_test,
        "n_complete": sum(1 for r in rows if r["n_missing"] == 0),
        "n_missing": sum(1 for r in rows if r["n_missing"] > 0),
        "n_outliers_excluded": n_outliers_excluded,
        "n_complete_filtered": n_complete_filtered,
        "n_missing_filtered": n_missing_filtered,
        "orchestrator_cindex": _round_or_null(orchestrator_cindex),
        "orchestrator_cindex_complete": _round_or_null(cindex_complete),
        "orchestrator_cindex_missing": _round_or_null(cindex_missing),
        "baseline_cindex": _round_or_null(
            _safe_cindex([r.get("baseline_score", float("nan")) for r in rows_filtered],
                        events_filtered, times_filtered, f"baseline_overall_{cohort}")
        ),
        "baseline_cindex_complete": _round_or_null(baseline_cindex_complete),
        "baseline_cindex_missing": _round_or_null(baseline_cindex_missing),
        "mean_provenance": _mean_valid(prov_vals),
        "mean_mahal_pct": _mean_valid(mahal_vals),
        "mean_ci_width": _mean_valid(ci_vals),
        "reanalysis_timestamp": datetime.now(timezone.utc).isoformat(),
    }
    
    return summary


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s"
    )
    
    parser = argparse.ArgumentParser(
        description="Offline re-analysis of evaluation results with outlier filtering and subgroup C-index."
    )
    parser.add_argument(
        "--cohort",
        choices=["luad", "lusc"],
        default=None,
        help="Cohort to reanalyze. If not specified, reanalyze both.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Reanalyze both cohorts.",
    )
    parser.add_argument(
        "--imputation",
        choices=["mice", "knn_tuned"],
        default="mice",
        help="Imputation method (affects which pipeline is loaded).",
    )
    args = parser.parse_args()
    
    cohorts = ["luad", "lusc"] if (args.all or args.cohort is None) else [args.cohort]
    
    for cohort in cohorts:
        logger.info(f"\n{'='*70}")
        logger.info(f"Reanalyzing {cohort.upper()} (imputation={args.imputation})")
        logger.info(f"{'='*70}")
        
        summary = reanalyze_cohort(cohort, imputation=args.imputation)
        
        if not summary:
            logger.error("Reanalysis failed for %s", cohort)
            continue
        
        # Save reanalysis results with imputation method in filename
        output_json = RESULTS_DIR / f"cindex_reanalysis_{cohort}_{args.imputation}.json"
        with open(output_json, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved reanalysis: {output_json}")
        
        # Print summary
        print(f"\n=== Reanalysis: {cohort.upper()} ===")
        print(f"Test patients (total)       : {summary['n_test']}")
        print(f"  Complete                  : {summary['n_complete']}")
        print(f"  Missing modalities        : {summary['n_missing']}")
        print(f"  Outliers excluded         : {summary['n_outliers_excluded']}")
        print(f"  → After filtering         : {summary['n_complete_filtered']} complete, {summary['n_missing_filtered']} missing")
        print(f"\nComparison: Orchestrator vs. Baseline (zero-fill)")
        print(f"  Overall:")
        print(f"    Orchestrator            : {summary['orchestrator_cindex']}")
        print(f"    Baseline                : {summary['baseline_cindex']}")
        print(f"  Complete patients (n={summary['n_complete_filtered']}):")
        print(f"    Orchestrator            : {summary['orchestrator_cindex_complete']}")
        print(f"    Baseline                : {summary['baseline_cindex_complete']}")
        print(f"  Missing modalities (n={summary['n_missing_filtered']}):")
        print(f"    Orchestrator            : {summary['orchestrator_cindex_missing']}")
        print(f"    Baseline                : {summary['baseline_cindex_missing']}")
        print(f"\nReliability metrics:")
        print(f"  Mean provenance           : {summary['mean_provenance']}")
        print(f"  Mean mahal_pct            : {summary['mean_mahal_pct']}")
        print(f"  Mean ci_width             : {summary['mean_ci_width']}")


if __name__ == "__main__":
    main()
