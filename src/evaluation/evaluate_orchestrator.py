"""
End-to-end predictive evaluation.

For each cohort test split, runs the full orchestrator pipeline and
computes C-index vs the baseline CoxNet+MICE predictor (zero-fill for
missing modalities, no generation).

Usage:
    # Mock mode (no GPU, no LLM, no .joblib):
    python -m src.evaluation.evaluate_orchestrator --cohort luad --mock

    # Real mode (after .joblib regenerated on AI-LAB):
    python -m src.evaluation.evaluate_orchestrator --cohort luad
    python -m src.evaluation.evaluate_orchestrator --cohort lusc
"""

import argparse
import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

DATA_DIR = Path("data/extracted/cache_data")
SPLITS_DIR = DATA_DIR / "splits"
RESULTS_DIR = Path("results/evaluation")

RISK_SCORE_OUTLIER_THRESHOLD = 100.0

_SPLIT_FILES = {
    "luad": "tcga_luad_DSS_k3_r1_test0.2_val0.2_seed42.json",
    "lusc": "tcga_lusc_DSS_k5_r1_test0.2_val0.2_seed42.json",
}


# ---------------------------------------------------------------------------
# State builder
# ---------------------------------------------------------------------------


def _build_initial_state(patient_id: str) -> dict:
    """Build a complete initial PatientState dict for graph.invoke()."""
    return {
        "user_query": "",
        "parsed_query": {},
        "patient_id": patient_id,
        "cohort": "",
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
    }


# ---------------------------------------------------------------------------
# C-index helper
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Baseline risk (predictor only, no generation)
# ---------------------------------------------------------------------------


def _compute_baseline_risk(patient: dict, pipeline, cohort: str) -> float:
    """
    Run the baseline predictor directly on a patient, zero-filling missing
    modalities (no LLM generation).  Raises on pipeline failure so the
    caller can record nan.
    """
    from src.orchestrator.nodes.predictor import _apply_pipeline, _assemble_features

    state: dict = {
        "patient_id": patient["patient_id"],
        "cohort": cohort,
        "clinical": patient.get("clinical"),
        "transcriptomics": patient.get("transcriptomics"),
        "wsi": patient.get("wsi"),
        "methylation": patient.get("methylation"),
        "available_modalities": patient.get("available_modalities", []),
        "missing_modalities": patient.get("missing_modalities", []),
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


# ---------------------------------------------------------------------------
# Core evaluation (injectable for tests)
# ---------------------------------------------------------------------------


def run_evaluation(
    test_patients: list[dict],
    pipeline,
    graph,
    cohort: str,
    output_dir: Path | None = None,
) -> tuple[dict, list[dict]]:
    """
    Run full orchestrator evaluation on a list of patients.

    Args:
        test_patients: List of patient dicts from load_split_patients().
        pipeline:      FittedPipeline for baseline computation, or None.
        graph:         Compiled LangGraph runnable.
        cohort:        'luad' or 'lusc'.
        output_dir:    Directory to write JSON + CSV; skipped when None.

    Returns:
        (summary_dict, per_patient_rows)
    """
    orchestrator_scores: list[float] = []
    baseline_scores: list[float] = []
    events: list[bool] = []
    times: list[float] = []
    rows: list[dict] = []

    for patient in test_patients:
        pid = patient["patient_id"]
        label = patient.get("label", {})
        event = label.get("DSS")
        time_val = label.get("DSS.time")

        if event is None or time_val is None:
            logger.debug("Skipping %s: missing label.", pid)
            continue
        try:
            if np.isnan(event) or np.isnan(time_val) or float(time_val) <= 0:
                logger.debug("Skipping %s: invalid label values.", pid)
                continue
        except (TypeError, ValueError):
            logger.debug("Skipping %s: non-numeric label.", pid)
            continue

        # --- Orchestrator path ---
        initial_state = _build_initial_state(pid)
        try:
            result = graph.invoke(initial_state)
        except Exception as exc:
            logger.error("Orchestrator failed for %s: %s", pid, exc)
            continue

        risk_score = result.get("survival_prediction")
        if risk_score is None:
            logger.warning("No survival_prediction for %s; skipping.", pid)
            continue

        risk_class = result.get("risk_class", "unknown")
        reliability = result.get("prediction_reliability") or {}

        prov = reliability.get("provenance_proportion", float("nan"))

        mahal_info = reliability.get("mahalanobis_ood_distance") or {}
        mahal_pct = mahal_info.get("percentile_rank", float("nan"))

        ci_info = reliability.get("bootstrap_ci_risk_score") or {}
        ci_lo = ci_info.get("lower", float("nan"))
        ci_up = ci_info.get("upper", float("nan"))
        try:
            ci_width = (
                float(ci_up) - float(ci_lo)
                if not (np.isnan(ci_lo) or np.isnan(ci_up))
                else float("nan")
            )
        except (TypeError, ValueError):
            ci_width = float("nan")

        n_missing = len(
            result.get("missing_modalities") or patient.get("missing_modalities", [])
        )

        # --- Baseline path ---
        baseline_score = float("nan")
        if pipeline is not None:
            try:
                baseline_score = _compute_baseline_risk(patient, pipeline, cohort)
            except Exception as exc:
                logger.warning("Baseline failed for %s: %s", pid, exc)

        orchestrator_scores.append(float(risk_score))
        baseline_scores.append(baseline_score)
        events.append(bool(event))
        times.append(float(time_val))

        rows.append(
            {
                "patient_id": pid,
                "risk_score": float(risk_score),
                "risk_class": risk_class,
                "provenance": prov,
                "mahal_pct": mahal_pct,
                "ci_width": ci_width,
                "n_missing": n_missing,
                "event": int(bool(event)),
                "time": float(time_val),
            }
        )

    # --- FIX 1: Outlier filtering ---
    # Filter out patients with |risk_score| > threshold before C-index calculation
    outlier_mask = [
        abs(score) <= RISK_SCORE_OUTLIER_THRESHOLD for score in orchestrator_scores
    ]
    n_outliers_excluded = sum(1 for m in outlier_mask if not m)
    if n_outliers_excluded > 0:
        logger.warning(
            "Excluding %d outlier patients (|risk_score| > %.1f) from C-index calculation.",
            n_outliers_excluded,
            RISK_SCORE_OUTLIER_THRESHOLD,
        )
        for i, (score, pid) in enumerate(
            zip(orchestrator_scores, [r["patient_id"] for r in rows])
        ):
            if not outlier_mask[i]:
                logger.warning("  Outlier: %s (score=%.4f)", pid, score)

    # Apply mask to all lists
    orchestrator_scores_filtered = [
        s for s, m in zip(orchestrator_scores, outlier_mask) if m
    ]
    baseline_scores_filtered = [b for b, m in zip(baseline_scores, outlier_mask) if m]
    events_filtered = [e for e, m in zip(events, outlier_mask) if m]
    times_filtered = [t for t, m in zip(times, outlier_mask) if m]
    rows_for_metrics = [r for r, m in zip(rows, outlier_mask) if m]

    # --- C-indices ---
    n_test = len(rows)
    n_complete = sum(1 for r in rows if r["n_missing"] == 0)

    orchestrator_cindex = _safe_cindex(
        orchestrator_scores_filtered, events_filtered, times_filtered, "orchestrator"
    )

    # Prefer the true MICE baseline test C-index stored in the pipeline
    # (computed in main_baseline.py with the fitted IterativeImputer, which
    # is NOT persisted to disk).  Fall back to the zero-fill computed baseline
    # only when the field is absent (old .joblib before this change).
    stored_mice_cindex = (
        getattr(pipeline, "baseline_cindex", None) if pipeline is not None else None
    )

    if stored_mice_cindex is not None:
        baseline_cindex = float(stored_mice_cindex)
        baseline_source = "mice_stored"
        logger.info(
            "Using stored MICE baseline C-index: %.4f (from pipeline.baseline_cindex).",
            baseline_cindex,
        )
    else:
        all_baseline_valid = bool(baseline_scores_filtered) and all(
            not np.isnan(b) for b in baseline_scores_filtered
        )
        baseline_cindex = (
            _safe_cindex(
                baseline_scores_filtered,
                events_filtered,
                times_filtered,
                "baseline_zero_fill",
            )
            if all_baseline_valid
            else float("nan")
        )
        baseline_source = "zero_fill" if all_baseline_valid else "none"
        if pipeline is not None and stored_mice_cindex is None:
            logger.warning(
                "pipeline.baseline_cindex not set — using zero-fill fallback. "
                "Regenerate .joblib with current main_baseline.py to get the "
                "true CoxNet+MICE baseline."
            )

    delta_cindex = (
        orchestrator_cindex - baseline_cindex
        if not (np.isnan(orchestrator_cindex) or np.isnan(baseline_cindex))
        else float("nan")
    )

    # --- FIX 2: Subgroup C-index (complete vs. missing modalities) ---
    # Split filtered rows into complete and incomplete
    rows_complete_filtered = [r for r in rows_for_metrics if r["n_missing"] == 0]
    rows_missing_filtered = [r for r in rows_for_metrics if r["n_missing"] > 0]

    if rows_complete_filtered:
        orch_scores_complete = [r["risk_score"] for r in rows_complete_filtered]
        events_complete = [r["event"] for r in rows_complete_filtered]
        times_complete = [r["time"] for r in rows_complete_filtered]
        cindex_complete = _safe_cindex(
            orch_scores_complete,
            events_complete,
            times_complete,
            "orchestrator_complete",
        )
    else:
        cindex_complete = float("nan")

    if rows_missing_filtered:
        orch_scores_missing = [r["risk_score"] for r in rows_missing_filtered]
        events_missing = [r["event"] for r in rows_missing_filtered]
        times_missing = [r["time"] for r in rows_missing_filtered]
        cindex_missing = _safe_cindex(
            orch_scores_missing, events_missing, times_missing, "orchestrator_missing"
        )
    else:
        cindex_missing = float("nan")

    # Baseline subgroup C-index (using filtered baseline_scores and events)
    if rows_complete_filtered and stored_mice_cindex is None:
        baseline_scores_complete = [
            baseline_scores_filtered[i]
            for i, r in enumerate(rows_for_metrics)
            if r in rows_complete_filtered
        ]
        events_complete_all = [
            events_filtered[i]
            for i, r in enumerate(rows_for_metrics)
            if r in rows_complete_filtered
        ]
        times_complete_all = [
            times_filtered[i]
            for i, r in enumerate(rows_for_metrics)
            if r in rows_complete_filtered
        ]
        baseline_cindex_complete = (
            _safe_cindex(
                baseline_scores_complete,
                events_complete_all,
                times_complete_all,
                "baseline_complete",
            )
            if all(not np.isnan(b) for b in baseline_scores_complete)
            else float("nan")
        )
    else:
        baseline_cindex_complete = float("nan")

    if rows_missing_filtered and stored_mice_cindex is None:
        baseline_scores_missing = [
            baseline_scores_filtered[i]
            for i, r in enumerate(rows_for_metrics)
            if r in rows_missing_filtered
        ]
        events_missing_all = [
            events_filtered[i]
            for i, r in enumerate(rows_for_metrics)
            if r in rows_missing_filtered
        ]
        times_missing_all = [
            times_filtered[i]
            for i, r in enumerate(rows_for_metrics)
            if r in rows_missing_filtered
        ]
        baseline_cindex_missing = (
            _safe_cindex(
                baseline_scores_missing,
                events_missing_all,
                times_missing_all,
                "baseline_missing",
            )
            if all(not np.isnan(b) for b in baseline_scores_missing)
            else float("nan")
        )
    else:
        baseline_cindex_missing = float("nan")

    # --- Mean reliability metrics (exclude nan) ---
    def _mean_valid(vals: list) -> float | None:
        clean = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
        return round(float(np.mean(clean)), 4) if clean else None

    prov_vals = [r["provenance"] for r in rows]
    mahal_vals = [r["mahal_pct"] for r in rows]
    ci_vals = [r["ci_width"] for r in rows]

    summary: dict = {
        "cohort": cohort,
        "n_test": n_test,
        "n_complete": n_complete,
        "n_missing": n_test - n_complete,
        "n_outliers_excluded": n_outliers_excluded,
        "orchestrator_cindex": _round_or_null(orchestrator_cindex),
        "orchestrator_cindex_complete": _round_or_null(cindex_complete),
        "orchestrator_cindex_missing": _round_or_null(cindex_missing),
        "baseline_cindex": _round_or_null(baseline_cindex),
        "baseline_cindex_complete": _round_or_null(baseline_cindex_complete),
        "baseline_cindex_missing": _round_or_null(baseline_cindex_missing),
        "baseline_source": baseline_source,
        "delta_cindex": _round_or_null(delta_cindex),
        "mean_provenance": _mean_valid(prov_vals),
        "mean_mahal_pct": _mean_valid(mahal_vals),
        "mean_ci_width": _mean_valid(ci_vals),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # --- Persist ---
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / f"cindex_comparison_{cohort}.json"
        csv_path = output_dir / f"per_patient_{cohort}.csv"

        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Saved JSON: %s", json_path)

        if rows:
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            logger.info("Saved CSV: %s", csv_path)

    _print_summary(summary, cohort, output_dir)
    return summary, rows


def _print_summary(summary: dict, cohort: str, output_dir: Path | None) -> None:
    n = summary["n_test"]
    nc = summary["n_complete"]
    nm = summary["n_missing"]
    n_outliers = summary.get("n_outliers_excluded", 0)
    oc = summary["orchestrator_cindex"]
    oc_complete = summary.get("orchestrator_cindex_complete")
    oc_missing = summary.get("orchestrator_cindex_missing")
    bc = summary["baseline_cindex"]
    bc_complete = summary.get("baseline_cindex_complete")
    bc_missing = summary.get("baseline_cindex_missing")
    dc = summary["delta_cindex"]
    src = summary.get("baseline_source", "unknown")
    src_label = {
        "mice_stored": "CoxNet+MICE (stored)",
        "zero_fill": "zero-fill fallback",
        "none": "N/A",
    }.get(src, src)
    print(f"\n=== Evaluation: {cohort.upper()} ===")
    print(
        f"Test patients        : {n}  ({nc} complete, {nm} with missing modalities, {n_outliers} outliers excluded)"
    )
    print(f"Orchestrator C-index : {oc if oc is not None else 'N/A'}")
    if oc_complete is not None or oc_missing is not None:
        print(
            f"  ├─ Complete (n={nc})     : {oc_complete if oc_complete is not None else 'N/A'}"
        )
        print(
            f"  └─ Missing (n={nm})      : {oc_missing if oc_missing is not None else 'N/A'}"
        )
    print(f"Baseline C-index     : {bc if bc is not None else 'N/A'}  [{src_label}]")
    if bc_complete is not None or bc_missing is not None:
        print(
            f"  ├─ Complete          : {bc_complete if bc_complete is not None else 'N/A'}"
        )
        print(
            f"  └─ Missing           : {bc_missing if bc_missing is not None else 'N/A'}"
        )
    print(
        f"Delta C-index        : {f'+{dc:.4f}' if dc is not None and dc >= 0 else (f'{dc:.4f}' if dc is not None else 'N/A')}"
    )
    if output_dir:
        print(f"\nJSON : {output_dir / f'cindex_comparison_{cohort}.json'}")
        print(f"CSV  : {output_dir / f'per_patient_{cohort}.csv'}")


# ---------------------------------------------------------------------------
# CLI wrapper
# ---------------------------------------------------------------------------


def evaluate(
    cohort: str,
    model_name: str = "coxnet",
    imputation: str = "mice",
    mock: bool = False,
    max_patients: int | None = None,
    output_dir: Path | None = RESULTS_DIR,
    n_candidates: int = 3,
    miner_temperature: float | None = None,
    generator_temperature: float | None = None,
) -> dict:
    """Load data from disk, build graph, run evaluation, save results."""
    from src.baseline.pipeline import load_pipeline, pipeline_path
    from src.data_loader import load_raw_data, load_split, load_split_patients
    from src.orchestrator.graph import build_graph

    logger.info("Loading %s data...", cohort.upper())
    raw_data, _ = load_raw_data(DATA_DIR, cohort)

    split_file = _SPLIT_FILES.get(cohort)
    if split_file is None:
        raise ValueError(f"Unknown cohort: {cohort!r}")
    train_ids, _val_ids, test_ids = load_split(SPLITS_DIR, split_file)
    test_patients = load_split_patients(test_ids, raw_data)
    if max_patients:
        test_patients = test_patients[:max_patients]

    logger.info("%s test patients: %d", cohort.upper(), len(test_patients))

    # Baseline pipeline (may not exist if not yet generated on AI-LAB)
    p_path = pipeline_path(cohort, model_name, imputation)
    pipeline = None
    if p_path.exists():
        pipeline = load_pipeline(cohort, model_name, imputation)
    else:
        logger.warning(
            "No pipeline at %s — baseline C-index will be N/A. "
            "Run main_baseline.py --cohort %s --model %s --imputation %s first.",
            p_path,
            cohort,
            model_name,
            imputation,
        )

    # Graph
    train_patient_ids = None if mock else train_ids
    graph = build_graph(
        data_dir=DATA_DIR,
        model_name=model_name,
        imputation=imputation,
        train_patient_ids=train_patient_ids,
        n_candidates=n_candidates,
        miner_temperature=miner_temperature,
        generator_temperature=generator_temperature,
    )

    summary, _rows = run_evaluation(
        test_patients=test_patients,
        pipeline=pipeline,
        graph=graph,
        cohort=cohort,
        output_dir=output_dir,
    )
    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description="End-to-end orchestrator evaluation.")
    parser.add_argument(
        "--cohort",
        required=True,
        choices=["luad", "lusc"],
        help="Which cohort to evaluate.",
    )
    parser.add_argument("--model", default="coxnet")
    parser.add_argument("--imputation", default="mice")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock LLM and no retrieval pool (for smoke-testing).",
    )
    parser.add_argument(
        "--max-patients",
        type=int,
        default=None,
        help="Limit number of test patients (useful for quick tests).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory where evaluation results are written.",
    )
    parser.add_argument(
        "--n-candidates",
        type=int,
        default=3,
        help="Number of generation candidates to request from the Generator (best-of-N).",
    )
    parser.add_argument(
        "--miner-temperature",
        type=float,
        default=None,
        help="Override Miner LLM temperature (None = use default T=0.3).",
    )
    parser.add_argument(
        "--generator-temperature",
        type=float,
        default=None,
        help="Override Generator LLM temperature (None = use default T=0.3).",
    )
    args = parser.parse_args()

    evaluate(
        cohort=args.cohort,
        model_name=args.model,
        imputation=args.imputation,
        mock=args.mock,
        max_patients=args.max_patients,
        output_dir=args.output_dir,
        n_candidates=args.n_candidates,
        miner_temperature=args.miner_temperature,
        generator_temperature=args.generator_temperature,
    )


if __name__ == "__main__":
    main()
