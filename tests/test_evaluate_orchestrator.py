"""
Unit tests for src/evaluation/evaluate_orchestrator.py

All tests use synthetic patients and a mock graph.
No TCGA data, no real LLM, no GPU required.
"""

import csv
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

# predictor.py → explain.py → shap (optional heavy dep).
# _assemble_features itself never calls shap, so a MagicMock is safe here.
if "shap" not in sys.modules:
    sys.modules["shap"] = MagicMock()

from src.evaluation.evaluate_orchestrator import (  # noqa: E402
    _build_initial_state,
    _safe_cindex,
    run_evaluation,
)
from src.orchestrator.nodes.predictor import _assemble_features  # noqa: E402

# ---------------------------------------------------------------------------
# Mock infrastructure
# ---------------------------------------------------------------------------


class _MockModel:
    """Predict the L2 norm of the input as a deterministic risk proxy."""

    def predict_risk(self, x: np.ndarray) -> np.ndarray:
        return np.array([float(np.linalg.norm(x))], dtype=np.float32)


class _MockScaler:
    def transform(self, x: np.ndarray) -> np.ndarray:
        return x


class _MockPCATransform:
    """Reduce any input to 2 components via truncated SVD projection."""

    n_components_ = 2
    explained_variance_ = np.array([5.0, 1.0], dtype=np.float32)

    def transform(self, x: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(0)
        proj = rng.standard_normal((x.shape[1], 2)).astype(np.float32)
        return (x @ proj).astype(np.float32)


class _MockPipeline:
    """Minimal FittedPipeline stub for evaluate_orchestrator tests."""

    model = _MockModel()
    scaler = _MockScaler()
    pca = _MockPCATransform()
    per_modality_transforms = None
    risk_tertiles = (0.33, 0.67)
    cohort = "luad"
    model_name = "coxnet"
    imputation = "mice"
    n_components = 2


class _MockGraph:
    """
    Simulates a compiled LangGraph runnable.
    Returns a deterministic orchestrator result for each patient without
    loading any real data or calling any LLM.
    """

    def __init__(self, patients_by_id: dict[str, dict]) -> None:
        self._patients = patients_by_id

    def invoke(self, state: dict) -> dict:
        pid = state.get("patient_id", "")
        patient = self._patients.get(pid, {})
        n_missing = len(patient.get("missing_modalities", []))
        # Vary risk score per patient so concordance computation is non-trivial.
        risk = 0.3 + (hash(pid) % 1000) / 5000.0
        provenance = max(0.0, 1.0 - n_missing * 0.25)
        return {
            **state,
            "cohort": "luad",
            "available_modalities": patient.get("available_modalities", []),
            "missing_modalities": patient.get("missing_modalities", []),
            "survival_prediction": risk,
            "risk_class": "medium",
            "prediction_reliability": {
                "provenance_proportion": provenance,
                "mahalanobis_ood_distance": {
                    "distance": 2.0,
                    "percentile_rank": 75.0,
                    "backend": "numpy",
                },
                "bootstrap_ci_risk_score": {
                    "lower": risk - 0.05,
                    "point": risk,
                    "upper": risk + 0.05,
                },
            },
            "source_map": {},
            "top_shap_features": [],
            "shap_feature_details": [],
            "execution_log": [],
        }


# ---------------------------------------------------------------------------
# Synthetic patients
# ---------------------------------------------------------------------------

_REQUIRED_JSON_FIELDS = {
    "cohort",
    "n_test",
    "n_complete",
    "n_missing",
    "orchestrator_cindex",
    "baseline_cindex",
    "baseline_source",
    "delta_cindex",
    "mean_provenance",
    "mean_mahal_pct",
    "mean_ci_width",
    "timestamp",
}

_REQUIRED_CSV_COLUMNS = {
    "patient_id",
    "risk_score",
    "risk_class",
    "provenance",
    "mahal_pct",
    "ci_width",
    "n_missing",
    "event",
    "time",
}


def _make_test_patients() -> list[dict]:
    """
    Three synthetic LUAD-like patients.
    Two have events (event=1) with different survival times so that at least
    one concordant pair exists for C-index computation.
    """
    rng = np.random.default_rng(42)
    configs = [
        ("TEST-0000", True, 365.0, [], ["clinical", "transcriptomics", "wsi", "methylation"]),
        ("TEST-0001", True, 180.0, ["wsi"], ["clinical", "transcriptomics", "methylation"]),
        ("TEST-0002", False, 730.0, ["wsi", "methylation"], ["clinical", "transcriptomics"]),
    ]
    patients = []
    for pid, event, time_val, missing, available in configs:
        patients.append(
            {
                "patient_id": pid,
                "clinical": rng.random(63).astype(np.float32) if "clinical" in available else None,
                "transcriptomics": rng.random(1824).astype(np.float32) if "transcriptomics" in available else None,
                "wsi": rng.random(1024).astype(np.float32) if "wsi" in available else None,
                "methylation": rng.random(16166).astype(np.float32) if "methylation" in available else None,
                "available_modalities": available,
                "missing_modalities": missing,
                "label": {"DSS": float(event), "DSS.time": time_val},
            }
        )
    return patients


def _make_graph_and_pipeline(patients: list[dict]):
    patients_by_id = {p["patient_id"]: p for p in patients}
    return _MockGraph(patients_by_id), _MockPipeline()


# ---------------------------------------------------------------------------
# Tests: _build_initial_state
# ---------------------------------------------------------------------------


class TestBuildInitialState:
    def test_contains_all_new_fields(self):
        state = _build_initial_state("TCGA-99-0001")
        assert "guidance" in state
        assert "generation_candidates" in state
        assert "prediction_reliability" in state

    def test_patient_id_is_set(self):
        state = _build_initial_state("TCGA-99-0001")
        assert state["patient_id"] == "TCGA-99-0001"

    def test_new_fields_are_empty_dicts(self):
        state = _build_initial_state("X")
        assert state["guidance"] == {}
        assert state["generation_candidates"] == {}
        assert state["prediction_reliability"] == {}


# ---------------------------------------------------------------------------
# Tests: _safe_cindex
# ---------------------------------------------------------------------------


class TestSafeCindex:
    def test_valid_scores_return_float(self):
        pytest.importorskip("sksurv", reason="scikit-survival not installed")
        scores = [0.8, 0.3, 0.6]
        events = [True, True, False]
        times = [100.0, 200.0, 300.0]
        result = _safe_cindex(scores, events, times, "test")
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_no_events_returns_nan(self):
        result = _safe_cindex([0.5, 0.5], [False, False], [100.0, 200.0])
        assert np.isnan(result)

    def test_single_patient_returns_nan(self):
        result = _safe_cindex([0.5], [True], [100.0])
        assert np.isnan(result)


# ---------------------------------------------------------------------------
# Tests: run_evaluation
# ---------------------------------------------------------------------------


class TestRunEvaluation:
    @pytest.fixture
    def eval_results(self, tmp_path):
        patients = _make_test_patients()
        graph, pipeline = _make_graph_and_pipeline(patients)
        summary, rows = run_evaluation(
            test_patients=patients,
            pipeline=pipeline,
            graph=graph,
            cohort="luad",
            output_dir=tmp_path,
        )
        return summary, rows, tmp_path

    def test_json_has_all_required_fields(self, eval_results):
        _summary, _rows, tmp_path = eval_results
        json_path = tmp_path / "cindex_comparison_luad.json"
        assert json_path.exists(), "JSON output file not created"
        with open(json_path) as f:
            data = json.load(f)
        missing = _REQUIRED_JSON_FIELDS - set(data.keys())
        assert not missing, f"JSON missing fields: {missing}"

    def test_csv_has_correct_columns(self, eval_results):
        _summary, _rows, tmp_path = eval_results
        csv_path = tmp_path / "per_patient_luad.csv"
        assert csv_path.exists(), "CSV output file not created"
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            cols = set(reader.fieldnames or [])
        missing = _REQUIRED_CSV_COLUMNS - cols
        assert not missing, f"CSV missing columns: {missing}"

    def test_summary_n_test_matches_patients(self, eval_results):
        summary, rows, _ = eval_results
        assert summary["n_test"] == 3
        assert len(rows) == 3

    def test_summary_cohort_field(self, eval_results):
        summary, _rows, _ = eval_results
        assert summary["cohort"] == "luad"

    def test_summary_counts_are_consistent(self, eval_results):
        summary, _rows, _ = eval_results
        assert summary["n_complete"] + summary["n_missing"] == summary["n_test"]

    def test_orchestrator_cindex_is_float_or_null(self, eval_results):
        summary, _rows, _ = eval_results
        val = summary["orchestrator_cindex"]
        assert val is None or isinstance(val, float)

    def test_timestamp_is_iso_format(self, eval_results):
        summary, _rows, _ = eval_results
        ts = summary["timestamp"]
        assert isinstance(ts, str) and "T" in ts

    def test_csv_row_count_matches_n_test(self, eval_results):
        summary, _rows, tmp_path = eval_results
        csv_path = tmp_path / "per_patient_luad.csv"
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == summary["n_test"]

    def test_csv_event_column_values(self, eval_results):
        _summary, _rows, tmp_path = eval_results
        csv_path = tmp_path / "per_patient_luad.csv"
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            events = [int(r["event"]) for r in reader]
        assert set(events).issubset({0, 1})

    def test_no_output_dir_does_not_raise(self):
        """run_evaluation with output_dir=None must not write files or raise."""
        patients = _make_test_patients()
        graph, pipeline = _make_graph_and_pipeline(patients)
        summary, rows = run_evaluation(
            test_patients=patients,
            pipeline=pipeline,
            graph=graph,
            cohort="luad",
            output_dir=None,
        )
        assert "n_test" in summary
        assert len(rows) == 3

    def test_skips_patients_with_invalid_labels(self):
        """Patients with missing or non-positive DSS labels are skipped."""
        patients = _make_test_patients()
        patients[0]["label"] = {"DSS": float("nan"), "DSS.time": 100.0}
        patients[1]["label"] = {"DSS": 1.0, "DSS.time": -5.0}
        graph, pipeline = _make_graph_and_pipeline(patients)
        summary, rows = run_evaluation(
            test_patients=patients,
            pipeline=pipeline,
            graph=graph,
            cohort="luad",
            output_dir=None,
        )
        assert summary["n_test"] == 1
        assert len(rows) == 1

    def test_pipeline_none_baseline_cindex_is_null(self):
        """When no pipeline is provided, baseline_cindex must be None."""
        patients = _make_test_patients()
        graph, _ = _make_graph_and_pipeline(patients)
        summary, _rows = run_evaluation(
            test_patients=patients,
            pipeline=None,
            graph=graph,
            cohort="luad",
            output_dir=None,
        )
        assert summary["baseline_cindex"] is None

    def test_stored_mice_cindex_used_when_present(self):
        """When pipeline.baseline_cindex is set, it is used verbatim and source is 'mice_stored'."""
        patients = _make_test_patients()
        graph, pipeline = _make_graph_and_pipeline(patients)
        pipeline.baseline_cindex = 0.706
        summary, _rows = run_evaluation(
            test_patients=patients,
            pipeline=pipeline,
            graph=graph,
            cohort="luad",
            output_dir=None,
        )
        assert summary["baseline_cindex"] == pytest.approx(0.706, abs=1e-4)
        assert summary["baseline_source"] == "mice_stored"

    def test_zero_fill_fallback_when_no_stored_cindex(self):
        """When pipeline.baseline_cindex is None, source must be 'zero_fill' or 'none'."""
        patients = _make_test_patients()
        graph, pipeline = _make_graph_and_pipeline(patients)
        pipeline.baseline_cindex = None
        summary, _rows = run_evaluation(
            test_patients=patients,
            pipeline=pipeline,
            graph=graph,
            cohort="luad",
            output_dir=None,
        )
        assert summary["baseline_source"] in ("zero_fill", "none")


# ---------------------------------------------------------------------------
# Tests: actual_dims correctness (LUSC cohort-specific dimensions)
# ---------------------------------------------------------------------------

_LUSC_DIMS = {
    "clinical": 56,
    "transcriptomics": 1824,
    "wsi": 1024,
    "methylation": 16206,
}


def _make_lusc_state(rng: np.random.Generator) -> dict:
    """Minimal state dict for a LUSC patient with cohort-correct array sizes."""
    return {
        "patient_id": "LUSC-TEST-0001",
        "cohort": "lusc",
        "clinical": rng.random(56).astype(np.float32),
        "transcriptomics": rng.random(1824).astype(np.float32),
        "wsi": rng.random(1024).astype(np.float32),
        "methylation": rng.random(16206).astype(np.float32),
        "available_modalities": ["clinical", "transcriptomics", "wsi", "methylation"],
        "missing_modalities": [],
        "generated_modalities": {},
        "verification_scores": {},
        "verification_passed": False,
    }


class TestActualDims:
    def test_lusc_real_data_not_zero_filled_when_actual_dims_set(self):
        """
        With pipeline.actual_dims = LUSC sizes, real LUSC arrays must be
        accepted as 'real', not silently zero-filled.
        """
        rng = np.random.default_rng(0)
        state = _make_lusc_state(rng)
        _, source_map = _assemble_features(state, _LUSC_DIMS)
        assert source_map["clinical"]["source"] == "real", (
            "clinical should be 'real' with actual_dims={'clinical': 56}"
        )
        assert source_map["methylation"]["source"] == "real", (
            "methylation should be 'real' with actual_dims={'methylation': 16206}"
        )

    def test_lusc_real_data_zero_filled_without_actual_dims(self):
        """
        Without actual_dims the global MODALITY_DIMS (clinical=63, meth=16166)
        mismatch LUSC arrays → both blocks must be zero-filled.
        """
        rng = np.random.default_rng(0)
        state = _make_lusc_state(rng)
        _, source_map = _assemble_features(state, actual_dims=None)
        assert source_map["clinical"]["source"] == "zero", (
            "clinical should be zero-filled when MODALITY_DIMS used for LUSC"
        )
        assert source_map["methylation"]["source"] == "zero", (
            "methylation should be zero-filled when MODALITY_DIMS used for LUSC"
        )

    def test_x_raw_shape_matches_actual_dims(self):
        """Total x_raw width must equal the sum of actual_dims, not MODALITY_DIMS."""
        rng = np.random.default_rng(0)
        state = _make_lusc_state(rng)
        x_raw, _ = _assemble_features(state, _LUSC_DIMS)
        expected_width = sum(_LUSC_DIMS.values())
        assert x_raw.shape == (1, expected_width)

    def test_luad_unchanged_without_actual_dims(self):
        """LUAD state with default MODALITY_DIMS must still produce 'real' sources."""
        from src.data_loader import MODALITY_DIMS
        rng = np.random.default_rng(1)
        state = {
            "patient_id": "LUAD-TEST-0001",
            "cohort": "luad",
            "clinical": rng.random(63).astype(np.float32),
            "transcriptomics": rng.random(1824).astype(np.float32),
            "wsi": rng.random(1024).astype(np.float32),
            "methylation": rng.random(16166).astype(np.float32),
            "available_modalities": ["clinical", "transcriptomics", "wsi", "methylation"],
            "missing_modalities": [],
            "generated_modalities": {},
            "verification_scores": {},
            "verification_passed": False,
        }
        _, source_map = _assemble_features(state, actual_dims=None)
        for mod in ["clinical", "transcriptomics", "wsi", "methylation"]:
            assert source_map[mod]["source"] == "real", f"{mod} should be real for LUAD"
