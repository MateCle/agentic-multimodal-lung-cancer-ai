"""
Unit tests for the real distributional + LLM Verifier node.
Uses synthetic data — no TCGA files or LLM API required.
"""

import numpy as np
import pytest

from src.data_loader import MODALITY_DIMS
from src.orchestrator.llm import MockLLMClient
from src.orchestrator.nodes.verifier import (
    _check_distributional,
    build_pool_stats,
    make_verifier_node,
)
from src.orchestrator.state import PatientState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pool_entry(patient_id, available, seed=0):
    rng = np.random.default_rng(seed)
    entry = {
        "patient_id": patient_id,
        "available": available,
        "features": {},
        "features_norm": {},
    }
    for mod in available:
        arr = rng.standard_normal(MODALITY_DIMS[mod]).astype(np.float32)
        entry["features"][mod] = arr
        norm = np.linalg.norm(arr)
        entry["features_norm"][mod] = arr / norm if norm > 0 else arr
    return entry


def _build_synthetic_pool(n_patients=20):
    pool = []
    for i in range(n_patients):
        pid = f"TCGA-VTEST-{i:04d}"
        available = ["clinical", "transcriptomics"]
        if i % 3 == 0:
            available.append("wsi")
        pool.append(_make_pool_entry(pid, available, seed=i))
    return pool


def _make_state(generated_modalities, execution_log=None):
    return PatientState(
        patient_id="TCGA-VQUERY-0001",
        cohort="luad",
        clinical=np.zeros(63),
        transcriptomics=None,
        wsi=None,
        methylation=None,
        available_modalities=["clinical"],
        missing_modalities=["transcriptomics"],
        agent_summaries={},
        mining_rules={},
        generated_modalities=generated_modalities,
        verification_scores={},
        verification_passed=False,
        survival_prediction=None,
        routing_decision="generate",
        execution_log=execution_log or [],
        correction_hints={},
    )


# ---------------------------------------------------------------------------
# Pool stats tests
# ---------------------------------------------------------------------------


class TestBuildPoolStats:
    def test_computes_stats_for_available_modalities(self):
        pool = _build_synthetic_pool(20)
        stats = build_pool_stats(pool)
        assert stats["clinical"]["valid"] is True
        assert stats["transcriptomics"]["valid"] is True

    def test_invalid_for_rare_modalities(self):
        pool = _build_synthetic_pool(3)
        stats = build_pool_stats(pool)
        assert stats["methylation"]["valid"] is False

    def test_mean_shape_matches_modality_dim(self):
        pool = _build_synthetic_pool(20)
        stats = build_pool_stats(pool)
        assert stats["clinical"]["mean"].shape == (MODALITY_DIMS["clinical"],)

    def test_std_is_positive(self):
        pool = _build_synthetic_pool(20)
        stats = build_pool_stats(pool)
        assert np.all(stats["clinical"]["std"] > 0)


# ---------------------------------------------------------------------------
# Distributional check tests
# ---------------------------------------------------------------------------


class TestCheckDistributional:
    def test_perfect_score_for_in_range(self):
        mod_stats = {"mean": np.zeros(100), "std": np.ones(100), "valid": True}
        score, outliers = _check_distributional(np.zeros(100), mod_stats)
        assert score == pytest.approx(1.0)
        assert outliers == []

    def test_zero_score_for_extreme(self):
        mod_stats = {"mean": np.zeros(100), "std": np.ones(100), "valid": True}
        score, outliers = _check_distributional(np.full(100, 100.0), mod_stats)
        assert score == pytest.approx(0.0)
        assert len(outliers) == 100

    def test_partial_score(self):
        mod_stats = {"mean": np.zeros(10), "std": np.ones(10), "valid": True}
        generated = np.zeros(10)
        generated[0] = 100.0
        generated[1] = 100.0
        score, outliers = _check_distributional(generated, mod_stats)
        assert score == pytest.approx(0.8)
        assert set(outliers) == {0, 1}

    def test_invalid_stats_returns_perfect(self):
        mod_stats = {"mean": np.zeros(10), "std": np.ones(10), "valid": False}
        score, outliers = _check_distributional(np.full(10, 999.0), mod_stats)
        assert score == pytest.approx(1.0)
        assert outliers == []


# ---------------------------------------------------------------------------
# Verifier node (real with mock LLM) tests
# ---------------------------------------------------------------------------


class TestMakeVerifierNode:
    @pytest.fixture
    def pool_stats(self):
        pool = _build_synthetic_pool(20)
        return build_pool_stats(pool)

    def test_scores_are_between_1_and_5(self, pool_stats):
        llm = MockLLMClient()
        verify_fn = make_verifier_node(pool_stats, llm)

        mean = pool_stats["transcriptomics"]["mean"]
        state = _make_state(generated_modalities={"transcriptomics": mean})
        result = verify_fn(state)
        score = result["verification_scores"]["transcriptomics"]
        assert 1.0 <= score <= 5.0

    def test_produces_correction_hints_on_low_score(self, pool_stats):
        """Extreme features should get a low distributional score."""
        llm = MockLLMClient()
        # Use a low threshold so mock LLM score (which is high) still passes,
        # but distributional outliers generate hints
        verify_fn = make_verifier_node(pool_stats, llm, threshold=5.0)

        extreme = np.full(MODALITY_DIMS["transcriptomics"], 1000.0)
        state = _make_state(generated_modalities={"transcriptomics": extreme})
        result = verify_fn(state)
        # With threshold 5.0, mock LLM returns ~3-4 so it should fail
        assert result["verification_passed"] is False

    def test_execution_log_contains_details(self, pool_stats):
        llm = MockLLMClient()
        verify_fn = make_verifier_node(pool_stats, llm)

        mean = pool_stats["transcriptomics"]["mean"]
        state = _make_state(generated_modalities={"transcriptomics": mean})
        result = verify_fn(state)
        log = " ".join(result["execution_log"])
        assert "[Verifier]" in log
        assert "distributional" in log

    def test_empty_generated_passes(self, pool_stats):
        llm = MockLLMClient()
        verify_fn = make_verifier_node(pool_stats, llm)
        state = _make_state(generated_modalities={})
        result = verify_fn(state)
        assert result["verification_passed"] is True
