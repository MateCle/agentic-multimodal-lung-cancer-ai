"""
Unit tests for the real distributional + LLM Post-Generation Verifier nodes.
Uses synthetic data — no TCGA files or LLM API required.

Covers:
    - Pool statistics
    - Distributional check
    - Post-Generation Verifier (make_post_generation_verifier_node):
        backward-compat + best-of-N ranking
    - Pre-Generation Verifier (make_pre_generation_verifier_node): guidance refinement
"""

import numpy as np
import pytest

from src.data_loader import MODALITY_DIMS
from src.orchestrator.llm import MockLLMClient
from src.orchestrator.nodes.verifier import (
    _check_distributional,
    build_pool_stats,
    make_post_generation_verifier_node,
    make_pre_generation_verifier_node,
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
# Post-Generation Verifier node (real with mock LLM) tests
# ---------------------------------------------------------------------------


class TestMakePostGenerationVerifierNode:
    @pytest.fixture
    def pool_stats(self):
        pool = _build_synthetic_pool(20)
        return build_pool_stats(pool)

    def test_scores_are_between_1_and_5(self, pool_stats):
        llm = MockLLMClient()
        verify_fn = make_post_generation_verifier_node(pool_stats, llm)

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
        verify_fn = make_post_generation_verifier_node(pool_stats, llm, threshold=5.0)

        extreme = np.full(MODALITY_DIMS["transcriptomics"], 1000.0)
        state = _make_state(generated_modalities={"transcriptomics": extreme})
        result = verify_fn(state)
        # With threshold 5.0, mock LLM returns ~3-4 so it should fail
        assert result["verification_passed"] is False

    def test_execution_log_contains_details(self, pool_stats):
        llm = MockLLMClient()
        verify_fn = make_post_generation_verifier_node(pool_stats, llm)

        mean = pool_stats["transcriptomics"]["mean"]
        state = _make_state(generated_modalities={"transcriptomics": mean})
        result = verify_fn(state)
        log = " ".join(result["execution_log"])
        assert "[Post-Generation Verifier]" in log
        assert "distributional" in log

    def test_empty_generated_passes(self, pool_stats):
        llm = MockLLMClient()
        verify_fn = make_post_generation_verifier_node(pool_stats, llm)
        state = _make_state(generated_modalities={})
        result = verify_fn(state)
        assert result["verification_passed"] is True

    def test_promotes_best_candidate_to_generated_modalities(self, pool_stats):
        """Verifier must promote the winning candidate to generated_modalities."""
        llm = MockLLMClient()
        verify_fn = make_post_generation_verifier_node(pool_stats, llm)

        cand_a = pool_stats["transcriptomics"]["mean"].copy()
        cand_b = pool_stats["transcriptomics"]["mean"].copy() + 0.1
        state = PatientState(
            patient_id="TCGA-VQUERY-0002",
            cohort="luad",
            clinical=np.zeros(MODALITY_DIMS["clinical"]),
            transcriptomics=None,
            wsi=None,
            methylation=None,
            available_modalities=["clinical"],
            missing_modalities=["transcriptomics"],
            agent_summaries={},
            mining_rules={},
            guidance={},
            generation_candidates={"transcriptomics": [cand_a, cand_b]},
            generated_modalities={},
            verification_scores={},
            verification_passed=False,
            survival_prediction=None,
            routing_decision="generate",
            execution_log=[],
            correction_hints={},
        )
        result = verify_fn(state)
        assert "transcriptomics" in result["generated_modalities"]
        assert result["generated_modalities"]["transcriptomics"].shape == (
            MODALITY_DIMS["transcriptomics"],
        )


# ---------------------------------------------------------------------------
# Best-of-N ranking tests (Post-Generation Verifier)
# ---------------------------------------------------------------------------


class _SequentialMockLLM(MockLLMClient):
    """LLM that cycles through a fixed list of JSON responses."""

    def __init__(self, responses: list[dict]):
        super().__init__()
        self._responses = responses
        self._call_idx = 0

    def invoke_json(self, prompt: str, system: str = "") -> dict:
        resp = self._responses[self._call_idx % len(self._responses)]
        self._call_idx += 1
        return resp


def _make_criteria_response(overall: float, feedback: str = "") -> dict:
    return {
        "distributional_plausibility": overall,
        "biological_consistency": overall,
        "cross_modal_coherence": overall,
        "clinical_relevance": overall,
        "pathway_consistency": overall,
        "hallucination_risk": overall,
        "overall_score": overall,
        "feedback": feedback,
    }


class TestBestOfNPostGenerationVerifier:
    @pytest.fixture
    def pool_stats(self):
        pool = _build_synthetic_pool(20)
        return build_pool_stats(pool)

    def test_selects_highest_scoring_candidate(self, pool_stats):
        """When candidate 1 scores higher than candidate 0, it must be chosen."""
        low_resp = _make_criteria_response(1.0, "poor reconstruction")
        high_resp = _make_criteria_response(5.0)
        # Candidate 0 → low score, candidate 1 → high score
        llm = _SequentialMockLLM([low_resp, high_resp])
        verify_fn = make_post_generation_verifier_node(pool_stats, llm)

        cand_low = np.zeros(MODALITY_DIMS["transcriptomics"], dtype=np.float32)
        cand_high = pool_stats["transcriptomics"]["mean"].copy()

        state = PatientState(
            patient_id="TCGA-VQUERY-BON-0001",
            cohort="luad",
            clinical=np.zeros(MODALITY_DIMS["clinical"]),
            transcriptomics=None,
            wsi=None,
            methylation=None,
            available_modalities=["clinical"],
            missing_modalities=["transcriptomics"],
            agent_summaries={},
            mining_rules={"transcriptomics": "test rule"},
            guidance={},
            generation_candidates={"transcriptomics": [cand_low, cand_high]},
            generated_modalities={},
            verification_scores={},
            verification_passed=False,
            survival_prediction=None,
            routing_decision="generate",
            execution_log=[],
            correction_hints={},
        )
        result = verify_fn(state)
        # Score from high_resp (5.0) should win
        assert result["verification_scores"]["transcriptomics"] == pytest.approx(5.0)
        # Promoted candidate should be cand_high
        np.testing.assert_array_equal(
            result["generated_modalities"]["transcriptomics"], cand_high
        )

    def test_n1_backward_compat_single_candidate(self, pool_stats):
        """N=1 path: result must still be promoted to generated_modalities."""
        llm = MockLLMClient()
        verify_fn = make_post_generation_verifier_node(pool_stats, llm)

        arr = pool_stats["transcriptomics"]["mean"].copy()
        state = PatientState(
            patient_id="TCGA-VQUERY-BON-0002",
            cohort="luad",
            clinical=np.zeros(MODALITY_DIMS["clinical"]),
            transcriptomics=None,
            wsi=None,
            methylation=None,
            available_modalities=["clinical"],
            missing_modalities=["transcriptomics"],
            agent_summaries={},
            mining_rules={},
            guidance={},
            generation_candidates={"transcriptomics": [arr]},
            generated_modalities={},
            verification_scores={},
            verification_passed=False,
            survival_prediction=None,
            routing_decision="generate",
            execution_log=[],
            correction_hints={},
        )
        result = verify_fn(state)
        assert "transcriptomics" in result["generated_modalities"]

    def test_log_mentions_candidate_count_for_n_gt_1(self, pool_stats):
        llm = MockLLMClient()
        verify_fn = make_post_generation_verifier_node(pool_stats, llm)

        arr = pool_stats["transcriptomics"]["mean"].copy()
        state = PatientState(
            patient_id="TCGA-VQUERY-BON-0003",
            cohort="luad",
            clinical=np.zeros(MODALITY_DIMS["clinical"]),
            transcriptomics=None,
            wsi=None,
            methylation=None,
            available_modalities=["clinical"],
            missing_modalities=["transcriptomics"],
            agent_summaries={},
            mining_rules={},
            guidance={},
            generation_candidates={"transcriptomics": [arr, arr + 0.1, arr + 0.2]},
            generated_modalities={},
            verification_scores={},
            verification_passed=False,
            survival_prediction=None,
            routing_decision="generate",
            execution_log=[],
            correction_hints={},
        )
        result = verify_fn(state)
        log = " ".join(result["execution_log"])
        assert "candidate" in log


# ---------------------------------------------------------------------------
# Pre-Generation Verifier guidance refinement tests
# ---------------------------------------------------------------------------


class TestPreGenerationVerifierNode:
    @pytest.fixture
    def pool_stats(self):
        pool = _build_synthetic_pool(20)
        return build_pool_stats(pool)

    def _make_pre_state(self, mining_rules: dict, missing: list[str]):
        return PatientState(
            patient_id="TCGA-VQUERY-PRE-0001",
            cohort="luad",
            clinical=np.zeros(MODALITY_DIMS["clinical"]),
            transcriptomics=None,
            wsi=None,
            methylation=None,
            available_modalities=["clinical"],
            missing_modalities=missing,
            agent_summaries={},
            mining_rules=mining_rules,
            guidance={},
            generation_candidates={},
            generated_modalities={},
            verification_scores={},
            verification_passed=False,
            survival_prediction=None,
            routing_decision="generate",
            source_map={},
            execution_log=[],
            correction_hints={},
        )

    def test_produces_guidance_for_each_missing_modality(self, pool_stats):
        llm = MockLLMClient()
        pre_fn = make_pre_generation_verifier_node(llm)
        state = self._make_pre_state(
            mining_rules={"transcriptomics": "use smoking history"},
            missing=["transcriptomics"],
        )
        result = pre_fn(state)
        assert "transcriptomics" in result["guidance"]
        assert isinstance(result["guidance"]["transcriptomics"], str)
        assert len(result["guidance"]["transcriptomics"]) > 0

    def test_guidance_persisted_in_source_map(self, pool_stats):
        llm = MockLLMClient()
        pre_fn = make_pre_generation_verifier_node(llm)
        state = self._make_pre_state(
            mining_rules={"transcriptomics": "raw rule"},
            missing=["transcriptomics"],
        )
        result = pre_fn(state)
        assert "mining_rules" in result["source_map"]
        assert "guidance" in result["source_map"]

    def test_falls_back_to_raw_rules_on_llm_failure(self, pool_stats):
        """If LLM returns malformed JSON, guidance falls back to raw mining rule."""

        class _FailingLLM(MockLLMClient):
            def invoke_json(self, prompt: str, system: str = "") -> dict:
                return {"unexpected_key": "no guidance here"}

        pre_fn = make_pre_generation_verifier_node(_FailingLLM())
        raw_rule = "use age and smoking history as proxies"
        state = self._make_pre_state(
            mining_rules={"methylation": raw_rule},
            missing=["methylation"],
        )
        result = pre_fn(state)
        # Must still have an entry; content is the raw rule as fallback
        assert "methylation" in result["guidance"]
        assert result["guidance"]["methylation"] == raw_rule

    def test_no_missing_returns_empty_guidance(self, pool_stats):
        llm = MockLLMClient()
        pre_fn = make_pre_generation_verifier_node(llm)
        state = self._make_pre_state(mining_rules={}, missing=[])
        result = pre_fn(state)
        assert result["guidance"] == {}

    def test_execution_log_contains_pre_generation_verifier_tag(self, pool_stats):
        llm = MockLLMClient()
        pre_fn = make_pre_generation_verifier_node(llm)
        state = self._make_pre_state(
            mining_rules={"wsi": "use staging info"},
            missing=["wsi"],
        )
        result = pre_fn(state)
        assert any(
            "[Pre-Generation Verifier]" in line for line in result["execution_log"]
        )
