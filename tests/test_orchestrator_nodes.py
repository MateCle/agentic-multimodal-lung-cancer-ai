"""
Unit tests for the LangGraph orchestrator nodes (AFM2-aligned).
Tests mock node interfaces without requiring TCGA data or LLM API.
"""

import numpy as np

from src.data_loader import MODALITY_DIMS
from src.orchestrator.nodes.generator import generator_node
from src.orchestrator.nodes.miner import miner_node
from src.orchestrator.nodes.planner import planner_node
from src.orchestrator.nodes.predictor import predictor_node
from src.orchestrator.nodes.router import (
    MAX_REFINEMENT_ATTEMPTS,
    route_after_planner,
    route_after_post_generation_verifier,
)
from src.orchestrator.nodes.verifier import post_generation_verifier_node
from src.orchestrator.state import PatientState

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_state(
    available: list[str] | None = None,
    missing: list[str] | None = None,
    cohort: str = "luad",
    routing_decision: str = "",
    verification_passed: bool = False,
    execution_log: list[str] | None = None,
    generated_modalities: dict | None = None,
    mining_rules: dict | None = None,
) -> PatientState:
    """Build a minimal PatientState for testing."""
    if available is None:
        available = ["clinical", "transcriptomics", "wsi", "methylation"]
    if missing is None:
        missing = []

    return PatientState(
        patient_id="TCGA-TEST-0001",
        cohort=cohort,
        clinical=np.zeros(63) if "clinical" in available else None,
        transcriptomics=np.zeros(1824) if "transcriptomics" in available else None,
        wsi=np.zeros(1024) if "wsi" in available else None,
        methylation=np.zeros(16166) if "methylation" in available else None,
        available_modalities=available,
        missing_modalities=missing,
        agent_summaries={},
        mining_rules=mining_rules or {},
        generated_modalities=generated_modalities or {},
        verification_scores={},
        verification_passed=verification_passed,
        survival_prediction=None,
        routing_decision=routing_decision,
        execution_log=execution_log or [],
        correction_hints={},
    )


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


class TestPlannerNode:
    def test_all_present_routes_to_predict(self):
        state = _make_state(
            available=["clinical", "transcriptomics", "wsi", "methylation"],
            missing=[],
        )
        result = planner_node(state)
        assert result["routing_decision"] == "predict"

    def test_missing_modality_routes_to_generate(self):
        state = _make_state(
            available=["clinical", "wsi"],
            missing=["transcriptomics", "methylation"],
        )
        result = planner_node(state)
        assert result["routing_decision"] == "generate"

    def test_returns_execution_log(self):
        state = _make_state(missing=["transcriptomics"])
        result = planner_node(state)
        assert isinstance(result["execution_log"], list)
        assert len(result["execution_log"]) == 1
        assert "[Planner]" in result["execution_log"][0]


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


class TestRouteAfterPlanner:
    def test_generate_goes_to_miner(self):
        state = _make_state(routing_decision="generate")
        assert route_after_planner(state) == "miner"

    def test_predict_goes_to_predictor(self):
        state = _make_state(routing_decision="predict")
        assert route_after_planner(state) == "predictor"


class TestRouteAfterPostGenerationVerifier:
    def test_passed_goes_to_predictor(self):
        state = _make_state(verification_passed=True)
        assert route_after_post_generation_verifier(state) == "predictor"

    def test_failed_goes_to_generator(self):
        state = _make_state(verification_passed=False, execution_log=[])
        assert route_after_post_generation_verifier(state) == "generator"

    def test_max_attempts_reached_goes_to_predictor(self):
        logs = [
            "[Post-Generation Verifier] Overall: FAIL."
            for _ in range(MAX_REFINEMENT_ATTEMPTS)
        ]
        state = _make_state(verification_passed=False, execution_log=logs)
        assert route_after_post_generation_verifier(state) == "predictor"

    def test_under_max_attempts_retries(self):
        logs = [
            "[Post-Generation Verifier] Overall: FAIL."
            for _ in range(MAX_REFINEMENT_ATTEMPTS - 1)
        ]
        state = _make_state(verification_passed=False, execution_log=logs)
        assert route_after_post_generation_verifier(state) == "generator"


# ---------------------------------------------------------------------------
# Mock Miner
# ---------------------------------------------------------------------------


class TestMockMinerNode:
    def test_generates_rules_for_each_missing_modality(self):
        state = _make_state(
            available=["clinical"],
            missing=["transcriptomics", "wsi"],
        )
        result = miner_node(state)
        assert "transcriptomics" in result["mining_rules"]
        assert "wsi" in result["mining_rules"]

    def test_no_missing_returns_empty_rules(self):
        state = _make_state(missing=[])
        result = miner_node(state)
        assert result["mining_rules"] == {}

    def test_returns_execution_log(self):
        state = _make_state(missing=["methylation"])
        result = miner_node(state)
        assert any("[Miner]" in line for line in result["execution_log"])


# ---------------------------------------------------------------------------
# Mock Generator
# ---------------------------------------------------------------------------


class TestMockGeneratorNode:
    def test_generates_correct_shapes(self):
        state = _make_state(
            available=["clinical"],
            missing=["transcriptomics", "wsi", "methylation"],
            mining_rules={
                "transcriptomics": "mock rule",
                "wsi": "mock rule",
                "methylation": "mock rule",
            },
        )
        result = generator_node(state)
        # Generator now returns generation_candidates (list per modality)
        cands = result["generation_candidates"]
        assert cands["transcriptomics"][0].shape == (MODALITY_DIMS["transcriptomics"],)
        assert cands["wsi"][0].shape == (MODALITY_DIMS["wsi"],)
        assert cands["methylation"][0].shape == (MODALITY_DIMS["methylation"],)

    def test_no_missing_generates_nothing(self):
        state = _make_state(missing=[])
        result = generator_node(state)
        assert result["generation_candidates"] == {}


# ---------------------------------------------------------------------------
# Mock Post-Generation Verifier (AFM2: scores 1-5)
# ---------------------------------------------------------------------------


class TestMockPostGenerationVerifierNode:
    def test_scores_all_generated_modalities(self):
        state = _make_state(
            generated_modalities={
                "transcriptomics": np.zeros(1824),
                "wsi": np.zeros(1024),
            },
        )
        result = post_generation_verifier_node(state)
        assert "transcriptomics" in result["verification_scores"]
        assert "wsi" in result["verification_scores"]

    def test_scores_are_between_1_and_5(self):
        state = _make_state(
            generated_modalities={"transcriptomics": np.zeros(1824)},
        )
        result = post_generation_verifier_node(state)
        score = result["verification_scores"]["transcriptomics"]
        assert 1.0 <= score <= 5.0

    def test_verification_passed_is_bool(self):
        state = _make_state(
            generated_modalities={"clinical": np.zeros(63)},
        )
        result = post_generation_verifier_node(state)
        assert isinstance(result["verification_passed"], bool)

    def test_empty_generated_passes(self):
        state = _make_state(generated_modalities={})
        result = post_generation_verifier_node(state)
        assert result["verification_passed"] is True


# ---------------------------------------------------------------------------
# Mock Predictor
# ---------------------------------------------------------------------------


class TestMockPredictorNode:
    def test_returns_float_prediction(self):
        state = _make_state()
        result = predictor_node(state)
        assert isinstance(result["survival_prediction"], float)

    def test_prediction_in_valid_range(self):
        state = _make_state()
        result = predictor_node(state)
        assert 0.0 <= result["survival_prediction"] <= 1.0

    def test_returns_execution_log(self):
        state = _make_state()
        result = predictor_node(state)
        assert any("[Predictor]" in line for line in result["execution_log"])
