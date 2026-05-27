"""
Unit tests for the real k-NN Generator node.
Uses synthetic patient data — no TCGA files required.
"""

import numpy as np
import pytest

from src.data_loader import MODALITY_DIMS, MODALITY_KEYS
from src.orchestrator.nodes.generator import (
    BASE_K,
    DEFAULT_N_CANDIDATES,
    K_INCREMENT,
    _knn_retrieve,
    _knn_retrieve_candidates,
    build_pool_index,
    make_generator_node,
)
from src.orchestrator.state import PatientState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_raw_patient(patient_id, available, seed=0):
    rng = np.random.default_rng(seed)
    avail = [1.0 if mod in available else 0.0 for mod in MODALITY_KEYS]
    record = {"avail": np.array(avail), "label": {"DSS": 1.0, "DSS.time": 500.0}}
    for mod in MODALITY_KEYS:
        if mod in available:
            record[mod] = rng.standard_normal(MODALITY_DIMS[mod]).astype(np.float32)
        else:
            record[mod] = None
    return {patient_id: record}


def _build_synthetic_pool(n_patients=20):
    raw_data = {}
    patient_ids = []
    for i in range(n_patients):
        pid = f"TCGA-SYNTH-{i:04d}"
        available = ["clinical"]
        if i % 2 == 0:
            available.append("transcriptomics")
        if i % 3 == 0:
            available.append("wsi")
        if i % 7 == 0:
            available.append("methylation")
        record = _make_raw_patient(pid, available, seed=i)
        raw_data.update(record)
        patient_ids.append(pid)
    return raw_data, patient_ids


def _make_state(
    available,
    missing,
    features=None,
    execution_log=None,
    mining_rules=None,
    correction_hints=None,
    guidance=None,
    seed=0,
):
    if features is None:
        features = {}
        rng = np.random.default_rng(seed)
        for mod in available:
            features[mod] = rng.standard_normal(MODALITY_DIMS[mod]).astype(np.float32)

    return PatientState(
        patient_id="TCGA-QUERY-0001",
        cohort="luad",
        clinical=features.get("clinical"),
        transcriptomics=features.get("transcriptomics"),
        wsi=features.get("wsi"),
        methylation=features.get("methylation"),
        available_modalities=available,
        missing_modalities=missing,
        agent_summaries={},
        mining_rules=mining_rules or {},
        guidance=guidance or {},
        generation_candidates={},
        generated_modalities={},
        verification_scores={},
        verification_passed=False,
        survival_prediction=None,
        routing_decision="generate",
        execution_log=execution_log or [],
        correction_hints=correction_hints or {},
    )


# ---------------------------------------------------------------------------
# Pool index tests
# ---------------------------------------------------------------------------


class TestBuildPoolIndex:
    def test_builds_correct_number(self):
        raw_data, pids = _build_synthetic_pool(10)
        pool = build_pool_index(raw_data, pids)
        assert len(pool) == 10

    def test_entries_have_correct_keys(self):
        raw_data, pids = _build_synthetic_pool(5)
        pool = build_pool_index(raw_data, pids)
        for entry in pool:
            assert "patient_id" in entry
            assert "features" in entry
            assert "features_norm" in entry

    def test_normalized_features_have_unit_norm(self):
        raw_data, pids = _build_synthetic_pool(5)
        pool = build_pool_index(raw_data, pids)
        for entry in pool:
            for _mod, arr in entry["features_norm"].items():
                norm = np.linalg.norm(arr)
                assert np.isclose(norm, 1.0, atol=1e-5) or np.isclose(norm, 0.0)


# ---------------------------------------------------------------------------
# k-NN retrieval tests
# ---------------------------------------------------------------------------


class TestKnnRetrieve:
    @pytest.fixture
    def pool(self):
        raw_data, pids = _build_synthetic_pool(20)
        return build_pool_index(raw_data, pids)

    def test_output_shape_matches_target(self, pool):
        query = {
            "clinical": np.random.default_rng(0).standard_normal(63).astype(np.float32)
        }
        result, _ = _knn_retrieve(query, "transcriptomics", pool, k=3)
        assert result.shape == (MODALITY_DIMS["transcriptomics"],)

    def test_returns_correct_number_of_neighbors(self, pool):
        query = {
            "clinical": np.random.default_rng(1).standard_normal(63).astype(np.float32)
        }
        _, neighbors = _knn_retrieve(query, "transcriptomics", pool, k=3)
        assert len(neighbors) <= 3

    def test_neighbors_sorted_by_similarity(self, pool):
        query = {
            "clinical": np.random.default_rng(2).standard_normal(63).astype(np.float32)
        }
        _, neighbors = _knn_retrieve(query, "transcriptomics", pool, k=5)
        if len(neighbors) > 1:
            sims = [n["similarity"] for n in neighbors]
            assert sims == sorted(sims, reverse=True)

    def test_excludes_specified_patient(self, pool):
        exclude_pid = pool[0]["patient_id"]
        query = {
            "clinical": np.random.default_rng(3).standard_normal(63).astype(np.float32)
        }
        _, neighbors = _knn_retrieve(
            query, "transcriptomics", pool, k=20, exclude_pid=exclude_pid
        )
        neighbor_ids = {n["patient_id"] for n in neighbors}
        assert exclude_pid not in neighbor_ids

    def test_no_candidates_returns_zeros(self):
        query = {
            "clinical": np.random.default_rng(4).standard_normal(63).astype(np.float32)
        }
        result, neighbors = _knn_retrieve(query, "transcriptomics", [], k=3)
        np.testing.assert_array_equal(
            result, np.zeros(MODALITY_DIMS["transcriptomics"])
        )
        assert neighbors == []


# ---------------------------------------------------------------------------
# Generator node tests
# ---------------------------------------------------------------------------


class TestMakeGeneratorNode:
    @pytest.fixture
    def pool(self):
        raw_data, pids = _build_synthetic_pool(20)
        return build_pool_index(raw_data, pids)

    def test_generates_correct_shapes(self, pool):
        gen_fn = make_generator_node(pool, n_candidates=1)
        state = _make_state(
            available=["clinical", "wsi"],
            missing=["transcriptomics"],
            seed=5,
        )
        result = gen_fn(state)
        candidates = result["generation_candidates"]["transcriptomics"]
        assert len(candidates) == 1
        assert candidates[0].shape == (MODALITY_DIMS["transcriptomics"],)

    def test_generates_all_missing(self, pool):
        gen_fn = make_generator_node(pool, n_candidates=1)
        state = _make_state(
            available=["clinical"],
            missing=["transcriptomics", "wsi", "methylation"],
            seed=6,
        )
        result = gen_fn(state)
        assert set(result["generation_candidates"].keys()) == {
            "transcriptomics",
            "wsi",
            "methylation",
        }

    def test_generated_values_not_all_zeros(self, pool):
        gen_fn = make_generator_node(pool, n_candidates=1)
        state = _make_state(
            available=["clinical"],
            missing=["transcriptomics"],
            seed=7,
        )
        result = gen_fn(state)
        assert not np.allclose(
            result["generation_candidates"]["transcriptomics"][0], 0.0
        )

    def test_self_refinement_increases_k(self, pool):
        gen_fn = make_generator_node(pool, n_candidates=1)
        state = _make_state(
            available=["clinical"],
            missing=["transcriptomics"],
            execution_log=["[Generator] Completed attempt 1 for TCGA-QUERY-0001."],
            seed=8,
        )
        result = gen_fn(state)
        log = " ".join(result["execution_log"])
        expected_k = BASE_K + K_INCREMENT
        assert f"k={expected_k}" in log

    def test_no_available_modalities_falls_back_to_zeros(self, pool):
        gen_fn = make_generator_node(pool, n_candidates=1)
        state = _make_state(
            available=[],
            missing=["clinical", "transcriptomics", "wsi", "methylation"],
            features={},
            seed=9,
        )
        result = gen_fn(state)
        for mod in ["clinical", "transcriptomics", "wsi", "methylation"]:
            np.testing.assert_array_equal(
                result["generation_candidates"][mod][0],
                np.zeros(MODALITY_DIMS[mod]),
            )

    def test_uses_guidance_over_mining_rules(self, pool):
        """Generator must prefer state['guidance'] when both are present."""
        gen_fn = make_generator_node(pool, n_candidates=1)
        state = _make_state(
            available=["clinical"],
            missing=["transcriptomics"],
            mining_rules={"transcriptomics": "raw rule"},
            guidance={
                "transcriptomics": "refined guidance from Pre-Generation Verifier"
            },
            seed=10,
        )
        result = gen_fn(state)
        log = " ".join(result["execution_log"])
        assert "refined guidance from Pre-Generation Verifier" in log


# ---------------------------------------------------------------------------
# N-candidate retrieval tests
# ---------------------------------------------------------------------------


class TestKnnRetrieveCandidates:
    @pytest.fixture
    def pool(self):
        raw_data, pids = _build_synthetic_pool(20)
        return build_pool_index(raw_data, pids)

    def test_returns_exactly_n_candidates(self, pool):
        query = {
            "clinical": np.random.default_rng(0)
            .standard_normal(MODALITY_DIMS["clinical"])
            .astype(np.float32)
        }
        for n in (1, 2, 3):
            candidates, _ = _knn_retrieve_candidates(
                query, "transcriptomics", pool, k=3, n_candidates=n
            )
            assert len(candidates) == n

    def test_candidate_shape_matches_modality_dim(self, pool):
        query = {
            "clinical": np.random.default_rng(1)
            .standard_normal(MODALITY_DIMS["clinical"])
            .astype(np.float32)
        }
        candidates, _ = _knn_retrieve_candidates(
            query, "transcriptomics", pool, k=3, n_candidates=3
        )
        for c in candidates:
            assert c.shape == (MODALITY_DIMS["transcriptomics"],)

    def test_candidates_are_distinct(self, pool):
        """Different chunks of neighbours should produce different averages."""
        query = {
            "clinical": np.random.default_rng(2)
            .standard_normal(MODALITY_DIMS["clinical"])
            .astype(np.float32)
        }
        candidates, _ = _knn_retrieve_candidates(
            query, "transcriptomics", pool, k=3, n_candidates=3
        )
        # At least two candidates must differ (unless pool is trivially small)
        if len(candidates) >= 2:
            assert not np.allclose(candidates[0], candidates[1])

    def test_empty_pool_returns_zeros(self):
        query = {
            "clinical": np.random.default_rng(3)
            .standard_normal(MODALITY_DIMS["clinical"])
            .astype(np.float32)
        }
        candidates, info = _knn_retrieve_candidates(
            query, "transcriptomics", [], k=3, n_candidates=3
        )
        assert len(candidates) == 3
        for c in candidates:
            np.testing.assert_array_equal(c, np.zeros(MODALITY_DIMS["transcriptomics"]))
        assert info == []

    def test_default_n_candidates_constant(self):
        assert DEFAULT_N_CANDIDATES == 3

    def test_generator_node_returns_n_candidates(self, pool):
        """make_generator_node with N=3 produces 3 candidates per modality."""
        gen_fn = make_generator_node(pool, n_candidates=3)
        state = _make_state(
            available=["clinical"],
            missing=["transcriptomics"],
            seed=11,
        )
        result = gen_fn(state)
        candidates = result["generation_candidates"]["transcriptomics"]
        assert len(candidates) == 3
        for c in candidates:
            assert c.shape == (MODALITY_DIMS["transcriptomics"],)
