"""
Unit tests for the real k-NN Generator node.
Uses synthetic patient data — no TCGA files required.
"""

import numpy as np
import pytest

from src.data_loader import MODALITY_DIMS, MODALITY_KEYS
from src.orchestrator.nodes.generator import (
    BASE_K,
    K_INCREMENT,
    _knn_retrieve,
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
    for i, mod in enumerate(MODALITY_KEYS):
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
            for mod, arr in entry["features_norm"].items():
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
        gen_fn = make_generator_node(pool)
        state = _make_state(
            available=["clinical", "wsi"],
            missing=["transcriptomics"],
            seed=5,
        )
        result = gen_fn(state)
        assert result["generated_modalities"]["transcriptomics"].shape == (
            MODALITY_DIMS["transcriptomics"],
        )

    def test_generates_all_missing(self, pool):
        gen_fn = make_generator_node(pool)
        state = _make_state(
            available=["clinical"],
            missing=["transcriptomics", "wsi", "methylation"],
            seed=6,
        )
        result = gen_fn(state)
        assert set(result["generated_modalities"].keys()) == {
            "transcriptomics",
            "wsi",
            "methylation",
        }

    def test_generated_values_not_all_zeros(self, pool):
        gen_fn = make_generator_node(pool)
        state = _make_state(
            available=["clinical"],
            missing=["transcriptomics"],
            seed=7,
        )
        result = gen_fn(state)
        assert not np.allclose(result["generated_modalities"]["transcriptomics"], 0.0)

    def test_self_refinement_increases_k(self, pool):
        gen_fn = make_generator_node(pool)
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
        gen_fn = make_generator_node(pool)
        state = _make_state(
            available=[],
            missing=["clinical", "transcriptomics", "wsi", "methylation"],
            features={},
            seed=9,
        )
        result = gen_fn(state)
        for mod in ["clinical", "transcriptomics", "wsi", "methylation"]:
            np.testing.assert_array_equal(
                result["generated_modalities"][mod],
                np.zeros(MODALITY_DIMS[mod]),
            )
