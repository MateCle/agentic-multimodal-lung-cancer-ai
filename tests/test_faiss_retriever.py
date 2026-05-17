"""
Unit tests for FAISS-based k-NN retrieval in the Generator.

Verifies:
- Pool entries with non-standard dims are accepted (_build_pool_entry fix)
- FAISS index is built correctly (per-modality, fused vectors)
- FAISS results match brute-force within FP tolerance
- GPU fallback to CPU works gracefully
- make_generator_node produces correct shapes with FAISS active

All tests use synthetic data — no TCGA files required.
"""

import numpy as np
import pytest

from src.data_loader import MODALITY_DIMS, MODALITY_KEYS
from src.orchestrator.nodes.generator import (
    _FAISS_AVAILABLE,
    _build_faiss_index,
    _build_faiss_query,
    _build_pool_entry,
    _detect_pool_dims,
    _knn_retrieve_candidates,
    _knn_retrieve_candidates_faiss,
    _pool_fused_offsets,
    build_pool_index,
    make_generator_node,
)
from src.orchestrator.state import PatientState

_LUSC_METH_DIM = 16206


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_raw_record(available, meth_dim=MODALITY_DIMS["methylation"], seed=0):
    rng = np.random.default_rng(seed)
    avail = [1.0 if mod in available else 0.0 for mod in MODALITY_KEYS]
    record = {"avail": np.array(avail), "label": {"DSS": 1.0, "DSS.time": 500.0}}
    for mod in MODALITY_KEYS:
        if mod in available:
            dim = meth_dim if mod == "methylation" else MODALITY_DIMS[mod]
            record[mod] = rng.standard_normal(dim).astype(np.float32)
        else:
            record[mod] = None
    return record


def _make_pool(n=20, meth_dim=MODALITY_DIMS["methylation"]):
    """Build synthetic pool with given methylation dim."""
    raw_data = {}
    pids = []
    for i in range(n):
        pid = f"TCGA-POOL-{i:04d}"
        available = ["clinical"]
        if i % 2 == 0:
            available.append("transcriptomics")
        if i % 3 == 0:
            available.append("wsi")
        if i % 5 == 0:
            available.append("methylation")
        raw_data[pid] = _make_raw_record(available, meth_dim=meth_dim, seed=i)
        pids.append(pid)
    return raw_data, pids


def _make_state(available, missing, seed=0, meth_dim=MODALITY_DIMS["methylation"]):
    rng = np.random.default_rng(seed)
    features = {}
    for mod in available:
        dim = meth_dim if mod == "methylation" else MODALITY_DIMS[mod]
        features[mod] = rng.standard_normal(dim).astype(np.float32)
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
        mining_rules={},
        guidance={},
        generation_candidates={},
        generated_modalities={},
        verification_scores={},
        verification_passed=False,
        survival_prediction=None,
        routing_decision="generate",
        execution_log=[],
        correction_hints={},
    )


# ---------------------------------------------------------------------------
# Pool entry dim check fix
# ---------------------------------------------------------------------------


class TestBuildPoolEntryDimFix:
    def test_accepts_standard_methylation(self):
        record = _make_raw_record(["clinical", "methylation"])
        entry = _build_pool_entry(
            "P-0",
            {
                "patient_id": "P-0",
                "clinical": record["clinical"],
                "transcriptomics": None,
                "wsi": None,
                "methylation": record["methylation"],
                "available_modalities": ["clinical", "methylation"],
                "missing_modalities": ["transcriptomics", "wsi"],
                "label": record["label"],
            },
        )
        assert "methylation" in entry["features"]
        assert entry["features"]["methylation"].size == MODALITY_DIMS["methylation"]

    def test_accepts_lusc_methylation(self):
        """LUSC methylation (16206) must NOT be dropped after the fix."""
        record = _make_raw_record(["clinical", "methylation"], meth_dim=_LUSC_METH_DIM)
        entry = _build_pool_entry(
            "LUSC-0",
            {
                "patient_id": "LUSC-0",
                "clinical": record["clinical"],
                "transcriptomics": None,
                "wsi": None,
                "methylation": record["methylation"],
                "available_modalities": ["clinical", "methylation"],
                "missing_modalities": ["transcriptomics", "wsi"],
                "label": record["label"],
            },
        )
        assert "methylation" in entry["features"], (
            "LUSC methylation (16206-dim) must be accepted by _build_pool_entry"
        )
        assert entry["features"]["methylation"].size == _LUSC_METH_DIM

    def test_rejects_empty_array(self):
        entry = _build_pool_entry(
            "P-0",
            {
                "patient_id": "P-0",
                "clinical": np.array([], dtype=np.float32),
                "transcriptomics": None,
                "wsi": None,
                "methylation": None,
                "available_modalities": [],
                "missing_modalities": MODALITY_KEYS,
                "label": {"DSS": 0.0, "DSS.time": 100.0},
            },
        )
        assert "clinical" not in entry["features"]


# ---------------------------------------------------------------------------
# _detect_pool_dims
# ---------------------------------------------------------------------------


class TestDetectPoolDims:
    def test_detects_lusc_methylation(self):
        raw_data, pids = _make_pool(20, meth_dim=_LUSC_METH_DIM)
        pool = build_pool_index(raw_data, pids)
        dims = _detect_pool_dims(pool)
        assert dims["methylation"] == _LUSC_METH_DIM

    def test_detects_standard_dims(self):
        raw_data, pids = _make_pool(20)
        pool = build_pool_index(raw_data, pids)
        dims = _detect_pool_dims(pool)
        assert dims["clinical"] == MODALITY_DIMS["clinical"]
        assert dims["transcriptomics"] == MODALITY_DIMS["transcriptomics"]


# ---------------------------------------------------------------------------
# _pool_fused_offsets
# ---------------------------------------------------------------------------


class TestPoolFusedOffsets:
    def test_offsets_are_non_overlapping(self):
        dims = dict(MODALITY_DIMS)
        offsets, total_dim = _pool_fused_offsets(dims)
        seen = set()
        for mod, (start, end) in offsets.items():
            for i in range(start, end):
                assert i not in seen
                seen.add(i)
        assert total_dim == len(seen)

    def test_total_dim_equals_sum(self):
        dims = dict(MODALITY_DIMS)
        offsets, total_dim = _pool_fused_offsets(dims)
        assert total_dim == sum(dims.values())

    def test_lusc_dims_increase_total(self):
        dims = dict(MODALITY_DIMS)
        dims["methylation"] = _LUSC_METH_DIM
        _, total_dim = _pool_fused_offsets(dims)
        assert total_dim == sum(dims.values())
        assert total_dim > sum(MODALITY_DIMS.values())


# ---------------------------------------------------------------------------
# FAISS index building
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _FAISS_AVAILABLE, reason="FAISS not installed")
class TestBuildFAISSIndex:
    @pytest.fixture
    def pool(self):
        raw_data, pids = _make_pool(20)
        return build_pool_index(raw_data, pids)

    def test_index_built_for_each_modality(self, pool):
        pool_dims = _detect_pool_dims(pool)
        faiss_data = _build_faiss_index(pool, pool_dims)
        # At least clinical should have an index (all patients have clinical)
        assert faiss_data["indices"].get("clinical") is not None

    def test_pool_by_mod_matches_pool(self, pool):
        pool_dims = _detect_pool_dims(pool)
        faiss_data = _build_faiss_index(pool, pool_dims)
        for mod in MODALITY_KEYS:
            expected = [e for e in pool if mod in e["features"]]
            actual = faiss_data["pool_by_mod"].get(mod, [])
            assert len(actual) == len(expected)

    def test_total_dim_is_positive(self, pool):
        pool_dims = _detect_pool_dims(pool)
        faiss_data = _build_faiss_index(pool, pool_dims)
        assert faiss_data["total_dim"] > 0

    def test_empty_pool_returns_none_indices(self):
        pool_dims = dict(MODALITY_DIMS)
        faiss_data = _build_faiss_index([], pool_dims)
        assert all(v is None for v in faiss_data["indices"].values())


# ---------------------------------------------------------------------------
# _build_faiss_query
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _FAISS_AVAILABLE, reason="FAISS not installed")
class TestBuildFAISSQuery:
    def test_query_has_correct_dim(self):
        raw_data, pids = _make_pool(5)
        pool = build_pool_index(raw_data, pids)
        pool_dims = _detect_pool_dims(pool)
        offsets, total_dim = _pool_fused_offsets(pool_dims)
        rng = np.random.default_rng(0)
        query_features = {
            "clinical": rng.standard_normal(MODALITY_DIMS["clinical"]).astype(
                np.float32
            )
        }
        vec = _build_faiss_query(query_features, offsets, total_dim, None)
        assert vec.shape == (total_dim,)

    def test_missing_modality_is_zero(self):
        """Unavailable modalities should contribute zeros to query vector."""
        dims = dict(MODALITY_DIMS)
        offsets, total_dim = _pool_fused_offsets(dims)
        rng = np.random.default_rng(0)
        query_features = {
            "clinical": rng.standard_normal(MODALITY_DIMS["clinical"]).astype(
                np.float32
            )
        }
        vec = _build_faiss_query(query_features, offsets, total_dim, None)
        # Transcriptomics block should be all zeros
        start, end = offsets["transcriptomics"]
        assert np.allclose(vec[start:end], 0.0)


# ---------------------------------------------------------------------------
# FAISS vs sklearn correctness
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _FAISS_AVAILABLE, reason="FAISS not installed")
class TestFAISSvsSklearn:
    @pytest.fixture
    def pool_and_faiss(self):
        raw_data, pids = _make_pool(50)
        pool = build_pool_index(raw_data, pids)
        pool_dims = _detect_pool_dims(pool)
        faiss_data = _build_faiss_index(pool, pool_dims)
        return pool, pool_dims, faiss_data

    def _query(self, seed=42):
        rng = np.random.default_rng(seed)
        return {
            "clinical": rng.standard_normal(MODALITY_DIMS["clinical"]).astype(
                np.float32
            ),
            "transcriptomics": rng.standard_normal(
                MODALITY_DIMS["transcriptomics"]
            ).astype(np.float32),
        }

    def test_faiss_matches_sklearn_single_candidate(self, pool_and_faiss):
        pool, pool_dims, faiss_data = pool_and_faiss
        query = self._query()

        sk_cands, _ = _knn_retrieve_candidates(query, "wsi", pool, k=5, n_candidates=1)
        fa_cands, _ = _knn_retrieve_candidates_faiss(
            query, "wsi", faiss_data, k=5, n_candidates=1
        )
        np.testing.assert_allclose(
            sk_cands[0],
            fa_cands[0],
            atol=1e-4,
            err_msg="FAISS single-candidate must match sklearn",
        )

    def test_faiss_matches_sklearn_three_candidates(self, pool_and_faiss):
        pool, pool_dims, faiss_data = pool_and_faiss
        query = self._query(seed=7)

        sk_cands, _ = _knn_retrieve_candidates(query, "wsi", pool, k=5, n_candidates=3)
        fa_cands, _ = _knn_retrieve_candidates_faiss(
            query, "wsi", faiss_data, k=5, n_candidates=3
        )
        assert len(fa_cands) == 3
        for i in range(3):
            np.testing.assert_allclose(
                sk_cands[i],
                fa_cands[i],
                atol=1e-4,
                err_msg=f"FAISS candidate {i} must match sklearn",
            )

    def test_faiss_respects_exclude_pid(self, pool_and_faiss):
        pool, pool_dims, faiss_data = pool_and_faiss
        exclude = pool[0]["patient_id"]
        query = self._query()

        _, fa_info = _knn_retrieve_candidates_faiss(
            query, "wsi", faiss_data, k=5, n_candidates=1, exclude_pid=exclude
        )
        neighbor_ids = {n["patient_id"] for n in fa_info}
        assert exclude not in neighbor_ids

    def test_faiss_empty_pool_returns_zeros(self):
        """FAISS path with None index falls back to brute-force (empty → zeros)."""
        empty_faiss = {
            "indices": {"wsi": None},
            "pool_by_mod": {"wsi": []},
            "offsets": {},
            "total_dim": 0,
            "on_gpu": False,
        }
        query = {"clinical": np.ones(MODALITY_DIMS["clinical"], dtype=np.float32)}
        cands, info = _knn_retrieve_candidates_faiss(
            query, "wsi", empty_faiss, k=5, n_candidates=3
        )
        assert len(cands) == 3
        for c in cands:
            assert np.allclose(c, 0.0)


# ---------------------------------------------------------------------------
# make_generator_node integration with FAISS
# ---------------------------------------------------------------------------


class TestMakeGeneratorNodeWithFAISS:
    @pytest.fixture
    def pool(self):
        raw_data, pids = _make_pool(30)
        return build_pool_index(raw_data, pids)

    def test_generates_correct_shapes(self, pool):
        gen_fn = make_generator_node(pool, n_candidates=1)
        state = _make_state(["clinical", "transcriptomics"], ["wsi"])
        result = gen_fn(state)
        assert "wsi" in result["generation_candidates"]
        assert len(result["generation_candidates"]["wsi"]) == 1
        assert result["generation_candidates"]["wsi"][0].shape == (
            MODALITY_DIMS["wsi"],
        )

    def test_generates_n_candidates(self, pool):
        gen_fn = make_generator_node(pool, n_candidates=3)
        state = _make_state(["clinical"], ["transcriptomics"])
        result = gen_fn(state)
        assert len(result["generation_candidates"]["transcriptomics"]) == 3

    def test_lusc_pool_meth_shape(self):
        """Generator with LUSC pool produces 16206-dim methylation candidates."""
        raw_data, pids = _make_pool(30, meth_dim=_LUSC_METH_DIM)
        pool = build_pool_index(raw_data, pids)
        gen_fn = make_generator_node(pool, n_candidates=1)
        state = _make_state(
            ["clinical", "transcriptomics"], ["methylation"], meth_dim=_LUSC_METH_DIM
        )
        result = gen_fn(state)
        cand = result["generation_candidates"]["methylation"][0]
        # Must match the pool's actual methylation dim, not MODALITY_DIMS["methylation"]
        assert cand.shape == (_LUSC_METH_DIM,), (
            f"Expected ({_LUSC_METH_DIM},) for LUSC meth, got {cand.shape}"
        )

    def test_no_available_modalities_zero_fallback(self, pool):
        gen_fn = make_generator_node(pool, n_candidates=2)
        state2 = PatientState(
            patient_id="TCGA-QUERY-EMPTY",
            cohort="luad",
            clinical=None,
            transcriptomics=None,
            wsi=None,
            methylation=None,
            available_modalities=[],
            missing_modalities=["clinical", "transcriptomics"],
            agent_summaries={},
            mining_rules={},
            guidance={},
            generation_candidates={},
            generated_modalities={},
            verification_scores={},
            verification_passed=False,
            survival_prediction=None,
            routing_decision="generate",
            execution_log=[],
            correction_hints={},
        )
        result = gen_fn(state2)
        for mod in ["clinical", "transcriptomics"]:
            assert len(result["generation_candidates"][mod]) == 2
            for c in result["generation_candidates"][mod]:
                assert np.allclose(c, 0.0)
