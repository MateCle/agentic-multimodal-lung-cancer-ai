"""
Unit tests for src/orchestrator/reliability.py

All tests use synthetic inputs — no TCGA data, no LLM, no GPU required.
CuPy-specific paths are exercised only when the library is installed; all
other tests always run and cover the NumPy fallback path.
"""

import numpy as np
import pytest

from src.orchestrator.reliability import (
    _mahalanobis_percentile,
    _run_mahalanobis_numpy,
    compute_bootstrap_ci,
    compute_mahalanobis_ood,
    compute_prediction_reliability,
    compute_provenance_proportion,
)

# ---------------------------------------------------------------------------
# Minimal pipeline stub
# ---------------------------------------------------------------------------


class _MockModel:
    """Predicts the mean of the input vector (deterministic, differentiable)."""

    def predict_risk(self, x: np.ndarray) -> np.ndarray:
        return np.array([float(np.mean(x))], dtype=np.float32)


class _MockPipeline:
    """Minimal FittedPipeline stub for reliability tests."""

    def __init__(self, n_components: int = 50, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.model = _MockModel()
        # Simulate PCA eigenvalues decreasing from ~10 to ~0.1
        self.pca = type(
            "PCA",
            (),
            {
                "explained_variance_": np.linspace(10.0, 0.1, n_components).astype(
                    np.float32
                )
            },
        )()


# ---------------------------------------------------------------------------
# compute_provenance_proportion
# ---------------------------------------------------------------------------


class TestComputeProvenanceProportion:
    def test_all_real_returns_one(self):
        source_map = {
            "clinical": {"source": "real"},
            "transcriptomics": {"source": "real"},
            "wsi": {"source": "real"},
            "methylation": {"source": "real"},
        }
        assert compute_provenance_proportion(source_map) == pytest.approx(1.0)

    def test_all_generated_returns_zero(self):
        source_map = {
            "clinical": {"source": "generated"},
            "transcriptomics": {"source": "generated"},
            "wsi": {"source": "generated"},
            "methylation": {"source": "generated"},
        }
        assert compute_provenance_proportion(source_map) == pytest.approx(0.0)

    def test_empty_source_map_returns_one(self):
        assert compute_provenance_proportion({}) == pytest.approx(1.0)

    def test_partial_real_weighted_by_dim(self):
        # Only clinical (63 dims) is real; methylation (16166), transcriptomics (1824),
        # wsi (1024) are generated.  Total = 63 + 1824 + 1024 + 16166 = 19077.
        source_map = {
            "clinical": {"source": "real"},
            "transcriptomics": {"source": "generated"},
            "wsi": {"source": "generated"},
            "methylation": {"source": "generated"},
        }
        result = compute_provenance_proportion(source_map)
        assert 0.0 < result < 0.1  # clinical is tiny relative to total dims

    def test_result_in_unit_interval(self):
        source_map = {
            "clinical": {"source": "real"},
            "wsi": {"source": "zero"},
            "methylation": {"source": "generated"},
        }
        result = compute_provenance_proportion(source_map)
        assert 0.0 <= result <= 1.0

    def test_unknown_source_not_counted_as_real(self):
        source_map = {"clinical": {"source": "unknown"}}
        assert compute_provenance_proportion(source_map) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _run_mahalanobis_numpy (internal helper, always runs)
# ---------------------------------------------------------------------------


class TestRunMahalanobisNumpy:
    def test_zero_vector_gives_zero_distance(self):
        n = 50
        x = np.zeros(n, dtype=np.float64)
        centroid = np.zeros(n, dtype=np.float64)
        inv_var = np.ones(n, dtype=np.float64)
        assert _run_mahalanobis_numpy(x, centroid, inv_var) == pytest.approx(0.0)

    def test_unit_vector_identity_covariance(self):
        n = 50
        x = np.ones(n, dtype=np.float64)
        centroid = np.zeros(n, dtype=np.float64)
        inv_var = np.ones(n, dtype=np.float64)
        # d = sqrt(n * 1^2 * 1) = sqrt(n)
        assert _run_mahalanobis_numpy(x, centroid, inv_var) == pytest.approx(
            float(np.sqrt(n)), rel=1e-5
        )

    def test_non_trivial_covariance(self):
        x = np.array([2.0, 3.0], dtype=np.float64)
        centroid = np.array([0.0, 1.0], dtype=np.float64)
        inv_var = np.array([0.25, 1.0], dtype=np.float64)
        # diff = [2, 2]; d^2 = 4*0.25 + 4*1.0 = 1 + 4 = 5; d = sqrt(5)
        expected = float(np.sqrt(5.0))
        assert _run_mahalanobis_numpy(x, centroid, inv_var) == pytest.approx(
            expected, rel=1e-5
        )

    def test_output_is_non_negative(self):
        rng = np.random.default_rng(7)
        x = rng.standard_normal(50).astype(np.float64)
        centroid = rng.standard_normal(50).astype(np.float64)
        inv_var = np.abs(rng.standard_normal(50)).astype(np.float64) + 0.1
        dist = _run_mahalanobis_numpy(x, centroid, inv_var)
        assert dist >= 0.0


# ---------------------------------------------------------------------------
# _mahalanobis_percentile
# ---------------------------------------------------------------------------


class TestMahalanobisPercentile:
    def test_median_chi_sq_near_50(self):
        from scipy.stats import chi2

        n = 50
        median_sq = chi2.median(n)
        pct = _mahalanobis_percentile(float(median_sq), n)
        assert 45.0 < pct < 55.0

    def test_large_distance_gives_high_percentile(self):
        pct = _mahalanobis_percentile(1_000.0, 50)
        assert pct > 99.0

    def test_zero_distance_gives_zero_percentile(self):
        pct = _mahalanobis_percentile(0.0, 50)
        assert pct == pytest.approx(0.0, abs=1e-6)

    def test_result_in_0_100(self):
        for d_sq in [0.0, 1.0, 50.0, 200.0, 1000.0]:
            pct = _mahalanobis_percentile(d_sq, 50)
            assert 0.0 <= pct <= 100.0


# ---------------------------------------------------------------------------
# compute_mahalanobis_ood  (end-to-end, NumPy backend guaranteed)
# ---------------------------------------------------------------------------


class TestComputeMahalanobisOod:
    def _make_inputs(self, n: int = 50, seed: int = 0):
        rng = np.random.default_rng(seed)
        x_pca = rng.standard_normal(n).astype(np.float32).reshape(1, n)
        ev = np.linspace(10.0, 0.1, n).astype(np.float32)
        return x_pca, ev

    def test_returns_required_keys(self):
        x_pca, ev = self._make_inputs()
        result = compute_mahalanobis_ood(x_pca, ev)
        assert "distance" in result
        assert "percentile_rank" in result
        assert "backend" in result

    def test_distance_is_non_negative(self):
        x_pca, ev = self._make_inputs()
        assert compute_mahalanobis_ood(x_pca, ev)["distance"] >= 0.0

    def test_percentile_in_0_100(self):
        x_pca, ev = self._make_inputs()
        pct = compute_mahalanobis_ood(x_pca, ev)["percentile_rank"]
        assert 0.0 <= pct <= 100.0

    def test_zero_vector_has_distance_zero(self):
        n = 50
        x_pca = np.zeros((1, n), dtype=np.float32)
        ev = np.ones(n, dtype=np.float32)
        result = compute_mahalanobis_ood(x_pca, ev)
        assert result["distance"] == pytest.approx(0.0, abs=1e-5)

    def test_backend_string_is_valid(self):
        x_pca, ev = self._make_inputs()
        backend = compute_mahalanobis_ood(x_pca, ev)["backend"]
        assert backend in ("cupy", "numpy")

    def test_accepts_flat_input(self):
        n = 50
        x_pca = np.ones(n, dtype=np.float32)
        ev = np.ones(n, dtype=np.float32)
        result = compute_mahalanobis_ood(x_pca, ev)
        # d = sqrt(n * 1^2 / 1) = sqrt(50)
        assert result["distance"] == pytest.approx(float(np.sqrt(n)), rel=1e-4)

    def test_larger_distance_for_outlier(self):
        n = 50
        ev = np.ones(n, dtype=np.float32)
        x_near = np.zeros((1, n), dtype=np.float32) + 0.1
        x_far = np.zeros((1, n), dtype=np.float32) + 10.0
        d_near = compute_mahalanobis_ood(x_near, ev)["distance"]
        d_far = compute_mahalanobis_ood(x_far, ev)["distance"]
        assert d_far > d_near


# ---------------------------------------------------------------------------
# compute_bootstrap_ci
# ---------------------------------------------------------------------------


class TestComputeBootstrapCi:
    def test_returns_required_keys(self):
        pipeline = _MockPipeline()
        x_pca = np.zeros((1, 50), dtype=np.float32)
        result = compute_bootstrap_ci(pipeline, x_pca)
        assert "lower" in result
        assert "point" in result
        assert "upper" in result

    def test_lower_le_point_le_upper(self):
        pipeline = _MockPipeline()
        x_pca = np.random.default_rng(0).standard_normal((1, 50)).astype(np.float32)
        result = compute_bootstrap_ci(pipeline, x_pca)
        assert result["lower"] <= result["point"] <= result["upper"]

    def test_point_matches_direct_prediction(self):
        pipeline = _MockPipeline()
        x_pca = np.ones((1, 50), dtype=np.float32) * 0.5
        expected_point = float(pipeline.model.predict_risk(x_pca)[0])
        result = compute_bootstrap_ci(pipeline, x_pca)
        assert result["point"] == pytest.approx(expected_point, rel=1e-5)

    def test_zero_noise_collapses_ci(self):
        pipeline = _MockPipeline()
        x_pca = np.ones((1, 50), dtype=np.float32) * 0.3
        result = compute_bootstrap_ci(pipeline, x_pca, noise_std=0.0)
        # All bootstrap samples equal the point estimate → CI collapses
        assert result["lower"] == pytest.approx(result["point"], abs=1e-5)
        assert result["upper"] == pytest.approx(result["point"], abs=1e-5)

    def test_larger_noise_widens_ci(self):
        pipeline = _MockPipeline()
        x_pca = np.zeros((1, 50), dtype=np.float32)
        r_narrow = compute_bootstrap_ci(
            pipeline, x_pca, noise_std=0.001, n_bootstrap=50
        )
        r_wide = compute_bootstrap_ci(pipeline, x_pca, noise_std=1.0, n_bootstrap=50)
        width_narrow = r_narrow["upper"] - r_narrow["lower"]
        width_wide = r_wide["upper"] - r_wide["lower"]
        assert width_wide > width_narrow

    def test_reproducible_with_same_seed(self):
        pipeline = _MockPipeline()
        x_pca = np.ones((1, 50), dtype=np.float32)
        r1 = compute_bootstrap_ci(pipeline, x_pca, seed=99)
        r2 = compute_bootstrap_ci(pipeline, x_pca, seed=99)
        assert r1["lower"] == pytest.approx(r2["lower"])
        assert r1["upper"] == pytest.approx(r2["upper"])

    def test_different_seeds_may_differ(self):
        pipeline = _MockPipeline()
        x_pca = np.zeros((1, 50), dtype=np.float32)
        r1 = compute_bootstrap_ci(pipeline, x_pca, noise_std=0.5, seed=1)
        r2 = compute_bootstrap_ci(pipeline, x_pca, noise_std=0.5, seed=2)
        # With different seeds the samples differ (extremely likely)
        assert r1["lower"] != r2["lower"] or r1["upper"] != r2["upper"]


# ---------------------------------------------------------------------------
# compute_prediction_reliability (aggregate)
# ---------------------------------------------------------------------------


class TestComputePredictionReliability:
    def _make_inputs(self, seed: int = 0):
        pipeline = _MockPipeline(n_components=50, seed=seed)
        rng = np.random.default_rng(seed)
        x_pca = rng.standard_normal((1, 50)).astype(np.float32)
        return pipeline, x_pca

    def test_returns_all_three_components(self):
        pipeline, x_pca = self._make_inputs()
        source_map = {"clinical": {"source": "real"}, "wsi": {"source": "generated"}}
        result = compute_prediction_reliability(source_map, x_pca, pipeline)
        assert "provenance_proportion" in result
        assert "mahalanobis_ood_distance" in result
        assert "bootstrap_ci_risk_score" in result

    def test_provenance_is_float_in_unit_interval(self):
        pipeline, x_pca = self._make_inputs()
        source_map = {
            "clinical": {"source": "real"},
            "transcriptomics": {"source": "generated"},
            "wsi": {"source": "zero"},
            "methylation": {"source": "real"},
        }
        result = compute_prediction_reliability(source_map, x_pca, pipeline)
        prov = result["provenance_proportion"]
        assert isinstance(prov, float)
        assert 0.0 <= prov <= 1.0

    def test_mahalanobis_keys_present(self):
        pipeline, x_pca = self._make_inputs()
        result = compute_prediction_reliability({}, x_pca, pipeline)
        mahal = result["mahalanobis_ood_distance"]
        assert "distance" in mahal
        assert "percentile_rank" in mahal
        assert "backend" in mahal

    def test_ci_keys_present(self):
        pipeline, x_pca = self._make_inputs()
        result = compute_prediction_reliability({}, x_pca, pipeline)
        ci = result["bootstrap_ci_risk_score"]
        assert "lower" in ci
        assert "point" in ci
        assert "upper" in ci

    def test_empty_source_map_full_provenance(self):
        pipeline, x_pca = self._make_inputs()
        result = compute_prediction_reliability({}, x_pca, pipeline)
        assert result["provenance_proportion"] == pytest.approx(1.0)

    def test_all_real_full_provenance(self):
        pipeline, x_pca = self._make_inputs()
        source_map = {
            mod: {"source": "real"}
            for mod in ["clinical", "transcriptomics", "wsi", "methylation"]
        }
        result = compute_prediction_reliability(source_map, x_pca, pipeline)
        assert result["provenance_proportion"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _format_reliability_block (language agent helper) — crash regression
# ---------------------------------------------------------------------------


class TestFormatReliabilityBlock:
    """Guard against ValueError when reliability sub-dicts are partially populated."""

    @staticmethod
    def _call(reliability: dict) -> str:
        from src.orchestrator.agents.language import LanguageAgent

        return LanguageAgent._format_reliability_block(reliability)

    def test_empty_dict_does_not_crash(self):
        result = self._call({})
        assert "unavailable" in result

    def test_empty_mahal_sub_dict_does_not_crash(self):
        # This is the case the review flagged: mahal present but empty
        result = self._call(
            {
                "provenance_proportion": 0.75,
                "mahalanobis_ood_distance": {},
                "bootstrap_ci_risk_score": {},
            }
        )
        # Should not raise; provenance line must still appear
        assert "0.750" in result

    def test_partial_mahal_missing_distance_does_not_crash(self):
        result = self._call(
            {
                "mahalanobis_ood_distance": {"percentile_rank": 55.0},
                "bootstrap_ci_risk_score": {},
            }
        )
        # distance is None → mahal line is skipped entirely, no ValueError
        assert isinstance(result, str)

    def test_partial_ci_missing_upper_does_not_crash(self):
        result = self._call(
            {
                "bootstrap_ci_risk_score": {"lower": 0.1, "point": 0.5},
            }
        )
        assert isinstance(result, str)

    def test_full_reliability_renders_all_three_lines(self):
        rel = {
            "provenance_proportion": 0.8,
            "mahalanobis_ood_distance": {
                "distance": 7.42,
                "percentile_rank": 88.3,
                "backend": "numpy",
            },
            "bootstrap_ci_risk_score": {
                "lower": 0.21,
                "point": 0.45,
                "upper": 0.68,
            },
        }
        result = self._call(rel)
        assert "0.800" in result
        assert "7.420" in result
        assert "88.3" in result
        assert "0.2100" in result
        assert "0.6800" in result
