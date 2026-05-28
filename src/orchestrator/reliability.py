"""
prediction_reliability: post-hoc uncertainty quantification for the Predictor.

Three components:

  1. provenance_proportion      — fraction of real (non-generated) dimensions,
                                  derived from the existing source_map.
  2. mahalanobis_ood_distance   — Mahalanobis distance from the patient's
                                  PCA-50 vector to the training-pool centroid
                                  (origin in PCA space) under diagonal covariance
                                  given by the PCA eigenvalues.
                                  Implemented as a custom CuPy CUDA kernel
                                  (anchors the NSC course requirement).
                                  Falls back to NumPy when CuPy is unavailable.
  3. bootstrap_ci_risk_score    — 95 % CI via 50 bootstrap samples: Gaussian
                                  noise (std estimated from CV residuals) is
                                  added to x_pca and predict_risk is called for
                                  each sample. Returns (lower, point, upper).

None of these close a feedback loop on unobservable accuracy — they surface
uncertainty post-hoc for the clinician.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.baseline.pipeline import FittedPipeline

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CuPy availability probe
# ---------------------------------------------------------------------------

_CUPY_AVAILABLE = False
_cp = None

try:
    import cupy as cp  # type: ignore

    _CUPY_AVAILABLE = True
    _cp = cp
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Custom CUDA kernel (NSC course anchor)
#
# Computes Mahalanobis distance from x to centroid under diagonal covariance:
#   d = sqrt( sum_i (x_i - mu_i)^2 / var_i )
#
# Implementation notes:
#   - Single block, blockDim.x = 64 (next power-of-2 >= PCA-50 components).
#   - Threads with tid >= n contribute 0 to shared memory, so the reduction
#     is exact even when n is not a power of 2.
#   - Dynamic shared memory: each thread holds one float (4 bytes).
#   - Shared-memory tree reduction (halving stride, __syncthreads barrier).
# ---------------------------------------------------------------------------

_MAHALANOBIS_KERNEL_SRC = r"""
extern "C" __global__
void mahalanobis_kernel(
    const float* __restrict__ x,
    const float* __restrict__ centroid,
    const float* __restrict__ inv_var,
    float*       __restrict__ out,
    const int n)
{
    /*
     * Parallel Mahalanobis distance (diagonal covariance).
     *
     * d = sqrt( sum_{i=0}^{n-1} (x[i] - centroid[i])^2 * inv_var[i] )
     *
     * Launched as a single block of BLOCK_SIZE threads (BLOCK_SIZE >= n).
     * Threads with tid >= n write 0 into shared memory so the reduction
     * produces the correct sum.
     */
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int stride = blockDim.x;

    float acc = 0.0f;
    /* Grid-stride loop handles arbitrary n with any block size. */
    for (int i = tid; i < n; i += stride) {
        float diff = x[i] - centroid[i];
        acc += diff * diff * inv_var[i];
    }
    sdata[tid] = acc;
    __syncthreads();

    /* Binary tree reduction in shared memory. */
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        out[0] = sqrtf(sdata[0]);
}
"""

_KERNEL_BLOCK = 64  # next power-of-2 >= 50 (PCA components)

_mahalanobis_kernel = None  # lazy-compiled once


def _get_mahalanobis_kernel():
    global _mahalanobis_kernel
    if _mahalanobis_kernel is None and _CUPY_AVAILABLE:
        _mahalanobis_kernel = _cp.RawKernel(
            _MAHALANOBIS_KERNEL_SRC, "mahalanobis_kernel"
        )
    return _mahalanobis_kernel


def _run_mahalanobis_cupy(
    x_vec: np.ndarray,
    centroid: np.ndarray,
    inv_var: np.ndarray,
) -> float:
    """Execute the custom CUDA kernel for Mahalanobis distance."""
    kernel = _get_mahalanobis_kernel()
    n = np.int32(len(x_vec))

    x_gpu = _cp.asarray(x_vec.astype(np.float32))
    centroid_gpu = _cp.asarray(centroid.astype(np.float32))
    inv_var_gpu = _cp.asarray(inv_var.astype(np.float32))
    out_gpu = _cp.zeros(1, dtype=np.float32)

    shared_bytes = _KERNEL_BLOCK * 4  # one float per thread
    kernel(
        (1,),
        (_KERNEL_BLOCK,),
        (x_gpu, centroid_gpu, inv_var_gpu, out_gpu, n),
        shared_mem=shared_bytes,
    )
    return float(out_gpu[0])


def _run_mahalanobis_numpy(
    x_vec: np.ndarray,
    centroid: np.ndarray,
    inv_var: np.ndarray,
) -> float:
    """NumPy fallback — identical arithmetic, no GPU."""
    diff = x_vec - centroid
    return float(np.sqrt(np.sum(diff * diff * inv_var)))


def _mahalanobis_percentile(distance_sq: float, n_components: int) -> float:
    """
    Analytical percentile via chi-squared CDF.

    Mahalanobis^2 ~ chi-squared(n_components) under the Gaussian assumption.
    Returns the percentile rank in [0, 100].  Falls back to simulation if
    scipy is absent.
    """
    try:
        from scipy.stats import chi2  # type: ignore

        return float(chi2.cdf(distance_sq, df=n_components) * 100.0)
    except ImportError:
        rng = np.random.default_rng(42)
        samples = rng.standard_normal((10_000, n_components))
        sim_dists_sq = np.sum(samples**2, axis=1)
        return float(np.mean(sim_dists_sq < distance_sq) * 100.0)


# ---------------------------------------------------------------------------
# Component 1 — provenance proportion
# ---------------------------------------------------------------------------


def compute_provenance_proportion(source_map: dict) -> float:
    """
    Fraction of raw feature dimensions sourced from real (non-generated) data.

    Weights by modality dimension so that a generated methylation block (16166
    dims) counts more than a generated clinical block (63 dims), which matches
    the actual impact on the prediction.

    Returns 1.0 when source_map is empty (no reconstruction occurred).
    """
    if not source_map:
        return 1.0

    from src.data_loader import MODALITY_DIMS

    total_dims = 0
    real_dims = 0
    for mod, info in source_map.items():
        dim = MODALITY_DIMS.get(mod, 0)
        total_dims += dim
        if info.get("source") == "real":
            real_dims += dim

    return float(real_dims / total_dims) if total_dims > 0 else 1.0


# ---------------------------------------------------------------------------
# Component 2 — Mahalanobis OOD distance
# ---------------------------------------------------------------------------


def compute_mahalanobis_ood(
    x_pca: np.ndarray,
    explained_variance: np.ndarray,
) -> dict:
    """
    Mahalanobis distance from x_pca to the training-pool centroid.

    The centroid is the origin in PCA-50 space (PCA is fitted on zero-mean,
    StandardScaler-normalised data, so the centroid projects to ~0).
    The covariance is diagonal with entries = PCA eigenvalues
    (sklearn's ``explained_variance_`` attribute).

    Tries the custom CuPy kernel first; falls back to NumPy on any failure.

    Args:
        x_pca:             (1, n) or (n,) patient PCA vector.
        explained_variance: (n,) eigenvalues from FittedPipeline.pca.

    Returns:
        {
            "distance":       float  — Mahalanobis distance,
            "percentile_rank": float  — percentile in [0, 100] vs chi-squared(n),
            "backend":        str    — "cupy" or "numpy",
        }
    """
    x_vec = np.asarray(x_pca, dtype=np.float32).flatten()
    n = len(x_vec)
    centroid = np.zeros(n, dtype=np.float32)
    ev = np.asarray(explained_variance, dtype=np.float32).flatten()
    inv_var = (1.0 / np.maximum(ev, 1e-10)).astype(np.float32)

    backend = "numpy"
    try:
        if _CUPY_AVAILABLE:
            distance = _run_mahalanobis_cupy(x_vec, centroid, inv_var)
            backend = "cupy"
        else:
            raise RuntimeError("CuPy not installed")
    except Exception as exc:
        logger.info(
            "[Reliability] CuPy Mahalanobis unavailable (%s) — NumPy fallback.", exc
        )
        distance = _run_mahalanobis_numpy(
            x_vec.astype(np.float64),
            centroid.astype(np.float64),
            inv_var.astype(np.float64),
        )

    percentile_rank = _mahalanobis_percentile(float(distance) ** 2, n)

    return {
        "distance": float(distance),
        "percentile_rank": float(percentile_rank),
        "backend": backend,
    }


# ---------------------------------------------------------------------------
# Component 3 — bootstrap CI on risk score
# ---------------------------------------------------------------------------


def compute_bootstrap_ci(
    pipeline: "FittedPipeline",
    x_pca: np.ndarray,
    noise_std: float = 0.1,
    n_bootstrap: int = 50,
    seed: int = 42,
) -> dict:
    """
    95 % bootstrap confidence interval on the risk score.

    Adds i.i.d. Gaussian noise (std=noise_std in PCA units, approximating
    cross-validation residuals) to the patient's x_pca vector, re-calls
    predict_risk for each of n_bootstrap samples, and returns the empirical
    2.5th / 50th / 97.5th percentiles.

    Args:
        pipeline:    FittedPipeline with .model.predict_risk().
        x_pca:       (1, n) patient PCA vector (the point-estimate input).
        noise_std:   Std-dev of additive Gaussian noise in PCA units.
        n_bootstrap: Number of bootstrap samples (default 50).
        seed:        RNG seed for reproducibility.

    Returns:
        {"lower": float, "point": float, "upper": float}
    """
    rng = np.random.default_rng(seed)
    x_base = np.asarray(x_pca, dtype=np.float32).reshape(1, -1)

    point = float(np.asarray(pipeline.model.predict_risk(x_base)).flatten()[0])

    scores: list[float] = []
    for _ in range(n_bootstrap):
        noise = rng.normal(0.0, noise_std, size=x_base.shape).astype(np.float32)
        try:
            s = float(
                np.asarray(pipeline.model.predict_risk(x_base + noise)).flatten()[0]
            )
        except Exception:
            s = point
        scores.append(s)

    arr = np.array(scores, dtype=np.float32)
    return {
        "lower": float(np.percentile(arr, 2.5)),
        "point": point,
        "upper": float(np.percentile(arr, 97.5)),
    }


# ---------------------------------------------------------------------------
# Aggregate entry point
# ---------------------------------------------------------------------------


def compute_prediction_reliability(
    source_map: dict,
    x_pca: np.ndarray,
    pipeline: "FittedPipeline",
    noise_std: float = 0.1,
    n_bootstrap: int = 50,
) -> dict:
    """
    Compute all three reliability components and return them as a dict.

    Called by the Predictor node; result is stored in
    state['prediction_reliability'] and surfaced by the Language Agent.

    Returns:
        {
            "provenance_proportion":    float,
            "mahalanobis_ood_distance": {"distance": float,
                                         "percentile_rank": float,
                                         "backend": str},
            "bootstrap_ci_risk_score":  {"lower": float,
                                         "point": float,
                                         "upper": float},
        }
    """
    provenance = compute_provenance_proportion(source_map)

    ev = np.asarray(pipeline.pca.explained_variance_, dtype=np.float32)
    mahal = compute_mahalanobis_ood(x_pca, ev)

    ci = compute_bootstrap_ci(
        pipeline, x_pca, noise_std=noise_std, n_bootstrap=n_bootstrap
    )

    return {
        "provenance_proportion": provenance,
        "mahalanobis_ood_distance": mahal,
        "bootstrap_ci_risk_score": ci,
    }
