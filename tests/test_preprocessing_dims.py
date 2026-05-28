"""
Unit tests for cohort-specific dimension handling in preprocessing.

Root cause of LUSC MICE skip bug: MODALITY_DIMS["methylation"] = 16166
but LUSC methylation vectors are 16206-dim. _concat_features treated any size
mismatch as NaN, so _impute_mice saw 0 LUSC patients with methylation.

These tests verify the fix (detect_actual_dims + per-cohort _concat_features)
without requiring TCGA data files.
"""

import numpy as np

from src.baseline.preprocessing import (
    apply_imputation,
    build_feature_matrix,
    detect_actual_dims,
)
from src.data_loader import MODALITY_DIMS, MODALITY_KEYS

# ---------------------------------------------------------------------------
# Synthetic patient factory
# ---------------------------------------------------------------------------

_LUSC_METH_DIM = 16206  # actual LUSC methylation size (differs from MODALITY_DIMS)


def _make_patient(pid, meth_dim=MODALITY_DIMS["methylation"], seed=0):
    """Build a minimal patient dict with all modalities present."""
    rng = np.random.default_rng(seed)
    return {
        "patient_id": pid,
        "clinical": rng.standard_normal(MODALITY_DIMS["clinical"]).astype(np.float32),
        "transcriptomics": rng.standard_normal(MODALITY_DIMS["transcriptomics"]).astype(
            np.float32
        ),
        "wsi": rng.standard_normal(MODALITY_DIMS["wsi"]).astype(np.float32),
        "methylation": rng.standard_normal(meth_dim).astype(np.float32),
        "available_modalities": MODALITY_KEYS,
        "missing_modalities": [],
        "label": {"DSS": 1.0, "DSS.time": 500.0},
    }


def _make_partial_patient(pid, meth_dim=MODALITY_DIMS["methylation"], seed=0):
    """Patient missing methylation (tests NaN column handling)."""
    rng = np.random.default_rng(seed)
    return {
        "patient_id": pid,
        "clinical": rng.standard_normal(MODALITY_DIMS["clinical"]).astype(np.float32),
        "transcriptomics": rng.standard_normal(MODALITY_DIMS["transcriptomics"]).astype(
            np.float32
        ),
        "wsi": rng.standard_normal(MODALITY_DIMS["wsi"]).astype(np.float32),
        "methylation": None,
        "available_modalities": ["clinical", "transcriptomics", "wsi"],
        "missing_modalities": ["methylation"],
        "label": {"DSS": 0.0, "DSS.time": 300.0},
    }


# ---------------------------------------------------------------------------
# detect_actual_dims
# ---------------------------------------------------------------------------


class TestDetectActualDims:
    def test_luad_dims_unchanged(self):
        """Standard LUAD methylation (16166) keeps global MODALITY_DIMS."""
        patients = [_make_patient(f"LUAD-{i}", seed=i) for i in range(5)]
        dims = detect_actual_dims(patients)
        assert dims["methylation"] == MODALITY_DIMS["methylation"]

    def test_lusc_methylation_detected(self):
        """LUSC methylation (16206) overrides the global constant."""
        patients = [
            _make_patient(f"LUSC-{i}", meth_dim=_LUSC_METH_DIM, seed=i)
            for i in range(5)
        ]
        dims = detect_actual_dims(patients)
        assert dims["methylation"] == _LUSC_METH_DIM

    def test_clinical_dim_always_correct(self):
        patients = [_make_patient(f"P-{i}", seed=i) for i in range(3)]
        dims = detect_actual_dims(patients)
        assert dims["clinical"] == MODALITY_DIMS["clinical"]

    def test_falls_back_to_global_when_modality_absent(self):
        """If all patients lack a modality, return the global default."""
        patients = [_make_partial_patient(f"P-{i}", seed=i) for i in range(3)]
        dims = detect_actual_dims(patients)
        assert dims["methylation"] == MODALITY_DIMS["methylation"]

    def test_returns_dict_with_all_modality_keys(self):
        patients = [_make_patient("P-0")]
        dims = detect_actual_dims(patients)
        assert set(dims.keys()) == set(MODALITY_KEYS)


# ---------------------------------------------------------------------------
# build_feature_matrix — LUSC methylation
# ---------------------------------------------------------------------------


class TestBuildFeatureMatrixLUSC:
    def _make_lusc_patients(self, n=10):
        return [
            _make_patient(f"LUSC-{i}", meth_dim=_LUSC_METH_DIM, seed=i)
            for i in range(n)
        ]

    def test_feature_matrix_shape_with_lusc_dims(self):
        patients = self._make_lusc_patients()
        actual_dims = detect_actual_dims(patients)
        X, y, _ = build_feature_matrix(patients, actual_dims)
        expected_total = sum(actual_dims[m] for m in MODALITY_KEYS)
        assert X.shape == (10, expected_total)

    def test_no_nan_in_methylation_block_lusc(self):
        """LUSC methylation should NOT be NaN-filled after the fix."""
        patients = self._make_lusc_patients()
        actual_dims = detect_actual_dims(patients)
        X, _, _ = build_feature_matrix(patients, actual_dims)
        meth_start = sum(
            actual_dims[m]
            for m in MODALITY_KEYS
            if m != "methylation"
            and MODALITY_KEYS.index(m) < MODALITY_KEYS.index("methylation")
        )
        # Compute methylation column slice
        offset = 0
        for mod in MODALITY_KEYS:
            if mod == "methylation":
                meth_start = offset
                break
            offset += actual_dims[mod]
        meth_block = X[:, meth_start : meth_start + actual_dims["methylation"]]
        assert not np.isnan(meth_block).any(), "LUSC methylation should not be NaN"

    def test_lusc_meth_nan_without_fix(self):
        """Regression: the old code (using global dims) would fill LUSC meth with NaN."""
        patients = self._make_lusc_patients()
        # Simulate old behaviour: use global MODALITY_DIMS for feature construction
        # (LUSC meth=16206 != 16166 → old code fills NaN)
        from src.baseline.preprocessing import _concat_features

        row = _concat_features(patients[0], MODALITY_DIMS)  # wrong dims
        offset = sum(
            MODALITY_DIMS[m]
            for m in MODALITY_KEYS
            if MODALITY_KEYS.index(m) < MODALITY_KEYS.index("methylation")
        )
        meth_block = row[offset : offset + MODALITY_DIMS["methylation"]]
        assert np.isnan(meth_block).all(), (
            "Old code should produce all-NaN methylation block"
        )

    def test_val_test_use_train_dims(self):
        """Val/test matrices built with train dims keep the same column count."""
        train = self._make_lusc_patients(8)
        val = self._make_lusc_patients(2)
        actual_dims = detect_actual_dims(train)
        X_train, _, _ = build_feature_matrix(train, actual_dims)
        X_val, _, _ = build_feature_matrix(val, actual_dims)
        assert X_train.shape[1] == X_val.shape[1]


# ---------------------------------------------------------------------------
# _impute_mice — no longer skips LUSC methylation
# ---------------------------------------------------------------------------


class TestImputeMiceLUSC:
    """Verify that _impute_mice handles LUSC methylation without skipping."""

    def _lusc_patients(self, n):
        return [
            _make_patient(f"LUSC-{i}", meth_dim=_LUSC_METH_DIM, seed=i)
            for i in range(n)
        ]

    def test_mice_does_not_skip_lusc_methylation(self, capsys):
        """After the fix, methylation should NOT appear in the 'skipped' log."""
        patients = self._lusc_patients(20)
        actual_dims = detect_actual_dims(patients)
        X, _, _ = build_feature_matrix(patients, actual_dims)
        # Split into train/val/test
        X_train, X_val, X_test = X[:12], X[12:16], X[16:]

        apply_imputation("mice", X_train, X_val, X_test, actual_dims=actual_dims)

        captured = capsys.readouterr()
        assert "methylation: skipped" not in captured.out, (
            "LUSC methylation should not be skipped by _impute_mice after the fix"
        )

    def test_mice_output_shape_lusc(self):
        """MICE output must be consistent (no crashed split due to skipped modality)."""
        patients = self._lusc_patients(20)
        actual_dims = detect_actual_dims(patients)
        X, _, _ = build_feature_matrix(patients, actual_dims)
        X_train, X_val, X_test = X[:12], X[12:16], X[16:]

        (X_tr, X_va, X_te), _ = apply_imputation(
            "mice", X_train, X_val, X_test, actual_dims=actual_dims
        )
        assert X_tr.shape[0] == 12
        assert X_va.shape[0] == 4
        assert X_te.shape[0] == 4
        assert X_tr.shape[1] == X_va.shape[1] == X_te.shape[1]
        assert not np.isnan(X_tr).any()
        assert not np.isnan(X_va).any()
        assert not np.isnan(X_te).any()


# ---------------------------------------------------------------------------
# apply_imputation backward compatibility
# ---------------------------------------------------------------------------


class TestApplyImputationBackwardCompat:
    """All strategies work with both old call signature and new actual_dims."""

    def _patients(self, n=15):
        return [_make_patient(f"P-{i}", seed=i) for i in range(n)]

    def _matrices(self, patients, actual_dims):
        X, _, _ = build_feature_matrix(patients, actual_dims)
        return X[:10], X[10:12], X[12:]

    def test_zero_strategy_no_dims(self):
        patients = self._patients()
        actual_dims = detect_actual_dims(patients)
        X_tr, X_va, X_te = self._matrices(patients, actual_dims)
        (out_tr, out_va, out_te), extra = apply_imputation("zero", X_tr, X_va, X_te)
        assert not np.isnan(out_tr).any()
        assert extra == {}

    def test_knn_strategy_no_dims(self):
        patients = self._patients()
        actual_dims = detect_actual_dims(patients)
        X_tr, X_va, X_te = self._matrices(patients, actual_dims)
        (out_tr, out_va, out_te), extra = apply_imputation("knn", X_tr, X_va, X_te)
        assert not np.isnan(out_tr).any()

    def test_mice_strategy_with_dims(self):
        patients = self._patients()
        actual_dims = detect_actual_dims(patients)
        X_tr, X_va, X_te = self._matrices(patients, actual_dims)
        (out_tr, _, _), extra = apply_imputation(
            "mice", X_tr, X_va, X_te, actual_dims=actual_dims
        )
        assert not np.isnan(out_tr).any()
        assert "per_modality_transforms" in extra
