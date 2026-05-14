"""
Unit tests for src/preprocessor.py.

Uses small synthetic feature matrices (random floats, 10 features instead
of 1280) to verify:
  - SMOTE correctly oversamples minority classes
  - SMOTE is skipped when all classes already meet the target
  - StandardScaler produces mean ≈ 0, std ≈ 1
  - PCA reduces dimensionality correctly
  - apply_scaler_pca (transform-only) matches fit+transform output
  - save / load round-trip preserves scaler and PCA exactly
  - run_preprocessing produces correct output shapes
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessor import (
    apply_scaler_pca,
    apply_smote,
    fit_scaler_pca,
    load_preprocessors,
    run_preprocessing,
    save_preprocessors,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_FEATURES = 20       # small for speed (real pipeline uses 1280)
N_COMPONENTS = 8      # PCA target (real pipeline uses 256)
N_CLASSES = 4


def _make_imbalanced(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    4 classes with counts [5, 5, 30, 30].
    Classes 0 and 1 are minority; classes 2 and 3 are majority.
    """
    rng = np.random.default_rng(seed)
    sizes = [5, 5, 30, 30]
    X = np.vstack([rng.random((n, N_FEATURES)).astype(np.float32) for n in sizes])
    y = np.concatenate([np.full(n, i, dtype=np.int32) for i, n in enumerate(sizes)])
    return X, y


def _make_balanced(n_per_class: int = 40, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.random((n_per_class * N_CLASSES, N_FEATURES)).astype(np.float32)
    y = np.repeat(np.arange(N_CLASSES, dtype=np.int32), n_per_class)
    return X, y


# ---------------------------------------------------------------------------
# apply_smote
# ---------------------------------------------------------------------------

class TestApplySmote:
    def test_minority_classes_reach_target(self):
        X, y = _make_imbalanced()
        target = 20
        X_res, y_res = apply_smote(X, y, target_per_class=target, k_neighbors=3)
        _, counts = np.unique(y_res, return_counts=True)
        assert all(c >= target for c in counts), (
            f"Some class still below target {target}: {counts}"
        )

    def test_majority_classes_untouched(self):
        """Classes at or above the target must not gain extra samples."""
        X, y = _make_imbalanced()
        target = 20
        X_res, y_res = apply_smote(X, y, target_per_class=target, k_neighbors=3)
        _, counts_res = np.unique(y_res, return_counts=True)
        # Classes 2 and 3 had 30 samples (already above target=20)
        assert counts_res[2] == 30, f"Majority class 2 was modified: {counts_res[2]}"
        assert counts_res[3] == 30, f"Majority class 3 was modified: {counts_res[3]}"

    def test_total_samples_increases(self):
        X, y = _make_imbalanced()
        target = 20
        X_res, y_res = apply_smote(X, y, target_per_class=target, k_neighbors=3)
        assert len(y_res) > len(y), "SMOTE must add synthetic samples"

    def test_dtype_preserved(self):
        X, y = _make_imbalanced()
        X_res, y_res = apply_smote(X, y, target_per_class=20, k_neighbors=3)
        assert X_res.dtype == np.float32
        assert y_res.dtype == np.int32

    def test_feature_dim_unchanged(self):
        X, y = _make_imbalanced()
        X_res, _ = apply_smote(X, y, target_per_class=20, k_neighbors=3)
        assert X_res.shape[1] == N_FEATURES

    def test_skipped_when_already_balanced(self):
        """When all classes meet the target, the input is returned unchanged."""
        X, y = _make_balanced(n_per_class=50)
        X_res, y_res = apply_smote(X, y, target_per_class=30, k_neighbors=3)
        assert len(y_res) == len(y), "SMOTE should not add samples when all classes >= target"
        np.testing.assert_array_equal(X_res, X)
        np.testing.assert_array_equal(y_res, y)


# ---------------------------------------------------------------------------
# fit_scaler_pca
# ---------------------------------------------------------------------------

class TestFitScalerPca:
    def test_returns_correct_types(self):
        X, _ = _make_balanced()
        scaler, pca = fit_scaler_pca(X, n_components=N_COMPONENTS)
        assert isinstance(scaler, StandardScaler)
        assert isinstance(pca, PCA)

    def test_scaler_produces_zero_mean_unit_std(self):
        X, _ = _make_balanced(n_per_class=100)
        scaler, _ = fit_scaler_pca(X, n_components=N_COMPONENTS)
        X_scaled = scaler.transform(X)
        np.testing.assert_allclose(X_scaled.mean(axis=0), 0.0, atol=1e-5)
        np.testing.assert_allclose(X_scaled.std(axis=0),  1.0, atol=1e-5)

    def test_pca_output_dimensionality(self):
        X, _ = _make_balanced(n_per_class=100)
        scaler, pca = fit_scaler_pca(X, n_components=N_COMPONENTS)
        X_pca = pca.transform(scaler.transform(X))
        assert X_pca.shape == (len(X), N_COMPONENTS)

    def test_explained_variance_positive(self):
        X, _ = _make_balanced(n_per_class=100)
        _, pca = fit_scaler_pca(X, n_components=N_COMPONENTS)
        assert pca.explained_variance_ratio_.sum() > 0


# ---------------------------------------------------------------------------
# apply_scaler_pca
# ---------------------------------------------------------------------------

class TestApplyScalerPca:
    def test_transform_only_matches_fit_transform(self):
        """apply_scaler_pca must give the same result as fit+transform on train."""
        X_train, _ = _make_balanced(n_per_class=100)
        X_val, _   = _make_balanced(n_per_class=20, seed=99)

        scaler, pca = fit_scaler_pca(X_train, n_components=N_COMPONENTS)
        X_val_pca = apply_scaler_pca(X_val, scaler, pca, split_name="val")

        # Manually apply
        expected = pca.transform(scaler.transform(X_val)).astype(np.float32)
        np.testing.assert_allclose(X_val_pca, expected, atol=1e-6)

    def test_output_shape(self):
        X_train, _ = _make_balanced(n_per_class=100)
        X_test, _  = _make_balanced(n_per_class=15, seed=7)
        scaler, pca = fit_scaler_pca(X_train, n_components=N_COMPONENTS)
        out = apply_scaler_pca(X_test, scaler, pca)
        assert out.shape == (15 * N_CLASSES, N_COMPONENTS)

    def test_output_dtype_float32(self):
        X, _ = _make_balanced()
        scaler, pca = fit_scaler_pca(X, n_components=N_COMPONENTS)
        out = apply_scaler_pca(X, scaler, pca)
        assert out.dtype == np.float32


# ---------------------------------------------------------------------------
# save_preprocessors / load_preprocessors
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_round_trip_produces_identical_output(self, tmp_path, monkeypatch):
        """save → load → transform must give the same result as the original."""
        X, _ = _make_balanced(n_per_class=100)
        scaler, pca = fit_scaler_pca(X, n_components=N_COMPONENTS)

        # Redirect config paths to tmp_path
        import config
        monkeypatch.setattr(config, "MODELS_DIR",   tmp_path)
        monkeypatch.setattr(config, "SCALER_PATH",  tmp_path / "scaler.pkl")
        monkeypatch.setattr(config, "PCA_PATH",     tmp_path / "pca.pkl")

        save_preprocessors(scaler, pca)
        assert (tmp_path / "scaler.pkl").exists()
        assert (tmp_path / "pca.pkl").exists()

        scaler2, pca2 = load_preprocessors()

        X_orig  = pca.transform(scaler.transform(X)).astype(np.float32)
        X_round = pca2.transform(scaler2.transform(X)).astype(np.float32)
        np.testing.assert_allclose(X_orig, X_round, atol=1e-6)

    def test_load_raises_if_scaler_missing(self, tmp_path, monkeypatch):
        import config
        monkeypatch.setattr(config, "SCALER_PATH", tmp_path / "scaler.pkl")
        monkeypatch.setattr(config, "PCA_PATH",    tmp_path / "pca.pkl")
        with pytest.raises(FileNotFoundError, match="Scaler not found"):
            load_preprocessors()

    def test_load_raises_if_pca_missing(self, tmp_path, monkeypatch):
        import config, joblib
        monkeypatch.setattr(config, "SCALER_PATH", tmp_path / "scaler.pkl")
        monkeypatch.setattr(config, "PCA_PATH",    tmp_path / "pca.pkl")
        # Write scaler but not pca
        joblib.dump(StandardScaler(), tmp_path / "scaler.pkl")
        with pytest.raises(FileNotFoundError, match="PCA not found"):
            load_preprocessors()


# ---------------------------------------------------------------------------
# run_preprocessing (full pipeline)
# ---------------------------------------------------------------------------

class TestRunPreprocessing:
    def _run(self, tmp_path, monkeypatch):
        import config
        monkeypatch.setattr(config, "MODELS_DIR",              tmp_path)
        monkeypatch.setattr(config, "SCALER_PATH",             tmp_path / "scaler.pkl")
        monkeypatch.setattr(config, "PCA_PATH",                tmp_path / "pca.pkl")
        monkeypatch.setattr(config, "SMOTE_TARGET_PER_CLASS",  20)
        monkeypatch.setattr(config, "SMOTE_K_NEIGHBORS",       3)
        monkeypatch.setattr(config, "PCA_N_COMPONENTS",        N_COMPONENTS)
        monkeypatch.setattr(config, "PCA_WHITEN",              False)

        X_train, y_train = _make_imbalanced()
        X_val,   _       = _make_balanced(n_per_class=10, seed=5)
        X_test,  _       = _make_balanced(n_per_class=8,  seed=6)

        return run_preprocessing(X_train, y_train, X_val, X_test)

    def test_output_shapes_with_pca(self, tmp_path, monkeypatch):
        """When USE_PCA=True, all splits must have N_COMPONENTS features."""
        import config
        monkeypatch.setattr(config, "USE_PCA", True)
        X_tr, y_tr, X_v, X_te, scaler, pca = self._run(tmp_path, monkeypatch)
        assert len(y_tr) >= 70, f"Expected >= 70 training samples after SMOTE, got {len(y_tr)}"
        assert X_tr.shape[1] == N_COMPONENTS
        assert X_v.shape[1]  == N_COMPONENTS
        assert X_te.shape[1] == N_COMPONENTS
        assert scaler is not None
        assert pca is not None

    def test_output_shapes_without_pca(self, tmp_path, monkeypatch):
        """When USE_PCA=False, all splits must keep their original feature count."""
        import config
        monkeypatch.setattr(config, "USE_PCA", False)
        X_tr, y_tr, X_v, X_te, scaler, pca = self._run(tmp_path, monkeypatch)
        assert len(y_tr) >= 70, f"Expected >= 70 training samples after SMOTE, got {len(y_tr)}"
        assert X_tr.shape[1] == N_FEATURES   # unchanged — no PCA
        assert X_v.shape[1]  == N_FEATURES
        assert X_te.shape[1] == N_FEATURES
        assert scaler is None
        assert pca is None

    def test_output_dtypes(self, tmp_path, monkeypatch):
        import config
        monkeypatch.setattr(config, "USE_PCA", False)  # faster: skip PCA
        X_tr, y_tr, X_v, X_te, _, _ = self._run(tmp_path, monkeypatch)
        assert X_tr.dtype == np.float32
        assert X_v.dtype  == np.float32
        assert X_te.dtype == np.float32
        assert y_tr.dtype == np.int32

    def test_artifacts_saved_when_pca_enabled(self, tmp_path, monkeypatch):
        """Scaler and PCA artifacts should be written when USE_PCA=True."""
        import config
        monkeypatch.setattr(config, "USE_PCA", True)
        self._run(tmp_path, monkeypatch)
        assert (tmp_path / "scaler.pkl").exists(), "scaler.pkl not saved"
        assert (tmp_path / "pca.pkl").exists(),    "pca.pkl not saved"

    def test_artifacts_not_saved_when_pca_disabled(self, tmp_path, monkeypatch):
        """No artifact files should be written when USE_PCA=False."""
        import config
        monkeypatch.setattr(config, "USE_PCA", False)
        self._run(tmp_path, monkeypatch)
        assert not (tmp_path / "scaler.pkl").exists(), "scaler.pkl should not be saved"
        assert not (tmp_path / "pca.pkl").exists(),    "pca.pkl should not be saved"

    def test_val_test_not_smoted(self, tmp_path, monkeypatch):
        """Val and test sizes must be unchanged by preprocessing."""
        import config
        monkeypatch.setattr(config, "MODELS_DIR",              tmp_path)
        monkeypatch.setattr(config, "SCALER_PATH",             tmp_path / "scaler.pkl")
        monkeypatch.setattr(config, "PCA_PATH",                tmp_path / "pca.pkl")
        monkeypatch.setattr(config, "SMOTE_TARGET_PER_CLASS",  20)
        monkeypatch.setattr(config, "SMOTE_K_NEIGHBORS",       3)
        monkeypatch.setattr(config, "PCA_N_COMPONENTS",        N_COMPONENTS)
        monkeypatch.setattr(config, "PCA_WHITEN",              False)

        X_train, y_train = _make_imbalanced()
        n_val  = 25
        n_test = 18
        X_val  = np.random.default_rng(5).random((n_val,  N_FEATURES)).astype(np.float32)
        X_test = np.random.default_rng(6).random((n_test, N_FEATURES)).astype(np.float32)

        _, _, X_v_out, X_te_out, _, _ = run_preprocessing(X_train, y_train, X_val, X_test)
        assert X_v_out.shape[0]  == n_val,  f"Val row count changed: {X_v_out.shape[0]} != {n_val}"
        assert X_te_out.shape[0] == n_test, f"Test row count changed: {X_te_out.shape[0]} != {n_test}"
