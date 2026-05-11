"""
Unit tests for src/classifier.py.

Uses synthetic feature data (random floats) to verify:
- Baseline model trains and produces valid predictions
- Predictions are consistent after save/load
- Probability matrix has correct shape
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import xgboost as xgb

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.classifier import load_model, save_model, train_baseline


# ---------------------------------------------------------------------------
# Synthetic dataset helpers
# ---------------------------------------------------------------------------

N_CLASSES = 5       # Use a small number for fast tests; real pipeline uses 15
N_FEATURES = 1280
N_TRAIN = 200
N_VAL = 50


def _make_data(n_samples: int, n_classes: int = N_CLASSES, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.random((n_samples, N_FEATURES)).astype(np.float32)
    y = rng.integers(0, n_classes, size=n_samples).astype(np.int32)
    return X, y


# ---------------------------------------------------------------------------
# train_baseline
# ---------------------------------------------------------------------------

def test_baseline_trains_without_error():
    X_train, y_train = _make_data(N_TRAIN)
    X_val, y_val = _make_data(N_VAL, seed=1)

    params = {
        **config.XGB_BASE_PARAMS,
        "num_class": N_CLASSES,
        "n_estimators": 20,    # Keep fast for tests
    }
    model = train_baseline(X_train, y_train, X_val, y_val, params=params)
    assert isinstance(model, xgb.XGBClassifier)


def test_baseline_above_chance():
    """Accuracy on a very simple linearly-separable dataset must beat chance."""
    rng = np.random.default_rng(42)
    n = 200
    # Each class has a distinct mean in feature space — easy to classify
    X = np.vstack([rng.normal(loc=i * 5, scale=1.0, size=(n // N_CLASSES, N_FEATURES))
                   for i in range(N_CLASSES)]).astype(np.float32)
    y = np.repeat(np.arange(N_CLASSES), n // N_CLASSES).astype(np.int32)

    # Shuffle BEFORE splitting so all classes appear in both train and val.
    # A sequential split puts class 4 entirely in val → XGBoost sees labels
    # 0-3 in training, auto-detects num_class=4, then crashes on val label 4.
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]

    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    params = {**config.XGB_BASE_PARAMS, "num_class": N_CLASSES, "n_estimators": 30}
    model = train_baseline(X_train, y_train, X_val, y_val, params=params)
    preds = model.predict(X_val)
    acc = (preds == y_val).mean()
    assert acc > 1 / N_CLASSES, f"Expected accuracy > chance ({1/N_CLASSES:.2f}), got {acc:.2f}"


def test_predict_proba_shape():
    X_train, y_train = _make_data(N_TRAIN)
    X_val, y_val = _make_data(N_VAL, seed=1)
    params = {**config.XGB_BASE_PARAMS, "num_class": N_CLASSES, "n_estimators": 10}
    model = train_baseline(X_train, y_train, X_val, y_val, params=params)

    probas = model.predict_proba(X_val)
    assert probas.shape == (N_VAL, N_CLASSES), f"Expected ({N_VAL}, {N_CLASSES}), got {probas.shape}"
    np.testing.assert_allclose(probas.sum(axis=1), np.ones(N_VAL), atol=1e-5)


# ---------------------------------------------------------------------------
# save_model / load_model
# ---------------------------------------------------------------------------

def test_save_load_predictions_match(tmp_path):
    X_train, y_train = _make_data(N_TRAIN)
    X_val, y_val = _make_data(N_VAL, seed=1)
    params = {**config.XGB_BASE_PARAMS, "num_class": N_CLASSES, "n_estimators": 10}
    model = train_baseline(X_train, y_train, X_val, y_val, params=params)

    path = tmp_path / "test_model.json"
    save_model(model, path)
    loaded = load_model(path)

    preds_orig = model.predict(X_val)
    preds_loaded = loaded.predict(X_val)
    np.testing.assert_array_equal(preds_orig, preds_loaded)


def test_save_creates_json_file(tmp_path):
    X_train, y_train = _make_data(N_TRAIN)
    X_val, y_val = _make_data(N_VAL, seed=1)
    params = {**config.XGB_BASE_PARAMS, "num_class": N_CLASSES, "n_estimators": 5}
    model = train_baseline(X_train, y_train, X_val, y_val, params=params)

    path = tmp_path / "model.json"
    save_model(model, path)
    assert path.exists()
    assert path.suffix == ".json"
