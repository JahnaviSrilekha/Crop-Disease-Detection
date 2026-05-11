"""
Unit tests for src/feature_extractor.py.

Tests verify:
- Feature model architecture (0 trainable params, correct output shape)
- Feature extraction produces deterministic output with frozen weights
- .npz save/load round-trips correctly
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.feature_extractor import (
    build_feature_model,
    extract_features,
    features_cached,
    load_features,
    save_features,
)


# ---------------------------------------------------------------------------
# build_feature_model
# ---------------------------------------------------------------------------

def test_feature_model_zero_trainable_params():
    model = build_feature_model()
    import tensorflow as tf
    trainable = sum(tf.size(v).numpy() for v in model.trainable_variables)
    assert trainable == 0, f"Expected 0 trainable params, got {trainable}"


def test_feature_model_output_shape():
    model = build_feature_model()
    batch = np.random.uniform(-1, 1, (4, *config.INPUT_SIZE, 3)).astype(np.float32)
    out = model.predict(batch, verbose=0)
    assert out.shape == (4, config.FEATURE_DIM), f"Expected (4, 1280), got {out.shape}"


def test_feature_model_deterministic():
    """Same input → identical output (frozen weights, no randomness)."""
    model = build_feature_model()
    batch = np.random.uniform(-1, 1, (2, *config.INPUT_SIZE, 3)).astype(np.float32)
    out1 = model.predict(batch, verbose=0)
    out2 = model.predict(batch, verbose=0)
    np.testing.assert_array_equal(out1, out2)


def test_feature_model_output_range():
    """GAP outputs are bounded — raw values should be reasonable."""
    model = build_feature_model()
    batch = np.random.uniform(-1, 1, (4, *config.INPUT_SIZE, 3)).astype(np.float32)
    out = model.predict(batch, verbose=0)
    assert np.abs(out).max() < 100, "Feature values look unreasonably large"


# ---------------------------------------------------------------------------
# extract_features
# ---------------------------------------------------------------------------

def test_extract_features_shapes():
    model = build_feature_model()
    n_images = 10
    batch_size = 4

    def fake_generator():
        for start in range(0, n_images, batch_size):
            end = min(start + batch_size, n_images)
            b = end - start
            imgs = np.random.uniform(-1, 1, (b, *config.INPUT_SIZE, 3)).astype(np.float32)
            labels = np.zeros(b, dtype=np.int32)
            yield imgs, labels

    import math
    n_batches = math.ceil(n_images / batch_size)
    X, y = extract_features(model, fake_generator(), n_batches, desc="Test")

    assert X.shape == (n_images, config.FEATURE_DIM)
    assert y.shape == (n_images,)
    assert X.dtype == np.float32
    assert y.dtype == np.int32


# ---------------------------------------------------------------------------
# save_features / load_features / features_cached
# ---------------------------------------------------------------------------

def test_save_load_round_trip(tmp_path):
    X = np.random.rand(50, 1280).astype(np.float32)
    y = np.arange(50, dtype=np.int32)
    path = tmp_path / "test_features.npz"

    save_features(X, y, path)
    X_loaded, y_loaded = load_features(path)

    np.testing.assert_array_equal(X, X_loaded)
    np.testing.assert_array_equal(y, y_loaded)


def test_features_cached_true(tmp_path):
    X = np.zeros((10, 1280), dtype=np.float32)
    y = np.zeros(10, dtype=np.int32)
    path = tmp_path / "cache.npz"
    save_features(X, y, path)
    assert features_cached(path) is True


def test_features_cached_false(tmp_path):
    path = tmp_path / "nonexistent.npz"
    assert features_cached(path) is False
