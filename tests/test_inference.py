"""
Unit tests for src/inference.py.

Tests the preprocessing and formatting steps using synthetic data.
The full pipeline (load_inference_pipeline + predict_disease) requires
trained model files, so those are tested via integration fixture.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.inference import format_prediction_report, preprocess_field_photo


# ---------------------------------------------------------------------------
# preprocess_field_photo
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_image(tmp_path: Path) -> Path:
    """Create a temporary JPEG image of arbitrary size."""
    img = Image.new("RGB", (640, 480), color=(100, 150, 200))
    path = tmp_path / "field_photo.jpg"
    img.save(path)
    return path


def test_preprocess_output_shape(sample_image):
    arr = preprocess_field_photo(sample_image)
    assert arr.shape == (1, *config.INPUT_SIZE, 3), (
        f"Expected (1, {config.INPUT_SIZE[0]}, {config.INPUT_SIZE[1]}, 3), got {arr.shape}"
    )


def test_preprocess_output_dtype(sample_image):
    arr = preprocess_field_photo(sample_image)
    assert arr.dtype == np.float32


def test_preprocess_value_range(sample_image):
    """MobileNetV2 preprocess_input scales to [-1, 1]."""
    arr = preprocess_field_photo(sample_image)
    assert arr.min() >= -1.1 and arr.max() <= 1.1, (
        f"Values out of expected range: min={arr.min():.2f}, max={arr.max():.2f}"
    )


def test_preprocess_handles_rgba(tmp_path):
    """RGBA images (alpha channel) should be converted cleanly to RGB."""
    img = Image.new("RGBA", (256, 256), color=(100, 200, 50, 128))
    path = tmp_path / "rgba_image.png"
    img.save(path)
    arr = preprocess_field_photo(path)
    assert arr.shape == (1, *config.INPUT_SIZE, 3)


def test_preprocess_handles_small_image(tmp_path):
    """Very small images (e.g. 32×32) should be upscaled cleanly."""
    img = Image.new("RGB", (32, 32), color=(255, 0, 0))
    path = tmp_path / "small.jpg"
    img.save(path)
    arr = preprocess_field_photo(path)
    assert arr.shape == (1, *config.INPUT_SIZE, 3)


# ---------------------------------------------------------------------------
# format_prediction_report
# ---------------------------------------------------------------------------

def test_format_prediction_report_structure():
    predictions = [
        {"rank": 1, "raw_label": "Tomato_Late_blight",
         "crop": "Tomato", "disease": "Late blight", "confidence": 0.87},
        {"rank": 2, "raw_label": "Tomato_Early_blight",
         "crop": "Tomato", "disease": "Early blight", "confidence": 0.08},
        {"rank": 3, "raw_label": "Tomato_healthy",
         "crop": "Tomato", "disease": "Healthy", "confidence": 0.05},
    ]
    report = format_prediction_report(predictions)
    assert "#1" in report
    assert "#2" in report
    assert "#3" in report
    assert "87.0%" in report or "87" in report
    assert "Tomato" in report


def test_format_prediction_report_single():
    predictions = [
        {"rank": 1, "raw_label": "Pepper__bell___Bacterial_spot",
         "crop": "Pepper bell", "disease": "Bacterial spot", "confidence": 1.0},
    ]
    report = format_prediction_report(predictions)
    assert "#1" in report
    assert "Pepper" in report


# ---------------------------------------------------------------------------
# utils: parse_class_name
# ---------------------------------------------------------------------------

from src.utils import parse_class_name


def test_parse_single_underscore():
    crop, disease = parse_class_name("Tomato_Late_blight")
    assert crop == "Tomato"
    assert "blight" in disease.lower()


def test_parse_multi_underscore():
    # Pepper__bell___Bacterial_spot is an actual class in your dataset
    # Tests the double+triple underscore separator handling
    crop, disease = parse_class_name("Pepper__bell___Bacterial_spot")
    assert crop == "Pepper"
    assert "bacterial" in disease.lower()


def test_parse_healthy():
    crop, disease = parse_class_name("Tomato_healthy")
    assert crop == "Tomato"
    assert disease.lower() == "healthy"
