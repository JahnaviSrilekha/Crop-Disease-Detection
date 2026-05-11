"""
Unit tests for src/data_pipeline.py.

These tests use synthetic data (small in-memory DataFrames and temporary
image files) — no real PlantVillage dataset required.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_pipeline import (
    build_class_index,
    image_generator,
    num_batches,
    scan_dataset,
    stratified_split,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_dataset(tmp_path: Path) -> Path:
    """Create a minimal fake PlantVillage directory with 3 classes, 10 images each."""
    classes = ["Tomato_healthy", "Tomato_Late_blight", "Potato_Late_blight"]
    for cls in classes:
        class_dir = tmp_path / cls
        class_dir.mkdir()
        for i in range(10):
            img = Image.new("RGB", (64, 64), color=(i * 20, i * 10, 50))
            img.save(class_dir / f"img_{i:03d}.jpg")
    return tmp_path


@pytest.fixture
def fake_df() -> pd.DataFrame:
    """Small DataFrame with 3 classes, 20 samples each (60 total)."""
    rows = []
    for label_int, label_str in enumerate(["ClassA", "ClassB", "ClassC"]):
        for i in range(20):
            rows.append({"filepath": f"/fake/{label_str}/img_{i}.jpg",
                         "label_str": label_str, "label_int": label_int})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# build_class_index
# ---------------------------------------------------------------------------

def test_build_class_index_sorted(fake_dataset):
    idx = build_class_index(fake_dataset)
    assert list(idx.keys()) == sorted(idx.keys()), "Classes must be sorted"
    assert len(idx) == 3
    assert set(idx.values()) == {0, 1, 2}


def test_build_class_index_empty_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_class_index(tmp_path)


# ---------------------------------------------------------------------------
# scan_dataset
# ---------------------------------------------------------------------------

def test_scan_dataset_counts(fake_dataset):
    idx = build_class_index(fake_dataset)
    df = scan_dataset(fake_dataset, idx)
    assert len(df) == 30, "3 classes × 10 images = 30 rows"
    assert df["label_int"].nunique() == 3
    assert set(df.columns) == {"filepath", "label_str", "label_int"}


# ---------------------------------------------------------------------------
# stratified_split
# ---------------------------------------------------------------------------

def test_stratified_split_sizes(fake_df):
    train, val, test = stratified_split(fake_df)
    total = len(train) + len(val) + len(test)
    assert total == len(fake_df), "No rows should be lost in the split"


def test_stratified_split_class_coverage(fake_df):
    train, val, test = stratified_split(fake_df)
    for split in (train, val, test):
        assert split["label_int"].nunique() == 3, "All splits must contain all classes"


def test_stratified_split_no_overlap(fake_df):
    train, val, test = stratified_split(fake_df)
    train_paths = set(train["filepath"])
    val_paths = set(val["filepath"])
    test_paths = set(test["filepath"])
    assert train_paths.isdisjoint(val_paths)
    assert train_paths.isdisjoint(test_paths)
    assert val_paths.isdisjoint(test_paths)


# ---------------------------------------------------------------------------
# image_generator
# ---------------------------------------------------------------------------

def test_image_generator_shapes(fake_dataset):
    idx = build_class_index(fake_dataset)
    df = scan_dataset(fake_dataset, idx)
    file_list = df["filepath"].tolist()
    labels = df["label_int"].values

    batches = list(image_generator(file_list, labels, batch_size=8))
    assert len(batches) == num_batches(30, 8)

    imgs, lbls = batches[0]
    assert imgs.ndim == 4
    assert imgs.shape[1:3] == (128, 128)  # resized to INPUT_SIZE
    assert imgs.shape[3] == 3


def test_image_generator_label_alignment(fake_dataset):
    idx = build_class_index(fake_dataset)
    df = scan_dataset(fake_dataset, idx)
    file_list = df["filepath"].tolist()
    labels = df["label_int"].values

    all_labels = []
    for _, lbls in image_generator(file_list, labels, batch_size=16):
        all_labels.extend(lbls.tolist())

    assert len(all_labels) == len(labels)
    assert all_labels == labels.tolist()


# ---------------------------------------------------------------------------
# num_batches
# ---------------------------------------------------------------------------

def test_num_batches_exact():
    assert num_batches(64, 64) == 1
    assert num_batches(65, 64) == 2
    assert num_batches(128, 64) == 2
    assert num_batches(1, 64) == 1
