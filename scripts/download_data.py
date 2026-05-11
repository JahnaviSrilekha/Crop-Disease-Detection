"""
Download and validate the PlantVillage dataset from Kaggle.

Prerequisites:
  1. Install the Kaggle CLI: pip install kaggle
  2. Create ~/.kaggle/kaggle.json with your API credentials:
       {"username": "YOUR_USERNAME", "key": "YOUR_API_KEY"}
  3. Set permissions: chmod 600 ~/.kaggle/kaggle.json

Usage:
  python scripts/download_data.py
"""

import sys
from pathlib import Path

# Allow imports from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.utils import get_logger

logger = get_logger("download_data")

KAGGLE_DATASET = "emmarex/plantdisease"
EXPECTED_MIN_CLASSES = 15  # emmarex/plantdisease: Pepper(2) + Potato(3) + Tomato(10)


def download_dataset() -> None:
    try:
        import kaggle
    except ImportError:
        logger.error("kaggle package not installed. Run: pip install kaggle")
        sys.exit(1)

    download_dir = config.DATA_DIR.parent
    download_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading dataset '%s' to %s ...", KAGGLE_DATASET, download_dir)
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        KAGGLE_DATASET,
        path=str(download_dir),
        unzip=True,
        quiet=False,
    )
    logger.info("Download complete.")


def validate_dataset() -> None:
    if not config.DATA_DIR.exists():
        logger.error(
            "PlantVillage directory not found at %s.\n"
            "If the dataset unzipped to a different folder, move it to:\n  %s",
            config.DATA_DIR, config.DATA_DIR,
        )
        sys.exit(1)

    class_dirs = [d for d in config.DATA_DIR.iterdir() if d.is_dir()]
    n_classes = len(class_dirs)

    if n_classes < EXPECTED_MIN_CLASSES:
        logger.error(
            "Only %d class directories found (expected >= %d). "
            "Dataset may be incomplete.", n_classes, EXPECTED_MIN_CLASSES
        )
        sys.exit(1)

    total_images = sum(
        1 for d in class_dirs
        for f in d.iterdir()
        if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    logger.info("Validation passed:")
    logger.info("  Classes found:  %d", n_classes)
    logger.info("  Total images:   %d", total_images)
    logger.info("  Data directory: %s", config.DATA_DIR)

    logger.info("Class list:")
    for d in sorted(class_dirs):
        count = sum(1 for f in d.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"})
        logger.info("  %-45s  %5d images", d.name, count)


if __name__ == "__main__":
    download_dataset()
    validate_dataset()
