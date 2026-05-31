"""
Farmer-facing inference pipeline.

Accepts a single field photo (any resolution, any phone format) and returns
the top-k most likely disease/crop combinations with confidence scores.

Design notes:
- When FEATURE_SOURCE == "finetuned", the saved fine-tuned extractor is loaded
  from disk. It has preprocess_input baked inside, so raw [0,255] pixels are
  passed directly — no external scaling step.
- When FEATURE_SOURCE == "224" (frozen), the frozen MobileNetV2 is rebuilt
  and preprocess_input is applied externally before calling the model.
- PIL is used for image loading so EXIF rotation metadata is respected
  (critical for phone photos that may be portrait or landscape).
- No augmentation is applied during inference — deterministic resize only.
"""

from pathlib import Path
from typing import Union

import numpy as np
import tensorflow as tf
from PIL import Image

import config
from src.feature_extractor import build_feature_model
from src.utils import get_logger, load_label_encoder, parse_class_name

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Load the complete inference pipeline
# ---------------------------------------------------------------------------

def load_inference_pipeline(model_dir: Path = config.MODELS_DIR) -> dict:
    """
    Load all components required for inference.

    Returns a dict with:
        feature_model                 — feature extractor (frozen or fine-tuned)
        model_has_internal_preprocessing — True when extractor includes preprocess_input
        xgb_model                     — loaded XGBClassifier
        label_encoder                 — fitted sklearn LabelEncoder
    """
    import xgboost as xgb

    xgb_path = model_dir / "xgb_model.json"
    le_path = model_dir / "label_encoder.pkl"

    if not xgb_path.exists():
        raise FileNotFoundError(
            f"XGBoost model not found at {xgb_path}. Run scripts/train.py first."
        )
    if not le_path.exists():
        raise FileNotFoundError(
            f"Label encoder not found at {le_path}. Run scripts/finetune.py first."
        )

    # Load the feature extractor that matches what XGBoost was trained on.
    #
    # FEATURE_SOURCE == "finetuned":
    #   Load saved fine-tuned extractor — preprocess_input is INSIDE the model,
    #   so it expects raw [0,255] pixels.
    #
    # FEATURE_SOURCE == "224" (frozen):
    #   Rebuild frozen MobileNetV2 — preprocess_input is NOT inside,
    #   so inference.py applies it externally before calling the model.
    if (
        config.FEATURE_SOURCE == "finetuned"
        and config.FINETUNED_EXTRACTOR_PATH.exists()
    ):
        logger.info(
            "Loading fine-tuned feature extractor from %s",
            config.FINETUNED_EXTRACTOR_PATH,
        )
        feature_model = tf.keras.models.load_model(
            str(config.FINETUNED_EXTRACTOR_PATH)
        )
        model_has_internal_preprocessing = True
    else:
        if config.FEATURE_SOURCE == "finetuned":
            logger.warning(
                "FEATURE_SOURCE='finetuned' but %s not found. "
                "Run scripts/finetune.py first to save the extractor.",
                config.FINETUNED_EXTRACTOR_PATH,
            )
        logger.info("Building frozen MobileNetV2 feature model...")
        feature_model = build_feature_model()
        model_has_internal_preprocessing = False

    logger.info("Loading XGBoost model from %s", xgb_path)
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(str(xgb_path))

    logger.info("Loading label encoder from %s", le_path)
    label_encoder = load_label_encoder(le_path)

    return {
        "feature_model": feature_model,
        "model_has_internal_preprocessing": model_has_internal_preprocessing,
        "xgb_model": xgb_model,
        "label_encoder": label_encoder,
    }


# ---------------------------------------------------------------------------
# Single-image preprocessing
# ---------------------------------------------------------------------------

def preprocess_field_photo(
    image_path: Union[str, Path],
    target_size: tuple[int, int] = config.INPUT_SIZE,
    apply_preprocess_input: bool = True,
) -> np.ndarray:
    """
    Load and preprocess a single field photo for inference.

    Steps:
      1. PIL.Image.open — handles EXIF rotation, any format (JPEG/PNG)
      2. Convert to RGB — strips alpha, normalises palette images
      3. Resize to target_size with LANCZOS (high-quality downsampling)
      4. Optionally apply MobileNetV2 preprocess_input ([0,255] → [-1,1]).
         Pass apply_preprocess_input=False when the feature model already
         contains preprocess_input internally (fine-tuned extractor).

    Returns:
        float32 array of shape (1, H, W, 3) ready for feature_model.predict
    """
    img = Image.open(str(image_path)).convert("RGB")
    img = img.resize((target_size[1], target_size[0]), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    if apply_preprocess_input:
        arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    return arr[np.newaxis, ...]  # (1, H, W, 3)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_disease(
    image_path: Union[str, Path],
    pipeline: dict,
    top_k: int = 3,
) -> list[dict]:
    """
    Run a single field photo through the full pipeline and return top-k
    disease predictions with confidence scores.

    Args:
        image_path: Path to the field photo (JPEG, PNG, etc.).
        pipeline:   Dict returned by load_inference_pipeline().
        top_k:      Number of top predictions to return (default 3).

    Returns:
        List of dicts (length top_k), sorted by confidence descending:
        [
            {"rank": 1, "raw_label": "Tomato_Late_blight",
             "crop": "Tomato", "disease": "Late blight", "confidence": 0.87},
            ...
        ]
    """
    # Fine-tuned extractor has preprocess_input inside → pass raw pixels.
    # Frozen extractor does not → apply preprocess_input externally.
    apply_pp = not pipeline.get("model_has_internal_preprocessing", False)
    img_batch = preprocess_field_photo(image_path, apply_preprocess_input=apply_pp)
    features = pipeline["feature_model"].predict(img_batch, verbose=0)  # (1, 1280)

    probas = pipeline["xgb_model"].predict_proba(features)[0]  # (15,)

    top_indices = np.argsort(probas)[::-1][:top_k]
    label_encoder = pipeline["label_encoder"]

    results = []
    for rank, idx in enumerate(top_indices, start=1):
        raw_label = label_encoder.inverse_transform([idx])[0]
        crop, disease = parse_class_name(raw_label)
        results.append(
            {
                "rank": rank,
                "raw_label": raw_label,
                "crop": crop,
                "disease": disease,
                "confidence": float(probas[idx]),
            }
        )
    return results


# ---------------------------------------------------------------------------
# Human-readable output
# ---------------------------------------------------------------------------

def format_prediction_report(predictions: list[dict]) -> str:
    """
    Format top-k predictions as a human-readable string.

    Example output:
        Crop Disease Classification Result
        ===================================
        #1  Tomato — Late blight            87.3%
        #2  Tomato — Early blight            8.1%
        #3  Tomato — Healthy                 3.2%
    """
    lines = [
        "Crop Disease Classification Result",
        "=" * 37,
    ]
    for p in predictions:
        label = f"{p['crop']} — {p['disease']}"
        lines.append(f"  #{p['rank']}  {label:<32}  {p['confidence']*100:5.1f}%")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.inference <path_to_image> [top_k]")
        sys.exit(1)

    image_path = sys.argv[1]
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    pipeline = load_inference_pipeline()
    predictions = predict_disease(image_path, pipeline, top_k=top_k)
    print(format_prediction_report(predictions))
