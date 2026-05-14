"""
Farmer-facing inference pipeline.

Accepts a single field photo (any resolution, any phone format) and returns
the top-k most likely disease/crop combinations with confidence scores.

Design notes:
- The MobileNetV2 feature model is rebuilt from scratch at load time
  (Keras downloads ImageNet weights automatically; ~9 MB).  Saving the
  TF SavedModel is unnecessary and avoids TF version compatibility issues.
- PIL is used for image loading so EXIF rotation metadata is respected
  (critical for phone photos that may be portrait or landscape).
- No augmentation is applied during inference — only deterministic
  resize + MobileNetV2 preprocess_input.
"""

from pathlib import Path
from typing import Union

import numpy as np
import tensorflow as tf
from PIL import Image

import config
from src.feature_extractor import build_feature_model
from src.preprocessor import load_preprocessors
from src.utils import get_logger, load_label_encoder, parse_class_name

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Load the complete inference pipeline
# ---------------------------------------------------------------------------

def load_inference_pipeline(model_dir: Path = config.MODELS_DIR) -> dict:
    """
    Load all components required for inference.

    Returns a dict with:
        feature_model  — frozen MobileNetV2+GAP (rebuilt from Keras)
        scaler         — fitted StandardScaler (1280 features → mean 0, std 1)
        pca            — fitted PCA (1280 → 256 components)
        xgb_model      — loaded XGBClassifier (XGBoost JSON, expects 256-dim input)
        label_encoder  — fitted sklearn LabelEncoder

    The scaler and pca are loaded from disk (outputs/models/scaler.pkl and
    outputs/models/pca.pkl).  They are None if preprocessing was skipped
    during training (--skip-preprocessing flag).  In that case XGBoost
    receives the raw 1280-dim features directly.
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
            f"Label encoder not found at {le_path}. Run scripts/train.py first."
        )

    logger.info("Building feature model...")
    feature_model = build_feature_model()

    logger.info("Loading XGBoost model from %s", xgb_path)
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(str(xgb_path))

    logger.info("Loading label encoder from %s", le_path)
    label_encoder = load_label_encoder(le_path)

    # Load scaler + PCA if they exist (only created when config.USE_PCA=True)
    scaler, pca = None, None
    if config.SCALER_PATH.exists() and config.PCA_PATH.exists():
        logger.info("Loading scaler and PCA from %s", model_dir)
        scaler, pca = load_preprocessors()
        logger.info(
            "Scaler + PCA loaded — features will be reduced from 1280 → %d dims",
            pca.n_components_,
        )
    elif config.USE_PCA:
        # PCA is enabled in config but files are missing — something went wrong
        logger.warning(
            "USE_PCA=True but scaler.pkl / pca.pkl not found in %s. "
            "Re-run scripts/train.py to generate them.", model_dir
        )
    else:
        # USE_PCA=False: no scaler/PCA expected — raw 1280-dim features used
        logger.info("PCA disabled (USE_PCA=False) — using raw 1280-dim features.")

    return {
        "feature_model": feature_model,
        "scaler": scaler,
        "pca": pca,
        "xgb_model": xgb_model,
        "label_encoder": label_encoder,
    }


# ---------------------------------------------------------------------------
# Single-image preprocessing
# ---------------------------------------------------------------------------

def preprocess_field_photo(
    image_path: Union[str, Path],
    target_size: tuple[int, int] = config.INPUT_SIZE,
) -> np.ndarray:
    """
    Load and preprocess a single field photo for inference.

    Steps:
      1. PIL.Image.open — handles EXIF rotation, any format (JPEG/PNG/HEIC*)
      2. Convert to RGB — strips alpha, normalises palette images
      3. Resize to target_size with LANCZOS (high-quality downsampling)
      4. Apply MobileNetV2 preprocess_input — scales uint8 [0,255] to [-1,1]

    Returns:
        float32 array of shape (1, H, W, 3) ready for feature_model.predict
    """
    img = Image.open(str(image_path)).convert("RGB")
    img = img.resize((target_size[1], target_size[0]), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)                       # (H, W, 3)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)  # [-1, 1]
    return arr[np.newaxis, ...]                                  # (1, H, W, 3)


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
    img_batch = preprocess_field_photo(image_path)                       # (1, H, W, 3)
    features = pipeline["feature_model"].predict(img_batch, verbose=0)   # (1, 1280)

    # Apply StandardScaler + PCA if the pipeline includes them (trained with preprocessing)
    scaler = pipeline.get("scaler")
    pca    = pipeline.get("pca")
    if scaler is not None and pca is not None:
        features = scaler.transform(features)        # (1, 1280) → mean 0, std 1
        features = pca.transform(features)           # (1, 1280) → (1, 256)
        logger.debug("Feature dim after Scaler+PCA: %s", features.shape)

    probas = pipeline["xgb_model"].predict_proba(features)[0]            # (15,) — one per class

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
