# Crop Disease Classifier

Automated identification of **15 plant disease conditions** across Pepper, Potato, and Tomato crops from field photographs.

**Pipeline:** Fine-tuned MobileNetV2 feature extractor → XGBoost classifier → LangGraph confidence-routing inference

---

## Results

| Metric | Frozen Baseline | Fine-tuned (Current) |
|---|---|---|
| Accuracy | 73.81% | **98.84%** |
| Macro F1 | 0.7125 | **0.9856** |
| Weighted F1 | 0.7241 | **0.9884** |
| Top-3 Accuracy | 93.19% | **99.97%** |

All metrics evaluated on **3,097 held-out test images** — never seen during training or tuning.

---

## Architecture

```
Field Photo (any resolution)
        │
        ▼
Resize LANCZOS → 224×224
        │
        ├──── TRAIN ──→ Augmentation (Flip · Rotate ±54° · Zoom · Brightness · Contrast · Noise)
        │                       │
        └──── VAL/TEST ─────────┤
                                ▼
                   MobileNetV2  (154 layers)
                   Layers 0–99   : frozen (ImageNet features)
                   Layers 100–153: fine-tuned on plant disease images
                                │
                                ▼
                   GlobalAveragePooling2D  →  (1, 1280)
                                │
               ┌────────────────┴───────────────────┐
               ▼ finetune.py                         ▼ train.py
      Save finetuned_extractor.keras        Load .npz feature caches
      Re-extract → .npz caches             XGBoost  (multi:softprob)
                                           Optuna 50 trials × 3-fold CV
                                           Best params auto-saved to config.py
                                                     │
                                                     ▼
                                           predict_langgraph.py
                                           validate → predict → route
                                           ≥ 60%: Diagnosis + Top-3
                                           < 60%: Ask for clearer photo
```

---

## Project Structure

```
crop_disease_classifier/
├── config.py                        # All hyperparameters and paths — single source of truth
├── requirements.txt
├── crop_disease_classifier.ipynb    # Full walkthrough notebook
│
├── scripts/
│   ├── finetune.py                  # Fine-tune MobileNetV2 + save extractor + re-extract features
│   ├── train.py                     # Load .npz → XGBoost → Optuna → evaluate
│   ├── predict_langgraph.py         # LangGraph inference on a single image
│   ├── extract_features.py          # (Optional) Extract frozen features only
│   └── generate_report.py          # Generate project_report.docx with live metrics
│
├── src/
│   ├── data_pipeline.py             # Dataset scanning, stratified split, augmentation
│   ├── feature_extractor.py         # Build frozen MobileNetV2, batch extraction, .npz cache
│   ├── classifier.py                # XGBoost training, Optuna tuning, sample weights
│   ├── evaluate.py                  # Metrics, confusion matrix, per-class F1 plots
│   ├── inference.py                 # Load pipeline, preprocess, predict
│   ├── utils.py                     # Logging, label encoding, auto-update config after Optuna
│   └── langgraph_workflow/
│       ├── __init__.py
│       ├── state.py                 # PlantDiseaseState TypedDict
│       ├── nodes.py                 # validate_image, run_prediction, final/low_confidence response
│       └── graph.py                 # Build and compile the LangGraph app
│
├── tests/                           # Unit tests (pytest)
│
└── outputs/                         # Generated automatically — not committed to git
    ├── features/                    # .npz feature caches (train/val/test)
    ├── models/                      # xgb_model.json, label_encoder.pkl, finetuned_feature_extractor.keras
    ├── plots/                       # confusion_matrix.png, per_class_f1.png, optuna_history.png
    ├── reports/                     # classification_report.txt, project_report.docx
    └── diagrams/                    # architecture.drawio/.jpg, data_flow.drawio/.jpg
```

---

## Setup

### 1. Clone and create virtual environment

```bash
git clone <repo-url>
cd crop_disease_classifier

python3.11 -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the dataset

Download **PlantVillage** from Kaggle ([emmarex/plantdisease](https://www.kaggle.com/datasets/emmarex/plantdisease)) and place it at:

```
data/raw/PlantVillage/
├── Pepper__bell___Bacterial_spot/
├── Pepper__bell___healthy/
├── Potato___Early_blight/
├── Potato___Late_blight/
├── Potato___healthy/
├── Tomato_Bacterial_spot/
...
```

---

## Running the Pipeline

Run scripts in this order. Each step caches its output so you only re-run what changes.

### Step 1 — Fine-tune MobileNetV2

Unfreezes top layers, trains on plant disease images, strips head, saves extractor and feature caches.

```bash
python scripts/finetune.py

# Options:
python scripts/finetune.py --unfreeze-from 100   # layer index to unfreeze from (default: 100)
python scripts/finetune.py --epochs 10           # max epochs (default: 10)
python scripts/finetune.py --lr 1e-4             # learning rate (default: 1e-4)
```

**Outputs:**
- `outputs/models/finetuned_feature_extractor.keras`
- `outputs/features/train_features_finetuned.npz`
- `outputs/features/val_features_finetuned.npz`
- `outputs/features/test_features_finetuned.npz`
- `outputs/models/label_encoder.pkl`

### Step 2 — Train XGBoost

Loads .npz caches → applies sample weights → trains XGBoost baseline → Optuna tuning → evaluates on test set.

```bash
python scripts/train.py

# Options:
python scripts/train.py --skip-tuning        # baseline only (fast, ~2 min)
python scripts/train.py --n-trials 50        # override number of Optuna trials
python scripts/train.py --subsample 0.1      # use 10% of cache (quick smoke-test)
```

After Optuna, **best hyperparameters are automatically written back to `config.py`** so the next run starts from the best known values.

**Outputs:**
- `outputs/models/xgb_model.json`
- `outputs/plots/confusion_matrix.png`
- `outputs/plots/per_class_f1.png`
- `outputs/plots/optuna_history.png`
- `outputs/reports/classification_report.txt`

### Step 3 — Inference (single image)

```bash
python scripts/predict_langgraph.py path/to/leaf.jpg

# Override confidence threshold:
python scripts/predict_langgraph.py path/to/leaf.jpg --threshold 0.75
```

**Example output (high confidence):**
```
Diagnosis: Tomato — Late blight  (87.3% confidence)

Top 3 predictions:
  #1  Tomato — Late blight          87.3%
  #2  Tomato — Early blight          8.1%
  #3  Tomato — Healthy               3.2%
```

**Example output (low confidence < 60%):**
```
Low confidence prediction (43.2% < 60% threshold).
Tips for a better result:
  • Photograph a single leaf filling most of the frame.
  • Use natural daylight — avoid shadows and flash glare.
  • Keep the camera steady and close (20–30 cm from the leaf).
```

### Step 4 — Generate Report (optional)

Generates `outputs/reports/project_report.docx` with live metrics loaded from the saved model.

```bash
python scripts/generate_report.py
```

---

## Configuration

All hyperparameters live in `config.py` — no magic numbers elsewhere.

| Key | Default | Description |
|-----|---------|-------------|
| `INPUT_SIZE` | `(224, 224)` | MobileNetV2 native resolution |
| `FEATURE_SOURCE` | `"finetuned"` | `"224"` = frozen backbone, `"finetuned"` = fine-tuned |
| `CONFIDENCE_THRESHOLD` | `0.60` | LangGraph routing: below this → low-confidence path |
| `OPTUNA_N_TRIALS` | `50` | Number of Optuna hyperparameter search trials |
| `OPTUNA_CV_FOLDS` | `3` | Cross-validation folds per trial |
| `USE_SMOTE` | `False` | SMOTE disabled — sample weights handle imbalance |
| `USE_PCA` | `False` | PCA disabled — XGBoost handles 1280 dims via colsample_bytree |

**XGBoost best params** (auto-updated by Optuna after each run):

```python
XGB_BASE_PARAMS = {
    "n_estimators"    : 361,
    "learning_rate"   : 0.1377,
    "max_depth"       : 3,
    "subsample"       : 0.9625,
    "colsample_bytree": 0.3280,
    "min_child_weight": 1,
    "reg_lambda"      : 9.0785,
    "reg_alpha"       : 0.0468,
}
```

---

## Key Design Decisions

| Decision | Why |
|----------|-----|
| Fine-tune layers 100–153 only | Bottom layers hold universal features (edges, colours); top layers learn plant-specific disease textures |
| `preprocess_input` baked inside extractor | Prevents double-preprocessing at inference regardless of how the model is called |
| No SMOTE | SMOTE blurs class boundaries in 1280-dim space; inverse-frequency sample weights handle the 21× imbalance correctly |
| GlobalAveragePooling2D not Flatten | 1,280 features / 305 MB RAM vs 62,720 features / 9.5 GB RAM |
| XGBoost JSON format | Version-stable — pickle ties the model to a specific XGBoost/Python version |
| Auto-update config after Optuna | Best params written back to `config.py`; each Optuna run builds on the previous best |
| LangGraph confidence routing | Uncertainty made explicit — low-confidence predictions ask for a clearer photo rather than silently returning an unreliable diagnosis |

---

## Dataset

**PlantVillage** — 20,638 RGB images, 15 disease classes

| Split | Images | Used for |
|-------|--------|----------|
| Train (70%) | 14,446 | Fine-tuning MobileNetV2 weights + training XGBoost |
| Val (15%) | 3,104 | EarlyStopping during fine-tuning; eval_set for XGBoost |
| Test (15%) | 3,097 | Final evaluation — evaluated **once only** |

Class imbalance: **21.2×** (Potato_healthy: 106 images → Tomato_TYLCV: 3,208 images)

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Classes

| # | Class | Crop |
|---|-------|------|
| 0 | Bacterial spot | Pepper |
| 1 | Healthy | Pepper |
| 2 | Early blight | Potato |
| 3 | Late blight | Potato |
| 4 | Healthy | Potato |
| 5 | Bacterial spot | Tomato |
| 6 | Early blight | Tomato |
| 7 | Late blight | Tomato |
| 8 | Leaf mold | Tomato |
| 9 | Septoria leaf spot | Tomato |
| 10 | Spider mites | Tomato |
| 11 | Target spot | Tomato |
| 12 | Yellow leaf curl virus | Tomato |
| 13 | Mosaic virus | Tomato |
| 14 | Healthy | Tomato |
