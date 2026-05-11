# Crop Disease Detection

Classifies plant leaf diseases from field photos using a frozen MobileNetV2 feature extractor and an XGBoost classifier. Supports 15 classes across Pepper, Potato, and Tomato from the PlantVillage dataset.

---

## Project Structure

```
Crop-Disease-Detection/
├── config.py                   # All paths and hyperparameters
├── requirements.txt
├── data/
│   └── raw/
│       └── PlantVillage/       # Dataset goes here (see Step 1)
├── notebooks/
│   └── crop_disease_detection_kaggle.ipynb
├── scripts/
│   ├── download_data.py        # Download PlantVillage from Kaggle
│   ├── extract_features.py     # Run MobileNetV2 feature extraction
│   └── train.py                # Train and evaluate XGBoost
├── src/
│   ├── data_pipeline.py        # Image loading, augmentation, splits
│   ├── feature_extractor.py    # MobileNetV2 + GAP model
│   ├── classifier.py           # XGBoost training and Optuna tuning
│   ├── evaluate.py             # Metrics, plots, reports
│   ├── inference.py            # Single-image prediction
│   └── utils.py
└── tests/
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download dataset
python scripts/download_data.py

# 3. Extract features
python scripts/extract_features.py

# 4. Train (fast — no tuning)
python scripts/train.py --skip-tuning

# 5. Run inference
python -m src.inference photo.jpg
```

---

## Steps to Run

### Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

---

### Step 2 — Download the dataset

```bash
python scripts/download_data.py
```

Downloads PlantVillage into `data/raw/PlantVillage/` — 15 classes: Pepper (2), Potato (3), Tomato (10).

> Alternatively, download manually from [Kaggle: emmarex/plantdisease](https://www.kaggle.com/datasets/emmarex/plantdisease) and unzip into `data/raw/PlantVillage/`.

---

### Step 3 — Extract features

```bash
python scripts/extract_features.py
```

Runs once and caches 1280-d MobileNetV2 feature vectors to `outputs/features/*.npz`. Subsequent runs load from cache (seconds vs ~15 min).

---

### Step 4 — Train the classifier

```bash
# Skip tuning — baseline XGBoost only (faster, ~5 min)
python scripts/train.py --skip-tuning

# Full run with Optuna hyperparameter tuning (recommended, ~20 min)
python scripts/train.py

# Custom number of Optuna trials
python scripts/train.py --n-trials 10
```

Saves the trained model, plots, and classification report to `outputs/`.

---

### Step 5 — Run inference on a single image

```bash
python -m src.inference photo.jpg

# Return top 5 predictions instead of 3
python -m src.inference photo.jpg 5
```

Example output:
```
Crop Disease Classification Result
=====================================
  #1  Tomato — Late blight               87.3%
  #2  Tomato — Early blight               8.1%
  #3  Tomato — Healthy                    3.2%
```

---

## Outputs

All outputs are saved to `outputs/`:

| File | Description |
|------|-------------|
| `features/train_features_128.npz` | Cached train features |
| `features/val_features_128.npz` | Cached val features |
| `features/test_features_128.npz` | Cached test features |
| `models/xgb_model.json` | Trained XGBoost model |
| `models/label_encoder.pkl` | Label encoder |
| `plots/confusion_matrix.png` | 15×15 confusion matrix |
| `plots/per_class_f1.png` | Per-class F1 bar chart |
| `plots/optuna_history.png` | Optuna optimisation history |
| `reports/classification_report.txt` | Full sklearn classification report |

---

## Run on Kaggle

Open `notebooks/crop_disease_detection_kaggle.ipynb` on Kaggle:

1. Go to [kaggle.com/code](https://www.kaggle.com/code) → **New Notebook**
2. **File → Import Notebook** → upload `crop_disease_detection_kaggle.ipynb`
3. **Add Data** → search `emmarex/plantdisease` → Add
4. **Run All**

> Expected runtime: ~25–35 min on CPU.

---

## Data Preprocessing

| Step | Details |
|------|---------|
| Image loading | PIL with EXIF rotation support |
| Resize | 128×128 (LANCZOS) |
| Normalisation | MobileNetV2 `preprocess_input` → \[-1, 1\] |
| Augmentation | Flip, rotation ±54°, zoom ±10%, brightness/contrast ±20%, Gaussian noise |
| Split | 70% train / 15% val / 15% test (stratified) |
| Feature extraction | Frozen MobileNetV2 + GAP → 1280-d vectors |

---

## Classes

| Crop | Classes |
|------|---------|
| Pepper | Bacterial spot, Healthy |
| Potato | Early blight, Late blight, Healthy |
| Tomato | Bacterial spot, Early blight, Late blight, Leaf mold, Septoria leaf spot, Spider mites, Target spot, Yellow leaf curl virus, Mosaic virus, Healthy |
