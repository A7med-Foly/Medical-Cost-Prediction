# 🏥 Medical Cost Prediction — End-to-End ML Project

Predict individual medical insurance charges from demographic and lifestyle
features using a full modular ML pipeline.

---

## 📁 Project Structure

```
medical_cost_project/
│
├── data/
│   ├── raw/               
│
├── models/                  ← Saved .pkl files for every model + best_model.pkl
│
├── logs/
│   ├── pipeline.log         ← Rotating log file
│   └── model_comparison.csv ← Metrics for all models
│
├── notebooks/  
│   └── Medical_Cost_Personal.ipynb     ← Original exploratory notebook
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py  ← Load, clean, split, scale, encode
│   ├── model_training.py      ← Train, save, load models
│   ├── model_evaluation.py    ← Compute & compare metrics
│   ├── predict.py             ← Inference helpers
│   └── logger.py              ← Logging setup
│
├── config.py          ← All paths, features, hyperparameters
├── train.py           ← Main training entrypoint
├── predict_cli.py     ← Prediction CLI
└── requirements.txt
```

---

## ⚙️ Setup

```bash
# 1. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate 

# 2. Install dependencies
pip install -r requirements.txt

```

---

## 🚀 Training

```bash
python train.py
```

This will:
1. Load and clean `data/raw/insurance.csv`
2. Apply `MinMaxScaler` on numeric features and `OneHotEncoder` on categoricals
3. Train 6 models: Linear Regression, Polynomial Regression, Gradient Boosting,
   XGBoost, Random Forest, KNN
4. Print a comparison table of R², RMSE, MAE
5. Save the best-performing model to `models/best_model.pkl`
6. Write all metrics to `logs/model_comparison.csv`

**Sample output:**
```
══════════════════════════════════════════════════════════════════════
  MODEL COMPARISON RESULTS (sorted by R²)
══════════════════════════════════════════════════════════════════════
         Model          R2          MSE     RMSE      MAE
   XGBoost            0.8929  19671631.82  4435.61  2588.34
   GradientBoosting   0.8872  20726607.55  4552.81  2667.27
   RandomForest       0.8842  21280960.29  4613.27  2537.00
 ...
══════════════════════════════════════════════════════════════════════
```

---

## 🔮 Prediction

### Interactive mode
```bash
python predict_cli.py
```

### Single patient via flags
```bash
python predict_cli.py \
    --age 35 --sex male --bmi 28.5 \
    --children 1 --smoker no --region southwest
```

### Batch prediction from CSV
```bash
python predict_cli.py --csv path/to/new_patients.csv
```

### Programmatic use
```python
from src.predict import predict_from_paths
from config import PREPROCESSOR_PATH, BEST_MODEL_PATH

patient = {
    "age": 45, "sex": "female", "bmi": 32.0,
    "children": 2, "smoker": "yes", "region": "northeast",
}
charge = predict_from_paths(patient, PREPROCESSOR_PATH, BEST_MODEL_PATH)
print(f"Predicted charge: ${charge[0]:,.2f}")
```

---

## 🧩 Dataset

| Feature   | Type        | Description                                    |
|-----------|-------------|------------------------------------------------|
| age       | int         | Age of the primary beneficiary                 |
| sex       | categorical | `male` / `female`                              |
| bmi       | float       | Body mass index                                |
| children  | int         | Number of dependents (0–5)                     |
| smoker    | categorical | `yes` / `no`                                   |
| region    | categorical | `southwest`, `southeast`, `northwest`, `northeast` |
| **charges** | float (target) | Medical costs billed by the insurer      |

1 338 records, no missing values.

---

## 🔧 Configuration

Edit `config.py` to change:
- File paths
- Train/test split ratio
- Model hyperparameters

---

## 📈 Model Summary

| Model                | R²     | RMSE     | MAE      |
|----------------------|--------|----------|----------|
| XGBoost              | **0.8929** | 4 435.61 | 2 588.34 |
| Gradient Boosting    | 0.8872 | 4 552.65 | 2 667.27 |
| Random Forest        | 0.8842 | 4 613.13 | 2 537.00 |
| Polynomial Regression| 0.8825 | 4 646.06 | 2 867.32 |
| Linear Regression    | 0.8069 | 5 956.34 | 4 177.05 |
| KNN                  | 0.8063 | 5 966.58 | 3 744.47 |