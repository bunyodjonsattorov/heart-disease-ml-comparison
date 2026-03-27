# Heart disease classification — model comparison

Benchmark of three classical classifiers on tabular heart-health indicators: logistic regression, random forest, and decision tree. Each script trains with held-out validation for hyperparameter search and reports metrics on a final test split.

**This repository is for research and education only. It is not medical advice or a clinical decision tool.**

## Highlights

- Train/validation/test split (75% / 15% / 10%) with `random_state=42` for reproducibility
- `GridSearchCV` on the training set with ROC-AUC as the selection metric
- Standardized features before fitting (consistent with the comparison script)
- Per-model scripts plus a single comparison entry point with a simple bar chart

## Repository layout

```
├── data/
│   └── heart_disease.csv
├── model1/train_model1.py    # Logistic regression
├── model2/train_model2.py    # Random forest
├── model3/train_model3.py    # Decision tree
├── model_comparison.py       # All models + comparison plot
├── requirements.txt
└── README.md
```

## Setup

Python 3.10+ recommended.

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run from the repository root so paths to `data/` resolve correctly.

## Usage

```bash
python model1/train_model1.py
python model2/train_model2.py
python model3/train_model3.py
python model_comparison.py
```

## Data

- **Target**: `HeartDisease` (0 = no heart disease, 1 = heart disease)
- **Features** (after one-hot encoding of categoricals): age, blood pressure, cholesterol, fasting blood sugar, max heart rate, ST depression, sex, chest pain type, exercise angina, ST slope, and related engineered columns from `pandas.get_dummies(..., drop_first=True)`.

## Methodology (summary)

| Model               | Role                                      |
|---------------------|-------------------------------------------|
| Logistic regression | Linear decision boundary, interpretable coefficients |
| Random forest       | Ensemble trees, non-linear interactions  |
| Decision tree       | Single tree, rule-like structure         |

Each training script prints accuracy, ROC-AUC, precision/recall/F1, and either coefficients or feature importances where applicable.

## Example results

On the bundled dataset with the fixed split above, a typical run reports roughly:

| Model               | ROC-AUC | Accuracy |
|---------------------|---------|----------|
| Logistic regression | ~0.97   | ~90%     |
| Random forest       | ~0.96   | ~91%     |
| Decision tree       | ~0.91   | ~84%     |

Re-run the scripts locally for exact figures; small differences can appear with library version changes.
