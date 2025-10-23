# Heart Disease Prediction Models

This project contains 3 beginner-friendly machine learning models for predicting heart disease using different algorithms.

## Models Included

### 1. **Model 1: Logistic Regression** (`model1/train_model1.py`)
- **Algorithm**: Linear classification
- **Best for**: Understanding linear relationships
- **Output**: Performance metrics + feature coefficients
- **Color theme**: Blue

### 2. **Model 2: Random Forest** (`model2/train_model2.py`)
- **Algorithm**: Ensemble of decision trees
- **Best for**: Handling non-linear patterns
- **Output**: Performance metrics + feature importance
- **Color theme**: Green

### 3. **Model 3: Decision Tree** (`model3/train_model3.py`)
- **Algorithm**: Single decision tree
- **Best for**: Understanding decision-making process
- **Output**: Performance metrics + feature importance
- **Color theme**: Orange

## How to Run

```bash
# Run individual models
python3 model1/train_model1.py
python3 model2/train_model2.py
python3 model3/train_model3.py

# Run comprehensive model comparison
python3 model_comparison.py
```

## Data Splitting

All models use a proper train/validation/test split:
- **Training set**: 75% (688 samples) - Used for model training
- **Validation set**: 15% (138 samples) - Used for hyperparameter tuning
- **Test set**: 10% (92 samples) - Used for final evaluation only

## What Each Model Shows

### 📊 **Output Included**:
1. **Performance Metrics** - Accuracy, ROC-AUC, Precision, Recall, F1-Score
2. **Feature Analysis** - Coefficients (Logistic) or Importance (Tree-based)
3. **Classification Report** - Detailed precision/recall for each class
4. **Hyperparameter Tuning** - Best parameters found via validation
5. **Model Comparison** - Comprehensive analysis in `model_comparison.py`

### 📈 **Performance Metrics**:
- **Accuracy**: Overall correctness
- **Precision**: How many predicted positives were actually positive
- **Recall**: How many actual positives were correctly identified
- **F1-Score**: Balance between precision and recall
- **ROC-AUC**: Model's ability to distinguish between classes

## Dataset

- **Target Variable**: `HeartDisease` (0 = No Heart Disease, 1 = Heart Disease)
- **Features**: Age, RestingBP, Cholesterol, FastingBS, MaxHR, Oldpeak, Sex, ChestPainType, ExerciseAngina, ST_Slope
- **Data Source**: `data/heart 2.csv`

## Requirements

```bash
pip install pandas scikit-learn matplotlib
```

## Why These Models Are Beginner-Friendly

1. **No External Dependencies**: All use standard scikit-learn libraries
2. **Concise Code**: Clean, readable code without complex visualizations
3. **Clear Output**: Easy to understand performance metrics and feature analysis
4. **Educational Comments**: Extensive explanations in the code
5. **Consistent Structure**: All models follow the same pattern for easy comparison

## Key Learning Points

- **Logistic Regression**: Shows linear relationships and feature coefficients
- **Random Forest**: Shows ensemble learning and feature importance
- **Decision Tree**: Shows actual decision rules and tree structure

Each model demonstrates different aspects of machine learning, making this perfect for learning and understanding how different algorithms work!

## Model Comparison

The `model_comparison.py` script provides simple analysis:

### 📊 **What It Shows**:
- **Performance Comparison**: Side-by-side accuracy and ROC-AUC
- **Simple Chart**: Clean bar chart comparing all models
- **Clear Rankings**: Which model performs best
- **Key Insights**: Easy-to-understand conclusions

### 🏆 **Performance Results**:
1. **Logistic Regression**: ROC-AUC = 0.969, Accuracy = 90.2%
2. **Random Forest**: ROC-AUC = 0.957, Accuracy = 91.3%
3. **Decision Tree**: ROC-AUC = 0.908, Accuracy = 83.7%

### 💡 **Key Insights**:
- All models achieve >90% ROC-AUC (excellent performance)
- Logistic Regression has best discrimination ability
- Random Forest has highest accuracy
- Decision Tree is most interpretable
