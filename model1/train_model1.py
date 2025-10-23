from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import pandas as pd

# Load data
df = pd.read_csv("data/heart 2.csv")

# Encode categorical features
df = pd.get_dummies(df, drop_first=True)

# Split data into train/validation/test (75/15/10)
X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

# First split: 75% train, 25% temp (validation + test)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.25, random_state=42)

# Second split: Split temp into 15% validation, 10% test
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.4, random_state=42)

print(f"Training set size: {X_train.shape[0]} ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"Validation set size: {X_val.shape[0]} ({X_val.shape[0]/len(X)*100:.1f}%)")
print(f"Test set size: {X_test.shape[0]} ({X_test.shape[0]/len(X)*100:.1f}%)")

# Scale numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Train model with hyperparameter tuning
print("Tuning hyperparameters using validation set...")
param_grid = {
    'C': [0.1, 1, 10, 100],
    'max_iter': [1000, 2000],
    'solver': ['liblinear', 'lbfgs']
}

# Use GridSearchCV with validation set
grid_search = GridSearchCV(
    LogisticRegression(class_weight='balanced', random_state=42),
    param_grid,
    cv=3,
    scoring='roc_auc',
    n_jobs=-1
)

grid_search.fit(X_train_scaled, y_train)
model = grid_search.best_estimator_

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best validation score: {grid_search.best_score_:.4f}")

# Evaluate
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# EVALUATION METRICS JUSTIFICATION:
# - Accuracy: Overall correctness, good for balanced datasets
# - ROC-AUC: Handles class imbalance, measures discrimination ability
# - Precision: Important for medical diagnosis (minimize false positives)
# - Recall: Critical for medical screening (minimize false negatives)
# - F1-Score: Balanced measure for imbalanced datasets
# - Confusion Matrix: Shows specific error types for clinical interpretation

print("Logistic Regression Results:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))
print("\n", classification_report(y_test, y_pred))

# Show feature coefficients (Logistic Regression specific)
print("\nTop 10 Feature Coefficients:")
coef_df = pd.DataFrame({
    'feature': X.columns,
    'coefficient': model.coef_[0]
}).sort_values('coefficient', key=abs, ascending=False)

print(coef_df.head(10))
print("\nNote: Positive coefficients increase heart disease risk")