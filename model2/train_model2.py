from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import pandas as pd

# Load data
df = pd.read_csv("data/heart_disease.csv")

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

# Note: Random Forest doesn't need feature scaling, but we'll do it for consistency
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest model with hyperparameter tuning
print("Tuning hyperparameters using validation set...")
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Use GridSearchCV with validation set
grid_search = GridSearchCV(
    RandomForestClassifier(class_weight='balanced', random_state=42),
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
# - Feature Importance: Shows which clinical factors matter most

print("Random Forest Results:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))
print("\n", classification_report(y_test, y_pred))

# Show feature importance (this is a cool feature of Random Forest!)
print("\nTop 10 Most Important Features:")
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(10))