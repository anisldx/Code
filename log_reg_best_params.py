import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression

# Load and prepare data
df = pd.read_csv('/Users/anis.larid/Desktop/Flight_Delay_Project/3_Gen_AI/ML_flight_delay_data_2024_balanced.csv')

X = df.drop(columns='Departure_Status')
y = df['Departure_Status'].map({'On-Time': 0, 'Delayed': 1})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define features
categorical_features = [
    'Aircraft_Type_IATA',
    'Departure_Airport_IATA',
    'Arrival_Airport_IATA',
    'Departure_Day_of_Week',
    'Season',
    'Public_Holiday'
]
numerical_features = [f for f in X_train.columns if f not in categorical_features]

# Create pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

lr_clf = LogisticRegression(
    C=1,                    # Inverse of regularization strength. It's the dial for the regularization (low C means strong regularization risk underfitting, and high C means weak regularization risk overfitting)
    penalty='l1',           # L1 regularization (LASSO) this works like a built in feature selection if C is low.
    solver='liblinear',     # The solver is the optimization algorithm used by logistic regression to minimize the loss function (i.e., find the best coefficients for the model by guiding it to converge to the global minimum)
    class_weight=None,      # No class weights
    random_state=42,
    max_iter=1000           # the maximum number of iterations the optimization algorithm (solver) is allowed to run. Essentially, it controls how long the model will try to converge to the best solution.
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', lr_clf)
])

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cross_val_scores = cross_val_score(
    pipeline,
    X_train,
    y_train,
    scoring='accuracy',
    cv=cv
)

print("\n========== Cross-Validation Results ==========")
print(f"Cross-Validation Accuracy Scores: {cross_val_scores}")
print(f"Mean Accuracy: {np.mean(cross_val_scores):.4f}")
print(f"Standard Deviation: {np.std(cross_val_scores):.4f}")

# Train and evaluate
pipeline.fit(X_train, y_train)

# Training predictions
y_train_pred_proba = pipeline.predict_proba(X_train)[:, 1]
threshold = 0.5
y_train_pred = (y_train_pred_proba >= threshold).astype(int)

print("\n========== Training Data Performance ==========")
print(f"Training Accuracy: {accuracy_score(y_train, y_train_pred)}")
print("Training Classification Report:\n", classification_report(y_train, y_train_pred, 
      target_names=['On-Time', 'Delayed']))

# Test predictions
y_test_pred_proba = pipeline.predict_proba(X_test)[:, 1]
y_test_pred = (y_test_pred_proba >= threshold).astype(int)

print("\n========== Testing Data Performance ==========")
print(f'Accuracy: {accuracy_score(y_test, y_test_pred)}')
print("Classification Report:\n", classification_report(y_test, y_test_pred, 
      target_names=['On-Time', 'Delayed']))

# Confusion Matrix
plt.style.use('dark_background')
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_test_pred)
sns.heatmap(cm, annot=True, cmap='Blues', cbar=False, fmt='d', 
            xticklabels=['On-Time', 'Delayed'], yticklabels=['On-Time', 'Delayed'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

# ROC Curve
auc_score = roc_auc_score(y_test, y_test_pred_proba)
fpr, tpr, thresholds = roc_curve(y_test, y_test_pred_proba)
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.2f})', color='#7FB7FF')
plt.plot([0, 1], [0, 1], 'k--', color='grey', label='Random Guess')
plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16)
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate (Recall)', fontsize=14)
plt.legend(loc='lower right', fontsize=12)
plt.grid(alpha=0.1, linewidth=0.5)
plt.tight_layout()
plt.show()

# Feature Importance
preprocessor = pipeline.named_steps['preprocessor']
lr_model = pipeline.named_steps['classifier']

X_transformed = preprocessor.transform(X_train)

# Get feature names
numeric_features_list = numerical_features
onehot_features = []
for i, feature in enumerate(categorical_features):
    categories = preprocessor.named_transformers_['cat'].categories_[i]
    onehot_features.extend([f"{feature}_{cat}" for cat in categories])

all_feature_names = numeric_features_list + onehot_features

# Get coefficients (feature importance for logistic regression)
importance_scores = np.abs(lr_model.coef_[0])

feature_importance_df = pd.DataFrame({
    'Feature': all_feature_names,
    'Importance': importance_scores
})

feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)

plt.figure(figsize=(15, 8))
plt.bar(range(len(importance_scores)), feature_importance_df['Importance'], color='#7FB7FF')
plt.xticks(range(len(importance_scores)), feature_importance_df['Feature'], rotation=90)
plt.title('Feature Importance Scores')
plt.xlabel('Features')
plt.ylabel('Absolute Coefficient Value')
plt.tight_layout()
plt.show()

print("\nFeature Importance Scores:")
print(feature_importance_df.to_string())