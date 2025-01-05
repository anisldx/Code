import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, roc_auc_score
from sklearn.svm import SVC

# Load and prepare data
df = pd.read_csv('/Users/anis.larid/Desktop/Flight_Delay_Project/3_Gen_AI/ML_flight_delay_data_2024_balanced.csv')

X = df.drop(columns='Departure_Status')
y = df['Departure_Status'].map({'On-Time': 0, 'Delayed': 1})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature categories
categorical_features = [
    'Aircraft_Type_IATA',
    'Departure_Airport_IATA',
    'Arrival_Airport_IATA',
    'Departure_Day_of_Week',
    'Season',
    'Public_Holiday'
]
numerical_features = [f for f in X_train.columns if f not in categorical_features]

# Create pipeline with SVM
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

svm_clf = SVC(
    C=1,
    kernel='rbf',
    gamma='scale', # controls the influence of each data point when creating the decision boudary
    # a small gamma value gives each point a wide radius of influence, leading to a smoother, more generalized decision boundary.
    # a large gamma value gives each point a narrow radius of influence, making the decision boundary more complex and tightly fit to the training data.
    probability=True,
    random_state=42)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', svm_clf)
])

# Cross-validation
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
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
threshold = 0.4
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
