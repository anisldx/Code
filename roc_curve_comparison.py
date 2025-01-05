import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
import keras
from keras import layers, callbacks
import matplotlib.pyplot as plt

# Load and prepare data
df = pd.read_csv('/Users/anis.larid/Desktop/Flight_Delay_Project/ML_flight_delay_data_2024_balanced.csv')
X = df.drop(columns='Departure_Status')
y = df['Departure_Status'].map({'On-Time': 0, 'Delayed': 1})
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define features
categorical_features = [
    'Aircraft_Type_IATA', 'Departure_Airport_IATA', 'Arrival_Airport_IATA',
    'Departure_Day_of_Week', 'Season', 'Public_Holiday'
]
numerical_features = [f for f in X_train.columns if f not in categorical_features]

# Create preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

# Preprocess data for neural network
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)
input_dim = X_train_processed.shape[1]

# Neural Network Model
nn_model = keras.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(56, activation='relu'),
    layers.Dropout(0.1),
    layers.Dense(1, activation='sigmoid')
])

nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True
)

nn_model.fit(
    X_train_processed, y_train,
    epochs=50, batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

# XGBoost Pipeline
xgb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(
        n_estimators=800, max_depth=5, learning_rate=0.05,
        subsample=0.7, colsample_bytree=0.8, gamma=2.2,
        min_child_weight=5, reg_alpha=0.8, reg_lambda=1.5,
        random_state=42, eval_metric='logloss'
    ))
])

# Logistic Regression Pipeline
lr_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(
        C=1, penalty='l1', solver='liblinear',
        random_state=42, max_iter=1000
    ))
])

# SVM Pipeline
svc_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', SVC(
        C=1, kernel='rbf', gamma='scale',
        probability=True, random_state=42
        ))
])

# Train XGBoost, Logistic Regression, and Support Vector Machine
xgb_pipeline.fit(X_train, y_train)
lr_pipeline.fit(X_train, y_train)
svc_pipeline.fit(X_train, y_train)

# Get predictions and ROC curves
y_pred_proba_nn = nn_model.predict(X_test_processed)
y_pred_proba_xgb = xgb_pipeline.predict_proba(X_test)[:, 1]
y_pred_proba_lr = lr_pipeline.predict_proba(X_test)[:, 1]
y_pred_proba_svc = svc_pipeline.predict_proba(X_test)[:, 1]

# Calculate ROC curves
fpr_nn, tpr_nn, _ = roc_curve(y_test, y_pred_proba_nn)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_pred_proba_xgb)
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_proba_lr)
fpr_svc, tpr_svc, _ = roc_curve(y_test, y_pred_proba_svc)

# Calculate AUC scores
auc_nn = roc_auc_score(y_test, y_pred_proba_nn)
auc_xgb = roc_auc_score(y_test, y_pred_proba_xgb)
auc_lr = roc_auc_score(y_test, y_pred_proba_lr)
auc_svc = roc_auc_score(y_test, y_pred_proba_svc)

# Create a list of model data tuples (name, fpr, tpr, auc, color)
model_data = [
    ('Neural Network', fpr_nn, tpr_nn, auc_nn, '#00FFFF'),
    ('XGBoost', fpr_xgb, tpr_xgb, auc_xgb, '#FF4300'),
    ('Logistic Regression', fpr_lr, tpr_lr, auc_lr, '#D5FC79'),
    ('Support Vector Machine', fpr_svc, tpr_svc, auc_svc, '#E06EF3')
]

# Sort the model data by AUC score in descending order
model_data.sort(key=lambda x: x[3], reverse=True)

# Plot ROC curves
plt.style.use('dark_background')
plt.figure(figsize=(10, 8))

# Plot each model's ROC curve
for name, fpr, tpr, auc, color in model_data:
    plt.plot(fpr, tpr, color=color, linewidth=1.5, 
             label=f'{name} (AUC = {auc:.2f})')

# Add random classifier line
plt.plot([0, 1], [0, 1], 'k--', color='grey', label='Random Classifier')

plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=14)
plt.ylabel('True Positive Rate (Recall)', fontsize=14)
plt.title('Receiver Operating Characteristic (ROC) Curves Comparison', fontsize=14)
plt.legend(loc='lower right', fontsize=12)
plt.grid(alpha=0.1, linewidth=0.5)
plt.tight_layout()
plt.show()
