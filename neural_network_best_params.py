import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, roc_auc_score
import keras
from keras import layers, callbacks
import matplotlib.pyplot as plt
import seaborn as sns

# Load and prepare data
df = pd.read_csv('/Users/anis.larid/Desktop/Flight_Delay_Project/3_Gen_AI/ML_flight_delay_data_2024_balanced.csv')

X = df.drop(columns='Departure_Status')
y = df['Departure_Status']

status_mapping = {
    'On-Time': 0,
    'Delayed': 1
}

y = y.map(status_mapping)

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

# Create preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

# Preprocess data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

input_dim = X_train_processed.shape[1]

# Updated model architecture based on hyperparameter search
model = keras.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(56, activation='relu'),
    layers.Dropout(0.1),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam', # optimization algorithm with a default learning rate of 0.001
    loss='binary_crossentropy',
    metrics=['accuracy']
)

early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True # when training stops, the model's weights are reset to the weights of the epoch where the validation performance was best.
)

model.fit(
    X_train_processed,
    y_train,
    epochs=50,
    batch_size=32, # number of samples processed form X_train_processed and y_train before updating the weights
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

# Evaluate model
y_train_pred = (model.predict(X_train_processed) > 0.45).astype(int)
y_pred = (model.predict(X_test_processed) > 0.45).astype(int)
y_pred_proba = model.predict(X_test_processed)

# Print training metrics
print("\n========== Training Data Performance ==========")
print(f"Training Accuracy: {accuracy_score(y_train, y_train_pred)}")
print("Training Classification Report:\n", classification_report(y_train, y_train_pred, target_names=status_mapping.keys()))
print("Training Confusion Matrix:\n", confusion_matrix(y_train, y_train_pred))

# Print testing metrics
print("\n========== Testing Data Performance ==========")
print(f"Testing Accuracy: {accuracy_score(y_test, y_pred)}")
print("Testing Classification Report:\n", classification_report(y_test, y_pred, target_names=status_mapping.keys()))
print("Testing Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.style.use('dark_background')
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, cmap='Blues', cbar=False, fmt='d', xticklabels=status_mapping.keys(), yticklabels=status_mapping.keys())
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix - Feed Forward Neural Network')
plt.show()

auc_score = roc_auc_score(y_test, y_pred_proba)
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    
auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"AUC Score: {auc_score:.2f}")
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.2f})', color='#7FB7FF')
plt.plot([0, 1], [0, 1], 'k--', color='grey', label='Random Guess')  # Dashed diagonal line
plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16)
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate (Recall)', fontsize=14)
plt.legend(loc='lower right', fontsize=12)
plt.grid(alpha=0.1, linewidth=0.5)
plt.tight_layout()
plt.show()

# Get input layer weights
weights = model.layers[0].get_weights()[0]
importance = np.abs(weights).mean(axis=1)
    
# Get feature names
numeric_features_list = numerical_features
onehot_features = []
for i, feature in enumerate(categorical_features):
    categories = preprocessor.named_transformers_['cat'].categories_[i]
    onehot_features.extend([f"{feature}_{cat}" for cat in categories])
    
all_feature_names = numeric_features_list + onehot_features
    
# Create and sort importance DataFrame
feature_importance_df = pd.DataFrame({
    'Feature': all_feature_names,
    'Importance': importance
}).sort_values('Importance', ascending=False)
    
plt.figure(figsize=(15, 8))
plt.bar(range(len(importance)), feature_importance_df['Importance'], color='#7FB7FF')
plt.xticks(range(len(importance)), feature_importance_df['Feature'], rotation=90)
plt.title('Feature Importance')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.tight_layout()
plt.show()
    
print("\nFeature Importance Scores:")
print(feature_importance_df.to_string())