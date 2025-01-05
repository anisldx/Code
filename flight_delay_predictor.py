import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

class FlightDelayPredictor:
    def __init__(self):
        self.pipeline = None
        self.categorical_features = [
            'Aircraft_Type_IATA',
            'Departure_Airport_IATA',
            'Arrival_Airport_IATA',
            'Departure_Day_of_Week',
            'Season',
            'Public_Holiday'
        ]
        self.status_mapping = {
            'On-Time': 0,
            'Delayed': 1
        }
        self.threshold = 0.4

    def _create_pipeline(self, X):
        numerical_features = [f for f in X.columns if f not in self.categorical_features]
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), self.categorical_features)
            ],
            remainder='passthrough'
        )

        xgb_clf = XGBClassifier(
            n_estimators=800,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.7,
            colsample_bytree=0.8,
            gamma=2.2,
            min_child_weight=5,
            reg_alpha=0.8,
            reg_lambda=1.5,
            random_state=42,
            eval_metric='logloss'
        )

        return Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', xgb_clf)
        ])

    def fit(self, X, y):
        y = pd.Series(y).map(self.status_mapping)
        self.pipeline = self._create_pipeline(X)
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        proba = self.pipeline.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int)

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)[:, 1]
    
    def save_model(self, filepath):
        joblib.dump(self, filepath)
    
    @classmethod
    def load_model(cls, filepath):
        return joblib.load(filepath)

def main():
    data = pd.read_csv('/Users/anis.larid/Desktop/Flight_Delay_Project/ML_flight_delay_data_2024_balanced.csv')
    X = data.drop(columns='Departure_Status')
    y = data['Departure_Status']
    
    model = FlightDelayPredictor()
    model.fit(X, y)
    model.save_model('flight_delay_model.joblib')

if __name__ == "__main__":
    main()
