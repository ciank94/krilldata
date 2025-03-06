import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

class KrillXGBoost:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.data = None
        self.presence_model = None
        self.abundance_model = None

    def load_data(self):
        self.data = pd.read_csv(f"{self.input_path}/krillFusedData.csv")
        self.data['DATE'] = pd.to_datetime(self.data['DATE'])

    def preprocess(self):
        # Filter and preprocess data
        self.data = self.data[self.data['STANDARDISED_KRILL_UNDER_1M2'] >= -2.0]
        self.data.fillna(self.data.mean(), inplace=True)

    def train_presence_model(self):
        X = self.data.drop(columns=['STANDARDISED_KRILL_UNDER_1M2', 'DATE'])
        y = (self.data['STANDARDISED_KRILL_UNDER_1M2'] != -2.0).astype(int)
        # feature scaling
        X = (X - X.mean()) / X.std()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # Set parameters for classification
        # Create DMatrix for XGBoost
        dtrain_presence = xgb.DMatrix(X_train, label=y_train)
        dtest_presence = xgb.DMatrix(X_test, label=y_test)
        params_classification = {
            'objective': 'binary:logistic',
            'learning_rate': 0.1,
            'max_depth': 5,
            'alpha': 10
        }
        self.presence_model = xgb.train(params=params_classification, dtrain=dtrain_presence, num_boost_round=100)
        preds = self.presence_model.predict(dtest_presence)
        preds = (preds > 0.5).astype(int)
        accuracy = accuracy_score(y_test, preds)
        print(f"Presence Model Accuracy: {accuracy}")

    def train_abundance_model(self):
        X = self.data[self.data['STANDARDISED_KRILL_UNDER_1M2'] != -2.0].drop(columns=['STANDARDISED_KRILL_UNDER_1M2', 'DATE'])
        y = self.data[self.data['STANDARDISED_KRILL_UNDER_1M2'] != -2.0]['STANDARDISED_KRILL_UNDER_1M2']
        # feature scaling
        X = (X - X.mean()) / X.std()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # Set parameters for regression
        params_regression = {
            'objective': 'reg:squarederror',
            'learning_rate': 0.1,
            'max_depth': 5,
            'alpha': 10
        }
        dtrain_abundance = xgb.DMatrix(X_train, label=y_train)
        dtest_abundance = xgb.DMatrix(X_test, label=y_test)
        self.abundance_model = xgb.train(params=params_regression, dtrain=dtrain_abundance, num_boost_round=100)
        preds = self.abundance_model.predict(dtest_abundance)
        mse = mean_squared_error(y_test, preds)
        print(f"Abundance Model MSE: {mse}")
        breakpoint()

    def predict_conditional_abundance(self):
        X = self.data.drop(columns=['STANDARDISED_KRILL_UNDER_1M2', 'DATE'])
        presence_preds = self.presence_model.predict_proba(X)[:, 1]
        abundance_preds = self.abundance_model.predict(X)
        conditional_abundance = presence_preds * abundance_preds
        self.data['Conditional_Abundance'] = conditional_abundance
        self.data.to_csv(f"{self.output_path}/krill_predictions.csv", index=False)
        print("Predictions saved to krill_predictions.csv")
        breakpoint()


if __name__ == "__main__":
    krill_xgb = KrillXGBoost(input_path='input', output_path='output')
    krill_xgb.load_data()
    krill_xgb.preprocess()
    krill_xgb.train_presence_model()
    krill_xgb.train_abundance_model()
    krill_xgb.predict_conditional_abundance()

# Example usage:
# krill_xgb = KrillXGBoost(input_path='input', output_path='output')
# krill_xgb.load_data()
# krill_xgb.preprocess()
# krill_xgb.train_presence_model()
# krill_xgb.train_abundance_model()
# krill_xgb.predict_conditional_abundance()