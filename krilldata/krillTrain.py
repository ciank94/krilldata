import logging
import pandas as pd
import numpy as np
import json
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from joblib import dump
import os


class KrillTrain:
    # logger
    logging.basicConfig(level=logging.INFO)
    loggerDescription = "\nKrillTrain class description:\n\
        reads fusedData from output of DataFusion class\n\
        ML preprocess: feature scaling, train/test split\n\
        ML model: random forest\n\
        ML postprocess: predict\n\
        save trained model\n\
        load trained model\n\
        predict\n"

    fusedDataFilename = "krillFusedData.csv"

    # Dictionary of available model classes
    models = {
        'rfr': RandomForestRegressor,
        'gbr': GradientBoostingRegressor,
        'dtr': DecisionTreeRegressor,
        'svm': SVR,
        'mlr': LinearRegression,
        'nnr': MLPRegressor
    }

    featureColumns = ["LONGITUDE", "LATITUDE", "BATHYMETRY", "SST"]
    targetColumn = "STANDARDISED_KRILL_UNDER_1M2"

    def __init__(self, inputPath, outputPath, modelType):
        #====Instance variables====
        self.inputPath = inputPath
        self.outputPath = outputPath
        self.fusedDataPath = f"{inputPath}/{KrillTrain.fusedDataFilename}"
        self.fusedData = {}
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.modelType = modelType
        self.model_filename = f"{self.inputPath}/{self.modelType}Model.joblib"
        self.modelParams = json.load(open("krilldata/model_params.json"))
        self.model_exists = False

        #====Class Methods====
        self.initLogger()
        self.checkModelExists()
        if not self.model_exists:
            self.preprocess()
            self.training()
        return

    def checkModelExists(self):
        """Check if a trained model already exists and load it if found."""
        if os.path.exists(self.model_filename):
            self.logger.info(f"Model already exists at {self.model_filename}")
            self.model_exists = True
        else:
            self.logger.info(f"No model found, training new model")
            self.model_exists = False
        return

    def preprocess(self):
        #====Preprocess====
        self.readData()
        self.describeData()
        self.scaleFeatures()
        self.handleNan()
        self.loadXy()
        return

    def training(self):
        #====ML====
        self.trainTestSplit()
        self.trainModelRandomSearch()
        self.modelMetrics()
        self.saveMetrics()
        self.saveModel()                        
        return

    #====================Preprocess methods====================
    def initLogger(self):
        if self.modelType not in self.modelParams:
            raise ValueError(f"Model type '{self.modelType}' not found in parameters file. Available types: {list(self.modelParams.keys())}")
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"================={self.__class__.__name__}=====================")
        self.logger.info(f"{KrillTrain.loggerDescription}")
        self.logger.info(f"Loaded parameters for {self.modelType}: {self.modelParams[self.modelType]}")
        return

    def readData(self):
        self.df = pd.read_csv(self.fusedDataPath)
        self.df = self.df[KrillTrain.featureColumns + [KrillTrain.targetColumn]]
        self.logger.info(f"Read features from {self.fusedDataPath}")
        self.logger.info(f"\n{self.df}")
        return

    def describeData(self):
        self.logger.info(f"Describing data...")
        self.logger.info(f"Dataset:\n {self.df.describe()}")
        corr_matrix = self.df.corr()
        self.logger.info(f"Dataset correlation matrix:\n {corr_matrix}")
        self.logger.info(f"Correlation with target column: {KrillTrain.targetColumn}: \n \
            {corr_matrix[KrillTrain.targetColumn].sort_values(ascending=False)}")
        return

    def scaleFeatures(self):
        self.logger.info(f"Scaling features...")
        for col in KrillTrain.featureColumns:
            self.df[col] = (self.df[col] - self.df[col].mean()) / self.df[col].std()
            self.logger.info(f"Feature {col} scaled, with mean {self.df[col].mean()} and std {self.df[col].std()}")
        self.logger.info(f"Finished scaling features")
        return

    def handleNan(self):
        self.logger.info(f"Handling NaN...")
        self.logger.info(f"Before: {self.df.isna().sum()}")
        self.df = self.df.dropna()
        self.logger.info(f"Finished handling NaN")
        return

    def loadXy(self):
        self.logger.info(f"Loading Xy...") 
        self.X = self.df[KrillTrain.featureColumns]
        self.y = self.df[KrillTrain.targetColumn]
        self.logger.info(f"Finished loading Xy")
        return

    #====================ML methods====================
    def trainTestSplit(self):
        self.logger.info(f"Splitting train/test...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.logger.info(f"Finished train/test split with {len(self.X_train)} training samples and {len(self.X_test)} test samples")
        return

    def trainModelRandomSearch(self):
        """Run random search over specified parameters for a specified model."""
        self.logger.info(f"Training {self.modelType} model...")
        if self.modelType not in KrillTrain.models:
            raise ValueError(f"Model type '{self.modelType}' not supported. Choose from: \
            {list(KrillTrain.models.keys())}")

        # Run random search
        kwargs = self.modelParams["Search"][self.modelType]
        model_class = KrillTrain.models[self.modelType]
        self.logger.info(f"Running random search for {self.modelType} model...")
        self.model = RandomizedSearchCV(model_class(), kwargs, n_iter=10, cv=5, random_state=42)
        self.model.fit(self.X_train, self.y_train)
        self.logger.info(f"Best parameters: {self.model.best_params_}")

        # Evaluate on the test set
        test_score = self.model.score(self.X_test, self.y_test)
        self.logger.info(f"Test set score: {test_score}")

        # Optional: Cross-validation on the entire dataset with the best parameters
        cv_scores = cross_val_score(self.model.best_estimator_, self.X, self.y, cv=5)
        self.logger.info(f"Cross-validation scores on full dataset: {cv_scores}")
        return
    
    def modelMetrics(self):
        """Calculate metrics for the trained model using the best estimator."""
        self.logger.info(f"Calculating metrics...")
        
        # Get predictions using the best estimator
        best_model = self.model.best_estimator_
        y_pred = best_model.predict(self.X_test)
        
        # Calculate metrics
        r2 = r2_score(self.y_test, y_pred)
        self.logger.info(f"R^2: {r2}")
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        normalised_rmse = rmse / (self.y_train.max() - self.y_train.min())
        self.logger.info(f"MSE: {mse}")
        self.logger.info(f"RMSE: {rmse}")
        self.logger.info(f"Normalised RMSE: {normalised_rmse}")
        
        # Store metrics in a dictionary
        self.metrics = {
            'model_name': self.modelType,
            'best_params': self.model.best_params_,
            'r2': r2,
            'mse': mse,
            'rmse': rmse,
            'normalised_rmse': normalised_rmse,
            'cv_results_mean': float(self.model.best_score_),
            'timestamp': '2025-01-14T14:19:20+01:00'
        }
        self.logger.info(f"Stored dictionary of metrics: {self.metrics}")
        return
        
    def saveMetrics(self):
        """Save model metrics to a JSON file."""
        if not hasattr(self, 'metrics'):
            self.logger.warning("No metrics available to save")
            return
            
        metrics_filename = f"{self.outputPath}/{self.metrics['model_name']}Metrics.json"
        with open(metrics_filename, 'w') as f:
            json.dump(self.metrics, f, indent=4)
        self.logger.info(f"Saved metrics to {metrics_filename}")
        return

    def saveModel(self):
        """Train model on full dataset and save it for later predictions.
        Uses the best estimator from random search and retrains it on the full dataset."""
        self.logger.info(f"Training final {self.modelType} model on full dataset...")
        
        # Get the best estimator from random search (already configured with best parameters)
        full_model = self.model.best_estimator_
        
        # Retrain on full dataset
        full_model.fit(self.X, self.y)
        
        # Save the model
        dump(full_model, self.model_filename)
        self.logger.info(f"Saved model to {self.model_filename}")
        self.logger.info(f"Model parameters: {full_model.get_params()}")
        return