import logging
import pandas as pd
import numpy as np
import json
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from joblib import dump


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
        'rf': RandomForestRegressor,
        'gbr': GradientBoostingRegressor,
        'dtr': DecisionTreeRegressor,
        'svm': SVR
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
        self.modelParams = json.load(open("krilldata/model_params.json"))

        #====Class Methods====
        self.preprocess()
        self.training()
        return

    def preprocess(self):
        #====Preprocess====
        self.initLogger()
        self.readData()
        self.describeData()
        self.scaleFeatures()
        self.handleNan()
        self.loadXy()
        return

    def training(self):
        #====ML====
        self.trainTestSplit()
        self.trainModel()
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

    def trainModel(self, **kwargs):
        """Train a specified machine learning model.
            **kwargs: Additional arguments to pass to the model constructor. If not provided, 
            default parameters from model_params.json will be used.
        """
        self.logger.info(f"Training {self.modelType} model...")
        if self.modelType not in KrillTrain.models:
            raise ValueError(f"Model type '{self.modelType}' not supported. Choose from: \
            {list(KrillTrain.models.keys())}")
            
        # Use default parameters if none provided
        if not kwargs:
            kwargs = self.modelParams[self.modelType]
            
        # Initialize the selected model with parameters
        model_class = KrillTrain.models[self.modelType]
        self.model = model_class(**kwargs)
        
        # Train the model
        self.model.fit(self.X_train, self.y_train)
        self.logger.info(f"Finished {self.modelType} training")
        return
    
    def modelMetrics(self):
        """Calculate metrics for the trained model."""
        self.logger.info(f"Calculating metrics...")
        y_pred = self.model.predict(self.X_test)
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
            'r2': r2,
            'mse': mse,
            'rmse': rmse,
            'normalised_rmse': normalised_rmse,
            'timestamp': '2025-01-13T10:13:29+01:00'
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
        The model is trained on all available data (not just training set) 
        to maximize its predictive power for future use."""
        self.logger.info(f"Training {self.modelType} model on full dataset...")
        
        # Initialize a new model with the same parameters
        model_class = KrillTrain.models[self.modelType]
        full_model = model_class(**self.model.get_params())
        
        # fit the full dataset
        full_model.fit(self.X, self.y)
        
        # Save the model
        model_filename = f"{self.inputPath}/{self.modelType}Model.joblib"
        dump(full_model, model_filename)
        self.logger.info(f"Saved model to {model_filename}")
        return