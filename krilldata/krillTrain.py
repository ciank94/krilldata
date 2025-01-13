import logging
import pandas as pd
import numpy as np
import json
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
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
        'gbr': GradientBoostingRegressor
    }

    def __init__(self, inputPath, outputPath):
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
        self.modelType = None

        #====Class Methods====
        self.preprocess()
        self.training()
        return

    def preprocess(self):
        #====Preprocess====
        self.initLogger()
        self.readData()
        self.scaleFeatures()
        self.handleNan()
        self.loadXy()
        return

    def training(self):
        #====ML====
        self.trainTestSplit()
        self.trainModel(model_type='gbr', n_estimators=100, learning_rate=0.1, max_depth=3)
        self.modelMetrics()
        self.saveMetrics()
        self.saveModel()                        
        return

    #====================Preprocess methods====================
    def initLogger(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"================={self.__class__.__name__}=====================")
        self.logger.info(f"{KrillTrain.loggerDescription}")
        return

    def readData(self):
        data = pd.read_csv(self.fusedDataPath)
        self.fusedData = {"lon": data.LONGITUDE, 
                        "lat": data.LATITUDE, 
                        "depth": data.BATHYMETRY,
                        "sst": data.SST,
                        "krillDensity": data.STANDARDISED_KRILL_UNDER_1M2}
        self.logger.info(f"Read features from {self.fusedDataPath}")
        self.logger.info(f"\n{self.fusedData}")
        return

    def scaleFeatures(self):
        self.logger.info(f"Scaling features...")
        feature_columns = [key for key in self.fusedData.keys() if key != 'krillDensity']
        for key in feature_columns:
            self.fusedData[key] = (self.fusedData[key] - self.fusedData[key].mean()) / self.fusedData[key].std()
            self.logger.info(f"Feature {key} scaled, with mean {self.fusedData[key].mean()} and std {self.fusedData[key].std()}")
        self.logger.info(f"Finished scaling features")
        return

    def handleNan(self):
        # Convert dictionary of Series to DataFrame
        self.df = pd.DataFrame(self.fusedData)
        self.logger.info(f"Handling NaN...")
        self.logger.info(f"Before: {self.fusedData['krillDensity'].isna().sum()}")
        self.df = self.df.dropna()
        self.logger.info(f"Finished handling NaN")
        return

    def loadXy(self):
        self.logger.info(f"Loading Xy...") 
        self.X = self.df[['lon', 'lat', 'depth', 'sst']]
        self.y = self.df['krillDensity']
        self.logger.info(f"Finished loading Xy")
        return

    #====================ML methods====================
    def trainTestSplit(self):
        self.logger.info(f"Splitting train/test...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.logger.info(f"Finished train/test split with {len(self.X_train)} training samples and {len(self.X_test)} test samples")
        return

    def trainModel(self, model_type='rf', **kwargs):
        """Train a specified machine learning model.
            **kwargs: Additional arguments to pass to the model constructor e.g. n_estimators, max_depth
        """
        self.logger.info(f"Training {model_type} model...")
        if model_type not in KrillTrain.models:
            raise ValueError(f"Model type '{model_type}' not supported. Choose from: \
            {list(KrillTrain.models.keys())}")
        self.modelType = model_type
        # Initialize the selected model with any provided kwargs
        model_class = KrillTrain.models[model_type]
        self.model = model_class(**kwargs)
        
        # Train the model
        self.model.fit(self.X_train, self.y_train)
        self.logger.info(f"Finished {model_type} training")
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
        
        # Train on full dataset
        full_model.fit(self.X, self.y)
        
        # Save the model
        model_filename = f"{self.outputPath}/{self.modelType}Model.joblib"
        dump(full_model, model_filename)
        self.logger.info(f"Saved model to {model_filename}")
        return