import logging
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


class KrillPredict:
    # logger
    logging.basicConfig(level=logging.INFO)
    loggerDescription = "\nKrillPredict class description:\n\
        reads fusedData from output of DataFusion class\n\
        ML preprocess: feature scaling, train/test split\n\
        ML model: random forest\n\
        ML postprocess: predict\n\
        save trained model\n\
        load trained model\n\
        predict\n"

    fusedDataFilename = "krillFusedData.csv"

    # Dictionary of available models
    models = {
        'rf': RandomForestRegressor,
        'gbr': GradientBoostingRegressor
    }

    def __init__(self, inputPath, outputPath):
        #====Instance variables====
        self.inputPath = inputPath
        self.outputPath = outputPath
        self.fusedDataPath = f"{inputPath}/{KrillPredict.fusedDataFilename}"
        self.fusedData = {}
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None

        #====Preprocess====
        self.initLogger()
        self.readData()
        self.scaleFeatures()
        self.handleNan()
        self.loadXy()

        #====ML====
        self.trainTestSplit()
        self.trainModel(model_type='gbr', n_estimators=100, learning_rate=0.1, max_depth=3)
        self.modelMetrics()
        return

    #====================Preprocess methods====================
    def initLogger(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"================={self.__class__.__name__}=====================")
        self.logger.info(f"{KrillPredict.loggerDescription}")
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
        
        if model_type not in KrillPredict.models:
            raise ValueError(f"Model type '{model_type}' not supported. Choose from: \
            {list(KrillPredict.models.keys())}")
        
        # Initialize the selected model with any provided kwargs
        model_class = KrillPredict.models[model_type]
        self.model = model_class(**kwargs)
        
        # Train the model
        self.model.fit(self.X_train, self.y_train)
        self.logger.info(f"Finished {model_type} training")
        return
    
    def modelMetrics(self):
        self.logger.info(f"Calculating metrics...")
        y_pred = self.model.predict(self.X_test)
        self.logger.info(f"R^2: {r2_score(self.y_test, y_pred)}")
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        normalised_rmse = rmse / (self.y_train.max() - self.y_train.min())
        self.logger.info(f"MSE: {mse}")
        self.logger.info(f"RMSE: {rmse}")
        self.logger.info(f"Normalised RMSE: {normalised_rmse}")
        return