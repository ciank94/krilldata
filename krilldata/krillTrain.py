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
from sklearn.utils import resample
from joblib import dump
import os
import matplotlib.pyplot as plt
import seaborn as sns


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
    correlationSaveFig = "correlationMatrix.png"

    # Dictionary of available model classes
    models = {
        'rfr': RandomForestRegressor,
        'gbr': GradientBoostingRegressor,
        'dtr': DecisionTreeRegressor,
        'svm': SVR,
        'mlr': LinearRegression,
        'nnr': MLPRegressor
    }

    # featureColumns = ["YEAR", "LONGITUDE", "LATITUDE", "BATHYMETRY", "SST", \
    # "SSH", "UGO", "VGO", "NET_VEL", "CHL", "FE", "OXY"]
    #featureColumns = ["BATHYMETRY", "SST", "FE","SSH", "NET_VEL", "CHL", "YEAR", "LONGITUDE", "LATITUDE"]
    #featureColumns = [LONGITUDE,LATITUDE,STANDARDISED_KRILL_UNDER_1M2,DEPTH,CHL_MEAN,CHL_MIN,CHL_MAX,IRON_MEAN,IRON_MIN,IRON_MAX,SSH_MEAN,SSH_MIN,SSH_MAX,VEL_MEAN,VEL_MIN,VEL_MAX,SST_MEAN,SST_MIN,SST_MAX]
    featureColumns = ["LONGITUDE","LATITUDE","DEPTH","CHL_MEAN","CHL_MIN","CHL_MAX","IRON_MEAN","IRON_MIN","IRON_MAX","SSH_MEAN",
    "SSH_MIN","SSH_MAX","VEL_MEAN","VEL_MIN","VEL_MAX","SST_MEAN","SST_MIN","SST_MAX"]
    targetColumn = "STANDARDISED_KRILL_UNDER_1M2"

    def __init__(self, inputPath, outputPath, modelType, scenario='default'):
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
        self.modelParams = json.load(open("config/model_params.json"))
        self.model_exists = False
        self.scenario = scenario

        #====Class Methods====
        self.initLogger()
        self.checkModelExists()
        self.readData()
        self.describeData()
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
        
        self.scaleFeatures()
        self.handleNan()
        self.loadXy()
        return

    def training(self):
        """Execute the training pipeline in the correct sequence."""
        self.trainTestSplit()
        if self.scenario == "default":
            self.trainModel()
        else:
            self.trainModelRandomSearch()  # This includes model fitting
        self.modelMetrics()           # Calculate performance metrics
        self.saveMetrics()            # Save metrics to file
        self.saveModel()              # Save the final model
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
        
        # Filter out zero values (where target == -2.0)
        zero_count = (self.df[KrillTrain.targetColumn] == -2.0).sum()
        self.logger.info(f"Filtering out {zero_count} zero values (where {KrillTrain.targetColumn} == -2.0)")
        self.df = self.df[self.df[KrillTrain.targetColumn] > -2.0]
        self.logger.info(f"Remaining non-zero values: {len(self.df)}")
        
        self.logger.info(f"\n{self.df}")
        return

    def describeData(self):
        self.logger.info(f"Describing data...")
        self.logger.info(f"Dataset:\n {self.df.describe()}")
        
        # Create correlation matrix plot
        # Save correlation plot
        corr_matrix = self.df.corr()
        if os.path.exists(f"{self.outputPath}/{KrillTrain.correlationSaveFig}"):
            self.logger.info(f"File exists: {KrillTrain.correlationSaveFig}")
        else:
            self.logger.info(f"File does not exist: {KrillTrain.correlationSaveFig}")
            self.logger.info(f"File will be created: {KrillTrain.correlationSaveFig}")
            self.plotCorrelationMatrix(corr_matrix)
        return

    def scaleFeatures(self):
        self.logger.info(f"Scaling features...")
        for col in KrillTrain.featureColumns:
            self.df[col] = (self.df[col] - self.df[col].mean()) / (self.df[col].std() + 0.00001)
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

    def trainModel(self):
        self.logger.info(f"Training {self.modelType} model...")
        if self.modelType not in KrillTrain.models:
            raise ValueError(f"Model type '{self.modelType}' not supported. Choose from: \
            {list(KrillTrain.models.keys())}")
        model_class = KrillTrain.models[self.modelType]
        model_params = self.modelParams[self.modelType]
        self.model = model_class(**model_params)
        self.model.fit(self.X_train, self.y_train)
        self.logger.info(f"Model parameters: {self.model.get_params()}")
        self.logger.info(f"Model trained on {len(self.X_train)} training samples and {len(self.X_test)} test samples")
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
        # "rfr": {
        # "n_estimators": 200,
        # "max_depth": 50,
        # "min_samples_split": 2,
        # "min_samples_leaf": 1,
        # "max_features": "log2"
        #     }
        # "rfr": {
        # "n_estimators": 500,
        # "max_depth": 30,
        # "min_samples_split": 10,
        # "min_samples_leaf": 2,
        # "max_features": "log2"
        #     }
        breakpoint()
        return
    
    def getFeatureImportances(self):
        """Calculate feature importances for tree-based models."""
        self.logger.info(f"Calculating feature importances...")
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'best_estimator_') and hasattr(self.model.best_estimator_, 'feature_importances_'):
            importances = self.model.best_estimator_.feature_importances_
        else:
            self.logger.warning("Model doesn't support feature importances")
            return None
            
        # Create dictionary of feature names and their importances
        feature_importance_dict = {
            feature: float(importance) 
            for feature, importance in zip(KrillTrain.featureColumns, importances)
        }
        
        # Sort by importance
        sorted_importances = dict(sorted(feature_importance_dict.items(), 
                                       key=lambda x: x[1], 
                                       reverse=True))
        
        self.logger.info(f"Feature importances: {sorted_importances}")
        return sorted_importances

    def modelMetrics(self):
        """Calculate metrics for the trained model using the best estimator."""
        if os.path.exists(f"{self.outputPath}/{self.modelType}Metrics.json"):
            self.logger.info(f"File exists: {self.modelType}Metrics.json")
            return
        self.logger.info(f"Calculating metrics...")
        pi_stats = self.uncertaintyEstimation()
        
        # Get predictions using the best estimator
        if self.scenario == "default":
            best_model = self.model
        else:
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
       
        # Get feature importances
        feature_importances = self.getFeatureImportances()
        
        # Store metrics in a dictionary
        if self.scenario == "default":
            self.metrics = {
                'model_name': self.modelType,
                'best_params': self.model.get_params(),
                'r2': r2,
                'mse': mse,
                'rmse': rmse,
                'normalised_rmse': normalised_rmse,
                'cv_results_mean': float(self.model.score(self.X_test, self.y_test)),
                'timestamp': np.datetime64('now').astype(str),
                'PI_95_lower_mean': pi_stats['lower_mean'],
                'PI_95_lower_std': pi_stats['lower_std'],
                'PI_95_lower_min': pi_stats['lower_min'],
                'PI_95_lower_max': pi_stats['lower_max'],
                'PI_95_upper_mean': pi_stats['upper_mean'],
                'PI_95_upper_std': pi_stats['upper_std'],
                'PI_95_upper_min': pi_stats['upper_min'],
                'PI_95_upper_max': pi_stats['upper_max'],
                'PI_95_interval_width_mean': pi_stats['interval_width_mean'],
                'PI_95_interval_width_std': pi_stats['interval_width_std'],
                'feature_importances': feature_importances
            }
        else:
            self.metrics = {
                'model_name': self.modelType,
                'best_params': self.model.best_params_,
                'r2': r2,
                'mse': mse,
                'rmse': rmse,
                'normalised_rmse': normalised_rmse,
                'cv_results_mean': float(self.model.best_score_),
                'timestamp': np.datetime64('now').astype(str),
                'PI_95_lower_mean': pi_stats['lower_mean'],
                'PI_95_lower_std': pi_stats['lower_std'],
                'PI_95_lower_min': pi_stats['lower_min'],
                'PI_95_lower_max': pi_stats['lower_max'],
                'PI_95_upper_mean': pi_stats['upper_mean'],
                'PI_95_upper_std': pi_stats['upper_std'],
                'PI_95_upper_min': pi_stats['upper_min'],
                'PI_95_upper_max': pi_stats['upper_max'],
                'PI_95_interval_width_mean': pi_stats['interval_width_mean'],
                'PI_95_interval_width_std': pi_stats['interval_width_std'],
                'feature_importances': feature_importances
            }
        self.logger.info(f"Stored dictionary of metrics: {self.metrics}")
        return

    def uncertaintyEstimation(self):
        self.logger.info(f"Estimating uncertainty...")
        # Number of bootstrap samples
        n_bootstraps = 100
        predictions = np.zeros((n_bootstraps, len(self.X_test)))

        # Fit models on bootstrap samples and predict
        for i in range(n_bootstraps):
            X_resampled, y_resampled = resample(self.X_train, self.y_train)
            self.model.fit(X_resampled, y_resampled)
            predictions[i, :] = self.model.predict(self.X_test)
            self.logger.info(f"Finished bootstrapping {i+1}/{n_bootstraps}")

        # Calculate prediction intervals
        y_lower = np.percentile(predictions, 2.5, axis=0)
        y_upper = np.percentile(predictions, 97.5, axis=0)
        
        # Calculate summary statistics for the prediction intervals
        pi_stats = {
            'lower_mean': float(np.mean(y_lower)),
            'lower_std': float(np.std(y_lower)),
            'lower_min': float(np.min(y_lower)),
            'lower_max': float(np.max(y_lower)),
            'upper_mean': float(np.mean(y_upper)),
            'upper_std': float(np.std(y_upper)),
            'upper_min': float(np.min(y_upper)),
            'upper_max': float(np.max(y_upper)),
            'interval_width_mean': float(np.mean(y_upper - y_lower)),
            'interval_width_std': float(np.std(y_upper - y_lower))
        }
        
        return pi_stats

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
        if self.scenario == "default":
            best_model = self.model
        else:
            best_model = self.model.best_estimator_
        
        # Retrain on full dataset
        best_model.fit(self.X, self.y)
        
        # Save the model
        dump(best_model, self.model_filename)
        self.logger.info(f"Saved model to {self.model_filename}")
        self.logger.info(f"Model parameters: {best_model.get_params()}")
        return

    def plotCorrelationMatrix(self, corr_matrix):
        """Plot correlation matrix as a heatmap."""
        # Set the style for better visibility
        plt.style.use('default')
        
        # Create figure with adjusted size and white background
        plt.figure(figsize=(14, 12), facecolor='white')
        
        # Create a copy of the correlation matrix
        corr_matrix_plot = corr_matrix.copy()
        
        # Rename columns and index for better readability
        new_names = {
            KrillTrain.targetColumn: 'KRILL',
            'BATHYMETRY': 'DEPTH',
            'LONGITUDE': 'LON',
            'LATITUDE': 'LAT',
            'NET_VEL': 'VEL'
        }
        corr_matrix_plot = corr_matrix_plot.rename(columns=new_names, index=new_names)
        
        # Create heatmap with improved styling
        heatmap = sns.heatmap(corr_matrix_plot, 
                             annot=True,
                             cmap='RdBu_r',
                             center=0,
                             fmt='.2f',
                             vmin=-1,
                             vmax=1,
                             square=True,
                             annot_kws={'size': 14},  
                             cbar_kws={'label': 'Correlation Coefficient',
                                     'orientation': 'vertical',
                                     'pad': 0.05})  
        
        # Rotate x-axis labels
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # Adjust label sizes
        heatmap.set_xticklabels(heatmap.get_xticklabels(), size=14)  
        heatmap.set_yticklabels(heatmap.get_yticklabels(), size=14)  
        
        # Add title
        # plt.title('Feature Correlation Matrix', pad=20, size=16, weight='bold')  
        
        # Adjust colorbar
        cbar = heatmap.collections[0].colorbar
        cbar.ax.tick_params(labelsize=12)  
        cbar.set_label('Correlation Coefficient', size=14, weight='bold')  
        
        # Add more space at the bottom for x-labels
        plt.subplots_adjust(bottom=0.15)
        
        # Save figure
        plt.savefig(f"{self.outputPath}/{KrillTrain.correlationSaveFig}", 
                    dpi=300, 
                    bbox_inches='tight',
                    facecolor='white',
                    edgecolor='none')
        plt.close()
        
        self.logger.info(f"Saved correlation matrix plot to: {KrillTrain.correlationSaveFig}")
        return