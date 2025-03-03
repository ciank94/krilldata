import logging
import pandas as pd
import numpy as np
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
import os
import matplotlib.pyplot as plt
import seaborn as sns


class KrillClassify:
    # logger
    logging.basicConfig(level=logging.INFO)
    loggerDescription = "\nKrillClassify class description:\n\
        reads fusedData from output of DataFusion class\n\
        ML preprocess: feature scaling, creating binary target (zero/non-zero), train/test split\n\
        ML model: random forest classifier\n\
        ML postprocess: predict class probability\n\
        save trained classifier model\n\
        load trained classifier model\n\
        predict\n"

    fusedDataFilename = "krillFusedData.csv"
    confusionMatrixSaveFig = "classifierConfusionMatrix.png"
    
    # Dictionary of available classifier model classes
    models = {
        'rfc': RandomForestClassifier,
        'dtc': DecisionTreeClassifier, 
        'svc': SVC,
        'lr': LogisticRegression
    }

    # Using the same feature columns as the regressor for consistency
    featureColumns = ["LONGITUDE", "LATITUDE", "DEPTH", "CHL_MEAN", "CHL_MIN", "CHL_MAX", "IRON_MEAN", 
                     "IRON_MIN", "IRON_MAX", "SSH_MEAN", "SSH_MIN", "SSH_MAX", "VEL_MEAN", 
                     "VEL_MIN", "VEL_MAX", "SST_MEAN", "SST_MIN", "SST_MAX"]
    targetColumn = "STANDARDISED_KRILL_UNDER_1M2"

    def __init__(self, inputPath, outputPath, modelType, scenario='default'):
        #====Instance variables====
        self.inputPath = inputPath
        self.outputPath = outputPath
        self.fusedDataPath = f"{inputPath}/{KrillClassify.fusedDataFilename}"
        self.fusedData = {}
        self.df = None
        self.X = None
        self.y = None  # Binary classification target
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.modelType = modelType
        self.model_filename = f"{self.inputPath}/{self.modelType}Classifier.joblib"
        self.modelParams = json.load(open("config/classifier_params.json"))
        self.model_exists = False
        self.scenario = scenario
        self.threshold = 0  # Threshold for classification (presence/absence)

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
        """Check if a trained classifier model already exists and load it if found."""
        if os.path.exists(self.model_filename):
            self.logger.info(f"Classifier model already exists at {self.model_filename}")
            self.model_exists = True
        else:
            self.logger.info(f"No classifier model found, training new model")
            self.model_exists = False
        return

    def preprocess(self):
        """Preprocess data for classification"""
        self.scaleFeatures()
        self.handleNan()
        self.createBinaryTarget()
        self.loadXy()
        return

    def handleNan(self):
        """Handle missing values in the dataset using median imputation."""
        self.logger.info(f"Handling NaN values...")
        self.logger.info(f"Before: {self.df.isna().sum()}")
        self.df[KrillClassify.featureColumns] = self.df[KrillClassify.featureColumns].fillna(self.df[KrillClassify.featureColumns].median())
        self.logger.info(f"After: {self.df.isna().sum()}")
        self.logger.info(f"Finished handling NaN with median imputation")
        return

    def training(self):
        """Execute the training pipeline for the classifier in the correct sequence."""
        self.trainTestSplit()
        if self.scenario == "default":
            self.trainModel()
        else:
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
        self.logger.info(f"{KrillClassify.loggerDescription}")
        self.logger.info(f"Loaded parameters for {self.modelType}: {self.modelParams[self.modelType]}")
        return

    def readData(self):
        """Read the fused data file."""
        self.df = pd.read_csv(self.fusedDataPath)
        self.df = self.df[KrillClassify.featureColumns + [KrillClassify.targetColumn]]
        self.logger.info(f"Read features from {self.fusedDataPath}")
        self.logger.info(f"\n{self.df.head()}")
        return

    def describeData(self):
        """Describe the dataset and analyze class distribution."""
        self.logger.info(f"Describing data...")
        self.logger.info(f"Dataset:\n {self.df.describe()}")
        
        # Count zeros vs non-zeros
        zero_count = (self.df[KrillClassify.targetColumn] == -2.0).sum()
        nonzero_count = (self.df[KrillClassify.targetColumn] > -2.0).sum()
        
        self.logger.info(f"Class distribution:")
        self.logger.info(f"  Zero values (absence): {zero_count}")
        self.logger.info(f"  Non-zero values (presence): {nonzero_count}")
        self.logger.info(f"  Zero-inflation ratio: {zero_count / (zero_count + nonzero_count):.2f}")
        
        return

    def createBinaryTarget(self):
        """Create binary target for classification (0 = absence, 1 = presence)
        In this dataset, -2.0 values represent absence of krill (zeros)
        and all other values represent presence."""
        self.logger.info(f"Creating binary target for classification...")
        self.df['TARGET_CLASS'] = (self.df[KrillClassify.targetColumn] != -2.0).astype(int)
        self.logger.info(f"Binary target created: 0 = absence (-2.0 values), 1 = presence (all other values)")
        self.logger.info(f"Class counts: {self.df['TARGET_CLASS'].value_counts()}")
        return
        
    def scaleFeatures(self):
        """Scale features to standardize their values."""
        self.logger.info(f"Scaling features...")
        for col in KrillClassify.featureColumns:
            self.df[col] = (self.df[col] - self.df[col].mean()) / (self.df[col].std() + 0.00001)
            self.logger.info(f"Feature {col} scaled, with mean {self.df[col].mean():.6f} and std {self.df[col].std():.6f}")
        self.logger.info(f"Finished scaling features")
        return

    def loadXy(self):
        """Load feature matrix X and target vector y."""
        self.logger.info(f"Loading X and y...") 
        self.X = self.df[KrillClassify.featureColumns]
        self.y = self.df['TARGET_CLASS']
        self.logger.info(f"X shape: {self.X.shape}, y shape: {self.y.shape}")
        return

    #====================ML methods====================
    def trainTestSplit(self):
        """Split data into training and testing sets."""
        self.logger.info(f"Splitting train/test...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        self.logger.info(f"Finished train/test split with {len(self.X_train)} training samples and {len(self.X_test)} test samples")
        self.logger.info(f"Training class distribution: {pd.Series(self.y_train).value_counts()}")
        self.logger.info(f"Testing class distribution: {pd.Series(self.y_test).value_counts()}")
        return

    def trainModel(self):
        """Train the classifier model."""
        self.logger.info(f"Training {self.modelType} classifier model...")
        if self.modelType not in KrillClassify.models:
            raise ValueError(f"Model type '{self.modelType}' not supported. Choose from: {list(KrillClassify.models.keys())}")
        
        model_class = KrillClassify.models[self.modelType]
        model_params = self.modelParams[self.modelType]
        self.model = model_class(**model_params)
        self.model.fit(self.X_train, self.y_train)
        self.logger.info(f"Model parameters: {self.model.get_params()}")
        return

    def trainModelRandomSearch(self):
        """Run random search for hyperparameter optimization."""
        self.logger.info(f"Training {self.modelType} model with random search...")
        if self.modelType not in KrillClassify.models:
            raise ValueError(f"Model type '{self.modelType}' not supported. Choose from: {list(KrillClassify.models.keys())}")

        # Run random search
        kwargs = self.modelParams["Search"][self.modelType]
        model_class = KrillClassify.models[self.modelType]
        self.logger.info(f"Running random search for {self.modelType} model...")
        self.model = RandomizedSearchCV(model_class(), kwargs, n_iter=10, cv=5, random_state=42, verbose=2)
        self.model.fit(self.X_train, self.y_train)
        self.logger.info(f"Best parameters: {self.model.best_params_}")
        return

    def modelMetrics(self):
        """Calculate classification metrics."""
        self.logger.info(f"Calculating classifier metrics...")
        
        # Predict on test set
        y_pred = self.model.predict(self.X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        
        # Log metrics
        self.logger.info(f"Accuracy: {accuracy:.4f}")
        self.logger.info(f"Precision: {precision:.4f}")
        self.logger.info(f"Recall: {recall:.4f}")
        self.logger.info(f"F1 Score: {f1:.4f}")
        
        # Create and save confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        self.plot_confusion_matrix()
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, self.X, self.y, cv=5)
        self.logger.info(f"Cross-validation accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Store metrics
        self.metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        return
    

    def saveMetrics(self):
        """Save metrics to JSON file."""
        metrics_filename = f"{self.outputPath}/{self.modelType}ClassifierMetrics.json"
        with open(metrics_filename, 'w') as f:
            json.dump(self.metrics, f, indent=4)
        self.logger.info(f"Metrics saved to {metrics_filename}")
        return

    def saveModel(self):
        """Save trained model to disk."""
        dump(self.model, self.model_filename)
        self.logger.info(f"Model saved to {self.model_filename}")
        return

    def loadModel(self):
        """Load trained model from disk."""
        self.model = load(self.model_filename)
        self.logger.info(f"Model loaded from {self.model_filename}")
        return

    def predict(self, X):
        """Predict class (0 = absence, 1 = presence) for new data."""
        if self.model is None:
            self.loadModel()
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict class probabilities for new data."""
        if self.model is None:
            self.loadModel()
        return self.model.predict_proba(X)

    def featureImportance(self):
        """Calculate and plot feature importance."""
        if self.model is None:
            self.loadModel()
            
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_[0])
        else:
            self.logger.warning("Model doesn't support feature importance")
            return
            
        feature_importance = pd.DataFrame({
            'Feature': KrillClassify.featureColumns,
            'Importance': importances
        })
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance)
        plt.title('Feature Importance for Krill Presence/Absence')
        plt.tight_layout()
        plt.savefig(f"{self.outputPath}/{self.modelType}ClassifierFeatureImportance.png")
        plt.close()
        
        self.logger.info(f"Top 5 important features:")
        for i, row in feature_importance.head(5).iterrows():
            self.logger.info(f"  {row['Feature']}: {row['Importance']:.4f}")
            
        return feature_importance
    
    def plot_spatial_predictions(self):
        """Plot spatial distribution of predictions vs actual values."""
        # Get predictions
        y_pred = self.model.predict(self.X_test)
        
        # Get the original coordinates
        test_df = pd.DataFrame({
            'LONGITUDE': self.X_test['LONGITUDE'],
            'LATITUDE': self.X_test['LATITUDE'],
            'y_true': self.y_test,
            'y_pred': y_pred
        })
        
        # Categorize predictions
        test_df['result'] = 'Unknown'
        test_df.loc[(test_df['y_true'] == 1) & (test_df['y_pred'] == 1), 'result'] = 'True Positive'
        test_df.loc[(test_df['y_true'] == 0) & (test_df['y_pred'] == 0), 'result'] = 'True Negative'
        test_df.loc[(test_df['y_true'] == 0) & (test_df['y_pred'] == 1), 'result'] = 'False Positive'
        test_df.loc[(test_df['y_true'] == 1) & (test_df['y_pred'] == 0), 'result'] = 'False Negative'
        
        # Plot
        plt.figure(figsize=(12, 8))
        colors = {'True Positive': 'green', 'True Negative': 'blue', 
                 'False Positive': 'red', 'False Negative': 'orange'}
        
        for result, group in test_df.groupby('result'):
            plt.scatter(group['LONGITUDE'], group['LATITUDE'], c=colors[result], label=result, alpha=0.7)
        
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('Spatial Distribution of Classification Results')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{self.outputPath}/{self.modelType}_spatial_predictions.png")
        plt.close()
        
        self.logger.info(f"Spatial prediction map saved to {self.outputPath}/{self.modelType}_spatial_predictions.png")
        return

    def plot_confusion_matrix(self):
        """Plot confusion matrix for model predictions."""
        if self.model is None:
            self.logger.error("Model not trained yet!")
            return
            
        self.logger.info("Plotting confusion matrix...")
        y_pred = self.model.predict(self.X)
        cm = confusion_matrix(self.y, y_pred)
        
        plt.figure(figsize=(10, 8))
       
        # Create heatmap with a different colormap and normalization
        ax = sns.heatmap(cm, annot=True, fmt='d', 
                    cmap='YlOrRd',  # Yellow-Orange-Red sequential colormap
                    xticklabels=['Absence', 'Presence'],
                    yticklabels=['Absence', 'Presence'],
                    annot_kws={'size': 16},  # Larger numbers
                    vmin=0,  # Minimum value for colorbar
                    vmax=np.max(cm),  # Maximum value from confusion matrix
                    cbar_kws={'label': 'Number of Samples'})
       
        # Modify colorbar properties after creation
        cbar = ax.collections[0].colorbar
        cbar.ax.set_ylabel('Number of Samples', fontsize=14)
        cbar.ax.tick_params(labelsize=14)
       
        # Customize the plot
        plt.title('Confusion Matrix', fontsize=16, pad=20)
        plt.ylabel('True Label', fontsize=14, labelpad=10)
        plt.xlabel('Predicted Label', fontsize=14, labelpad=10)
       
        # Increase tick label size
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16, rotation=0)
       
        # Add total samples text with larger font
        plt.text(0.5, -0.15, f'Total samples: {len(self.y):,}', 
                ha='center', va='center', transform=plt.gca().transAxes, 
                fontsize=14)
       
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        plt.savefig(f"{self.outputPath}/{self.modelType}_confusion_matrix.png")
        plt.close()
        
        # Calculate and log metrics
        accuracy = accuracy_score(self.y, y_pred)
        precision = precision_score(self.y, y_pred)
        recall = recall_score(self.y, y_pred)
        f1 = f1_score(self.y, y_pred)
        
        self.logger.info("Classification metrics:")
        self.logger.info(f"  Accuracy: {accuracy:.4f}")
        self.logger.info(f"  Precision: {precision:.4f}")
        self.logger.info(f"  Recall: {recall:.4f}")
        self.logger.info(f"  F1-score: {f1:.4f}")
        
        self.logger.info(f"Confusion matrix saved to {self.outputPath}/{self.modelType}_confusion_matrix.png")
        return


# Example usage
if __name__ == "__main__":
    # Parameters
    input_path = "input"
    output_path = "output" 
    model_type = "rfc"  # Random Forest Classifier
    
    # Create and train classifier
    classifier = KrillClassify(input_path, output_path, model_type)
    
    # Feature importance
    classifier.featureImportance()