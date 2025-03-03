import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from joblib import load
import os

from krilldata import KrillClassify, KrillTrain

class KrillHurdle:
    # Logger configuration
    logging.basicConfig(level=logging.INFO)
    loggerDescription = "\nKrillHurdle class description:\n\
        Implements a two-stage hurdle model for krill density prediction:\n\
        1. Classification stage: Predicts presence/absence (KrillClassify)\n\
        2. Regression stage: Predicts density for presence cases (KrillTrain)\n\
        Combines both models for final predictions and evaluation\n"

    fusedDataFilename = "krillFusedData.csv"
    
    def __init__(self, inputPath, outputPath, classifier_type="rfc", regressor_type="rfr"):
        # Instance variables
        self.inputPath = inputPath
        self.outputPath = outputPath
        self.fusedDataPath = f"{inputPath}/{KrillHurdle.fusedDataFilename}"
        self.df = None
        self.classifier_type = classifier_type
        self.regressor_type = regressor_type
        
        # Model objects
        self.classifier = None
        self.regressor = None
        
        # Predictions
        self.y_pred_class = None
        self.y_pred_reg = None
        self.y_pred_combined = None
        
        # Original data
        self.X = None
        self.y = None
        
        # Initialization
        self.initLogger()
        self.readData()
        self.loadModels()
        
    def initLogger(self):
        """Initialize logger."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"================={self.__class__.__name__}=====================")
        self.logger.info(f"{KrillHurdle.loggerDescription}")
        return
        
    def readData(self):
        """Read the fused data file."""
        self.df = pd.read_csv(self.fusedDataPath)
        self.logger.info(f"Read data from {self.fusedDataPath}, {len(self.df)} rows")
        
        # For model evaluation, we'll need the features and target
        features = KrillClassify.featureColumns
        target = KrillClassify.targetColumn
        
        # Handle missing values
        self.handleNan()
        
        self.X = self.df[features]
        self.y = self.df[target]
        return
    
    def handleNan(self):
        """Handle missing values in the dataset using median imputation."""
        self.logger.info(f"Handling NaN values...")
        self.logger.info(f"Before: {self.df[KrillClassify.featureColumns].isna().sum().sum()} NaN values")
        self.df[KrillClassify.featureColumns] = self.df[KrillClassify.featureColumns].fillna(
            self.df[KrillClassify.featureColumns].median()
        )
        self.logger.info(f"After: {self.df[KrillClassify.featureColumns].isna().sum().sum()} NaN values")
        return
        
    def loadModels(self):
        """Load classifier and regressor models."""
        # Initialize the classifier
        self.logger.info(f"Loading classifier ({self.classifier_type})...")
        classifier_filename = f"{self.inputPath}/{self.classifier_type}Classifier.joblib"
        if os.path.exists(classifier_filename):
            self.classifier = load(classifier_filename)
            self.logger.info(f"Classifier model loaded from {classifier_filename}")
        else:
            self.logger.error(f"Classifier model not found at {classifier_filename}")
            # Create a new classifier
            self.logger.info("Training a new classifier...")
            classifier_obj = KrillClassify(self.inputPath, self.outputPath, self.classifier_type)
            self.classifier = classifier_obj.model
        
        # Initialize the regressor
        self.logger.info(f"Loading regressor ({self.regressor_type})...")
        regressor_filename = f"{self.inputPath}/{self.regressor_type}Model.joblib"
        if os.path.exists(regressor_filename):
            self.regressor = load(regressor_filename)
            self.logger.info(f"Regressor model loaded from {regressor_filename}")
        else:
            self.logger.error(f"Regressor model not found at {regressor_filename}")
            # Create a new regressor
            self.logger.info("Training a new regressor...")
            regressor_obj = KrillTrain(self.inputPath, self.outputPath, self.regressor_type)
            self.regressor = regressor_obj.model
        
        # Check if both models are initialized
        if self.classifier is None:
            self.logger.error("Classifier model was not initialized!")
        if self.regressor is None:
            self.logger.error("Regressor model was not initialized!")
        return
        
    def predict(self, X=None):
        """Make predictions using the hurdle model.
        
        If X is not provided, uses the data loaded at initialization.
        
        Args:
            X (pd.DataFrame, optional): Feature data for prediction
            
        Returns:
            pd.Series: Combined predictions from the hurdle model
        """
        if X is None:
            X = self.X
            
        self.logger.info(f"Making predictions on {len(X)} samples...")
        
        # Step 1: Classification (presence/absence)
        self.y_pred_class = self.classifier.predict(X)
        self.logger.info(f"Classification predictions: {sum(self.y_pred_class)} presences, {len(X) - sum(self.y_pred_class)} absences")
        
        # Step 2: Regression (density for presence cases)
        # For absence cases, we'll set the prediction to -2.0 (log-transformed zero)
        self.y_pred_reg = self.regressor.predict(X)
        
        # Step 3: Combine predictions
        # Where classifier predicts absence (0), use -2.0
        # Where classifier predicts presence (1), use regression prediction
        self.y_pred_combined = np.where(self.y_pred_class == 0, -2.0, self.y_pred_reg)
        
        self.logger.info(f"Combined predictions complete.")
        return self.y_pred_combined
    
    def evaluate(self):
        """Evaluate the hurdle model performance."""
        if self.y_pred_combined is None:
            self.predict()
            
        self.logger.info("Evaluating hurdle model performance...")
        
        # Calculate overall metrics
        presence_mask = (self.y != -2.0)
        absence_mask = (self.y == -2.0)
        
        # 1. Classification metrics
        true_positives = sum((self.y != -2.0) & (self.y_pred_combined != -2.0))
        true_negatives = sum((self.y == -2.0) & (self.y_pred_combined == -2.0))
        false_positives = sum((self.y == -2.0) & (self.y_pred_combined != -2.0))
        false_negatives = sum((self.y != -2.0) & (self.y_pred_combined == -2.0))
        
        total = len(self.y)
        accuracy = (true_positives + true_negatives) / total
        
        self.logger.info(f"Classification accuracy: {accuracy:.4f}")
        self.logger.info(f"True positives: {true_positives}")
        self.logger.info(f"True negatives: {true_negatives}")
        self.logger.info(f"False positives: {false_positives}")
        self.logger.info(f"False negatives: {false_negatives}")
        
        # 2. Regression metrics (only for true presences)
        # Filter only where both true and predicted values are presences
        true_presence_mask = (self.y != -2.0) & (self.y_pred_combined != -2.0)
        if sum(true_presence_mask) > 0:
            rmse = np.sqrt(mean_squared_error(
                self.y[true_presence_mask], 
                self.y_pred_combined[true_presence_mask]
            ))
            r2 = r2_score(
                self.y[true_presence_mask], 
                self.y_pred_combined[true_presence_mask]
            )
            self.logger.info(f"Regression RMSE (presence only): {rmse:.4f}")
            self.logger.info(f"Regression R2 (presence only): {r2:.4f}")
        
        # 3. Overall performance
        # Calculate metrics ignoring misclassifications
        correctly_classified = (self.y == -2.0) & (self.y_pred_combined == -2.0) | (self.y != -2.0) & (self.y_pred_combined != -2.0)
        if sum(correctly_classified) > 0:
            rmse_correct = np.sqrt(mean_squared_error(
                self.y[correctly_classified], 
                self.y_pred_combined[correctly_classified]
            ))
            self.logger.info(f"Overall RMSE (correctly classified): {rmse_correct:.4f}")
        
        # Create visualizations
        self.plot_qq()
        self.plot_residuals()
        self.plot_spatial_predictions()
        
        return {
            'accuracy': accuracy,
            'true_positives': true_positives,
            'true_negatives': true_negatives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'rmse': rmse if 'rmse' in locals() else None,
            'r2': r2 if 'r2' in locals() else None,
            'rmse_correct': rmse_correct if 'rmse_correct' in locals() else None
        }
    
    def plot_qq(self):
        """Create Q-Q plot for observed vs predicted values (for presences only)."""
        if self.y_pred_combined is None:
            self.predict()
            
        # Filter to only include true presences
        presence_mask = (self.y != -2.0) & (self.y_pred_combined != -2.0)
        
        if sum(presence_mask) > 0:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Get observed and predicted values for true presences
            obs = self.y[presence_mask]
            pred = self.y_pred_combined[presence_mask]
            
            # Sort both arrays
            obs_sorted = np.sort(obs)
            pred_sorted = np.sort(pred)
            
            # Plot Q-Q line
            ax.scatter(obs_sorted, pred_sorted, alpha=0.6)
            
            # Add perfect prediction line
            min_val = min(obs.min(), pred.min())
            max_val = max(obs.max(), pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--')
            
            ax.set_xlabel('Observed Krill Density (log-transformed)', fontsize=12)
            ax.set_ylabel('Predicted Krill Density (log-transformed)', fontsize=12)
            ax.set_title('Q-Q Plot: Observed vs Predicted Krill Density', fontsize=14)
            ax.grid(True, alpha=0.3)
            
            # Add correlation coefficient
            corr = np.corrcoef(obs, pred)[0, 1]
            ax.text(0.05, 0.95, f'Correlation: {corr:.4f}', transform=ax.transAxes, 
                    fontsize=12, ha='left', va='top')
            
            plt.tight_layout()
            plt.savefig(f"{self.outputPath}/hurdle_qq_plot.png")
            plt.close()
            
            self.logger.info(f"Q-Q plot saved to {self.outputPath}/hurdle_qq_plot.png")
        else:
            self.logger.warning("Cannot create Q-Q plot: no matching presence predictions")
        
        return
        
    def plot_residuals(self):
        """Plot residuals for presence predictions."""
        if self.y_pred_combined is None:
            self.predict()
            
        # Filter to only include true presences with presence predictions
        presence_mask = (self.y != -2.0) & (self.y_pred_combined != -2.0)
        
        if sum(presence_mask) > 0:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Calculate residuals
            obs = self.y[presence_mask]
            pred = self.y_pred_combined[presence_mask]
            residuals = obs - pred
            
            # Plot residuals vs predicted values
            ax.scatter(pred, residuals, alpha=0.6)
            ax.axhline(y=0, color='r', linestyle='-', alpha=0.5)
            
            # Add lowess smoothing trend
            try:
                import statsmodels.api as sm
                lowess = sm.nonparametric.lowess(residuals, pred, frac=0.3)
                ax.plot(lowess[:, 0], lowess[:, 1], 'r-', linewidth=2)
            except ImportError:
                self.logger.warning("statsmodels not available for lowess smoothing")
            
            ax.set_xlabel('Predicted Values', fontsize=12)
            ax.set_ylabel('Residuals', fontsize=12)
            ax.set_title('Residuals vs Predicted Values', fontsize=14)
            ax.grid(True, alpha=0.3)
            
            # Add residual statistics
            mean_resid = np.mean(residuals)
            std_resid = np.std(residuals)
            ax.text(0.05, 0.05, f'Mean: {mean_resid:.4f}\nStd: {std_resid:.4f}', 
                    transform=ax.transAxes, fontsize=12, ha='left', va='bottom')
            
            plt.tight_layout()
            plt.savefig(f"{self.outputPath}/hurdle_residuals.png")
            plt.close()
            
            self.logger.info(f"Residuals plot saved to {self.outputPath}/hurdle_residuals.png")
            
            # Also create a histogram of residuals
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(residuals, bins=30, alpha=0.7, color='steelblue')
            ax.axvline(x=0, color='r', linestyle='--')
            ax.set_xlabel('Residuals', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title('Distribution of Residuals', fontsize=14)
            
            plt.tight_layout()
            plt.savefig(f"{self.outputPath}/hurdle_residuals_hist.png")
            plt.close()
            
            self.logger.info(f"Residuals histogram saved to {self.outputPath}/hurdle_residuals_hist.png")
        else:
            self.logger.warning("Cannot create residuals plot: no matching presence predictions")
        
        return
        
    def plot_spatial_predictions(self):
        """Plot spatial distribution of predictions vs actual values."""
        if self.y_pred_combined is None:
            self.predict()
            
        plt.figure(figsize=(12, 10))
        
        # Create predictions dataframe
        pred_df = pd.DataFrame({
            'LONGITUDE': self.df['LONGITUDE'],
            'LATITUDE': self.df['LATITUDE'],
            'TRUE_VALUE': self.y,
            'PREDICTED': self.y_pred_combined
        })
        
        # Define categories for visualization
        pred_df['STATUS'] = 'Unknown'
        # Correctly predicted absences (true negatives)
        pred_df.loc[(pred_df['TRUE_VALUE'] == -2.0) & (pred_df['PREDICTED'] == -2.0), 'STATUS'] = 'True Negative'
        # Correctly predicted presences (true positives, in terms of classification)
        pred_df.loc[(pred_df['TRUE_VALUE'] != -2.0) & (pred_df['PREDICTED'] != -2.0), 'STATUS'] = 'True Positive'
        # False positives (predicted presence when actually absence)
        pred_df.loc[(pred_df['TRUE_VALUE'] == -2.0) & (pred_df['PREDICTED'] != -2.0), 'STATUS'] = 'False Positive'
        # False negatives (predicted absence when actually presence)
        pred_df.loc[(pred_df['TRUE_VALUE'] != -2.0) & (pred_df['PREDICTED'] == -2.0), 'STATUS'] = 'False Negative'
        
        # Colors for each category
        colors = {
            'True Negative': 'blue',
            'True Positive': 'green',
            'False Positive': 'red',
            'False Negative': 'orange'
        }
        
        # Plot points
        for status, group in pred_df.groupby('STATUS'):
            plt.scatter(group['LONGITUDE'], group['LATITUDE'], c=colors[status], 
                       label=f"{status} (n={len(group)})", alpha=0.7, s=50)
        
        plt.xlabel('Longitude', fontsize=12)
        plt.ylabel('Latitude', fontsize=12)
        plt.title('Spatial Distribution of Hurdle Model Predictions', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f"{self.outputPath}/hurdle_spatial_predictions.png")
        plt.close()
        
        self.logger.info(f"Spatial prediction map saved to {self.outputPath}/hurdle_spatial_predictions.png")
        return
    
    def save_predictions(self):
        """Save predictions to CSV file."""
        if self.y_pred_combined is None:
            self.predict()
            
        # Create dataframe with original data and predictions
        pred_df = self.df.copy()
        pred_df['PREDICTED_CLASS'] = self.y_pred_class
        pred_df['PREDICTED_VALUE'] = self.y_pred_combined
        
        # Add error columns
        pred_df['ERROR'] = pred_df[KrillClassify.targetColumn] - pred_df['PREDICTED_VALUE']
        pred_df['ABS_ERROR'] = np.abs(pred_df['ERROR'])
        
        # Save to CSV
        output_file = f"{self.outputPath}/hurdle_predictions.csv"
        pred_df.to_csv(output_file, index=False)
        
        self.logger.info(f"Predictions saved to {output_file}")
        return

# Example usage
if __name__ == "__main__":
    # Parameters
    input_path = "input"
    output_path = "output"
    classifier_type = "rfc"
    regressor_type = "rfr"
    
    # Initialize hurdle model
    hurdle_model = KrillHurdle(input_path, output_path, classifier_type, regressor_type)
    
    # Make predictions
    predictions = hurdle_model.predict()
    
    # Evaluate model
    metrics = hurdle_model.evaluate()
    
    # Save predictions
    hurdle_model.save_predictions()