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
        self.preprocessData()
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
        return
    
    def preprocessData(self):
        """Preprocess data for both classification and regression."""
        # Get feature columns and target column
        features = KrillClassify.featureColumns
        target = KrillClassify.targetColumn
        
        # Check for NaN values before preprocessing
        nan_count_before = self.df[features + [target]].isna().sum().sum()
        self.logger.info(f"NaN values before preprocessing: {nan_count_before}")
        
        # Handle missing values
        self.handleNan()
        
        # Scale features
        self.scaleFeatures()
        
        # Extract features and target
        self.X = self.df[features]
        self.y = self.df[target]
        
        # Final check for NaN values
        nan_count_x = self.X.isna().sum().sum()
        nan_count_y = self.y.isna().sum()
        
        # Store original dataframe before any row dropping
        self.original_df = self.df.copy()
        
        # Initialize valid indices to track which rows are kept
        self.valid_indices = np.ones(len(self.df), dtype=bool)
        
        if nan_count_x > 0 or nan_count_y > 0:
            self.logger.warning(f"NaN values remain after preprocessing: X={nan_count_x}, y={nan_count_y}")
            self.logger.info("Dropping remaining rows with NaN values...")
            # Get indices of rows with NaN values
            nan_indices = pd.concat([self.X.isna().any(axis=1), self.y.isna()], axis=1).any(axis=1)
            # Update valid indices
            self.valid_indices = ~nan_indices
            # Drop rows with NaN values
            self.X = self.X[self.valid_indices]
            self.y = self.y[self.valid_indices]
            # Also update the main dataframe to keep it in sync
            self.df = self.df[self.valid_indices].reset_index(drop=True)
            self.logger.info(f"After dropping NaN rows: X shape: {self.X.shape}, y shape: {self.y.shape}")
        
        self.logger.info(f"Data preprocessing completed: X shape: {self.X.shape}, y shape: {self.y.shape}")
        return
    
    def handleNan(self):
        """Handle missing values in the dataset using median imputation."""
        self.logger.info(f"Handling NaN values...")
        feature_cols = KrillClassify.featureColumns
        
        # Count NaN values before imputation
        nan_counts = self.df[feature_cols].isna().sum()
        self.logger.info(f"NaN values per feature before imputation:\n{nan_counts}")
        total_nans = nan_counts.sum()
        self.logger.info(f"Total NaN values in features: {total_nans}")
        
        # Impute NaN values in features with median
        if total_nans > 0:
            self.df[feature_cols] = self.df[feature_cols].fillna(
                self.df[feature_cols].median()
            )
            self.logger.info(f"NaN values in features imputed with median values")
        
        # Check for NaN values in target column
        target_nans = self.df[KrillClassify.targetColumn].isna().sum()
        if target_nans > 0:
            self.logger.warning(f"Found {target_nans} NaN values in target column")
            # For target column, we'll keep NaN values for now and handle them later
        
        # Verify imputation
        remaining_nans = self.df[feature_cols].isna().sum().sum()
        self.logger.info(f"Remaining NaN values in features after imputation: {remaining_nans}")
        return
    
    def scaleFeatures(self):
        """Scale features to standardize their values."""
        self.logger.info(f"Scaling features...")
        for col in KrillClassify.featureColumns:
            self.df[col] = (self.df[col] - self.df[col].mean()) / (self.df[col].std() + 0.00001)
            self.logger.info(f"Feature {col} scaled, with mean {self.df[col].mean():.6f} and std {self.df[col].std():.6f}")
        self.logger.info(f"Finished scaling features")
        return
        
    def loadModels(self):
        """Load classifier and regressor models from joblib files."""
        # Load the classifier
        classifier_filename = f"{self.inputPath}/{self.classifier_type}Classifier.joblib"
        regressor_filename = f"{self.inputPath}/{self.regressor_type}Model.joblib"
        
        # Check if classifier file exists
        if not os.path.exists(classifier_filename):
            self.logger.error(f"Classifier model not found at {classifier_filename}")
            raise FileNotFoundError(f"Classifier model file not found: {classifier_filename}")
        
        # Check if regressor file exists
        if not os.path.exists(regressor_filename):
            self.logger.error(f"Regressor model not found at {regressor_filename}")
            raise FileNotFoundError(f"Regressor model file not found: {regressor_filename}")
        
        # Load both models
        self.logger.info(f"Loading classifier from {classifier_filename}...")
        self.classifier = load(classifier_filename)
        self.logger.info(f"Classifier model loaded successfully")
        
        self.logger.info(f"Loading regressor from {regressor_filename}...")
        self.regressor = load(regressor_filename)
        self.logger.info(f"Regressor model loaded successfully")
        
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
        
        # Step 2: Regression (density for presence cases only)
        # Initialize with -2.0 (log-transformed zero) for all samples
        self.y_pred_combined = np.full(len(X), -2.0)
        
        # Get indices where classifier predicts presence (1)
        presence_indices = np.where(self.y_pred_class == 1)[0]
        self.logger.info(f"Running regression model on {len(presence_indices)} presence samples")
        
        # Only run regression on presence cases
        if len(presence_indices) > 0:
            # Extract presence samples
            X_presence = X.iloc[presence_indices]
            
            # Apply regression model only to presence samples
            presence_predictions = self.regressor.predict(X_presence)
            
            # Update combined predictions for presence cases
            self.y_pred_combined[presence_indices] = presence_predictions
        
        self.logger.info(f"Combined predictions complete.")
        return self.y_pred_combined
    
    def visualize_predictions(self, save_plots=True):
        """Create visualizations to compare predictions with observations.
        
        Args:
            save_plots (bool): Whether to save plots to files
            
        Returns:
            dict: Dictionary of created figures
        """
        if self.y_pred_combined is None:
            self.predict()
            
        self.logger.info("Creating visualization plots...")
        figures = {}
        
        # 1. Scatter plot of predictions vs observations
        self.logger.info("Creating scatter plot of predictions vs observations...")
        fig_scatter, ax_scatter = plt.subplots(figsize=(10, 8))
        
        # Filter out absence cases (-2.0) for both predictions and observations
        presence_mask = (self.y != -2.0) & (self.y_pred_combined != -2.0)
        
        if sum(presence_mask) > 0:
            # Create scatter plot for presence cases
            ax_scatter.scatter(self.y[presence_mask], self.y_pred_combined[presence_mask], 
                             alpha=0.6, edgecolor='k', s=50)
            
            # Add perfect prediction line
            min_val = min(self.y[presence_mask].min(), self.y_pred_combined[presence_mask].min())
            max_val = max(self.y[presence_mask].max(), self.y_pred_combined[presence_mask].max())
            ax_scatter.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
            
            # Add regression line
            from scipy import stats
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                self.y[presence_mask], self.y_pred_combined[presence_mask]
            )
            regression_line = slope * np.array([min_val, max_val]) + intercept
            ax_scatter.plot([min_val, max_val], regression_line, 'g-', linewidth=2)
            
            # Add R² value to plot
            ax_scatter.text(0.05, 0.95, f'R² = {r_value**2:.4f}', transform=ax_scatter.transAxes,
                          fontsize=12, verticalalignment='top')
            
            ax_scatter.set_xlabel('Observed Values (log10 scale)')
            ax_scatter.set_ylabel('Predicted Values (log10 scale)')
            ax_scatter.set_title('Predicted vs Observed Krill Density (Presence Only)')
            ax_scatter.grid(True, linestyle='--', alpha=0.7)
            
            figures['scatter'] = fig_scatter
            
            if save_plots:
                scatter_path = f"{self.outputPath}/predicted_vs_observed_scatter.png"
                fig_scatter.savefig(scatter_path)
                self.logger.info(f"Scatter plot saved to {scatter_path}")
        else:
            self.logger.warning("No matching presence cases found for scatter plot")
        
        # 2. Q-Q plot for presence cases
        self.logger.info("Creating Q-Q plot for presence cases...")
        fig_qq, ax_qq = plt.subplots(figsize=(10, 8))
        
        if sum(presence_mask) > 0:
            # Get quantiles for both observed and predicted values
            from scipy import stats
            
            # Sort the observed and predicted values
            obs_sorted = np.sort(self.y[presence_mask])
            pred_sorted = np.sort(self.y_pred_combined[presence_mask])
            
            # Create Q-Q plot
            ax_qq.scatter(obs_sorted, pred_sorted, alpha=0.6, edgecolor='k', s=50)
            
            # Add perfect match line
            min_val = min(obs_sorted.min(), pred_sorted.min())
            max_val = max(obs_sorted.max(), pred_sorted.max())
            ax_qq.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
            
            ax_qq.set_xlabel('Observed Quantiles (log10 scale)')
            ax_qq.set_ylabel('Predicted Quantiles (log10 scale)')
            ax_qq.set_title('Q-Q Plot of Predicted vs Observed Krill Density (Presence Only)')
            ax_qq.grid(True, linestyle='--', alpha=0.7)
            
            figures['qq'] = fig_qq
            
            if save_plots:
                qq_path = f"{self.outputPath}/predicted_vs_observed_qq.png"
                fig_qq.savefig(qq_path)
                self.logger.info(f"Q-Q plot saved to {qq_path}")
        else:
            self.logger.warning("No matching presence cases found for Q-Q plot")
        
        # 3. Histogram of residuals for presence cases
        self.logger.info("Creating histogram of residuals...")
        fig_hist, ax_hist = plt.subplots(figsize=(10, 8))
        
        if sum(presence_mask) > 0:
            # Calculate residuals
            residuals = self.y_pred_combined[presence_mask] - self.y[presence_mask]
            
            # Create histogram
            ax_hist.hist(residuals, bins=30, alpha=0.7, edgecolor='k')
            
            # Add vertical line at zero
            ax_hist.axvline(x=0, color='r', linestyle='--', linewidth=2)
            
            # Calculate mean and std of residuals
            mean_residual = np.mean(residuals)
            std_residual = np.std(residuals)
            
            # Add mean and std to plot
            ax_hist.text(0.05, 0.95, f'Mean: {mean_residual:.4f}\nStd: {std_residual:.4f}', 
                       transform=ax_hist.transAxes, fontsize=12, verticalalignment='top')
            
            ax_hist.set_xlabel('Residuals (Predicted - Observed)')
            ax_hist.set_ylabel('Frequency')
            ax_hist.set_title('Histogram of Residuals (Presence Only)')
            ax_hist.grid(True, linestyle='--', alpha=0.7)
            
            figures['histogram'] = fig_hist
            
            if save_plots:
                hist_path = f"{self.outputPath}/residuals_histogram.png"
                fig_hist.savefig(hist_path)
                self.logger.info(f"Histogram saved to {hist_path}")
        else:
            self.logger.warning("No matching presence cases found for residuals histogram")
        
        plt.close('all')  # Close all figures to free memory
        self.logger.info("Visualization complete.")
        return figures
    
    def evaluate(self):
        """Evaluate the hurdle model performance."""
        if self.y_pred_combined is None:
            self.predict()
            
        self.logger.info("Evaluating hurdle model performance...")
        
        # Check for NaN values in predictions or actual values
        y_nans = np.isnan(self.y).sum()
        pred_nans = np.isnan(self.y_pred_combined).sum()
        
        if y_nans > 0 or pred_nans > 0:
            self.logger.warning(f"Found NaN values: {y_nans} in actual values, {pred_nans} in predictions")
            self.logger.info("Removing samples with NaN values for evaluation...")
            
            # Create mask for non-NaN values
            valid_mask = ~(np.isnan(self.y) | np.isnan(self.y_pred_combined))
            valid_y = self.y[valid_mask]
            valid_pred = self.y_pred_combined[valid_mask]
            
            self.logger.info(f"Evaluation will use {sum(valid_mask)} out of {len(self.y)} samples")
        else:
            valid_y = self.y
            valid_pred = self.y_pred_combined
        
        # Calculate overall metrics
        presence_mask = (valid_y != -2.0)
        absence_mask = (valid_y == -2.0)
        
        # 1. Classification metrics
        true_positives = sum((valid_y != -2.0) & (valid_pred != -2.0))
        true_negatives = sum((valid_y == -2.0) & (valid_pred == -2.0))
        false_positives = sum((valid_y == -2.0) & (valid_pred != -2.0))
        false_negatives = sum((valid_y != -2.0) & (valid_pred == -2.0))
        
        total = len(valid_y)
        accuracy = (true_positives + true_negatives) / total if total > 0 else 0
        
        self.logger.info(f"Classification accuracy: {accuracy:.4f}")
        self.logger.info(f"True positives: {true_positives}")
        self.logger.info(f"True negatives: {true_negatives}")
        self.logger.info(f"False positives: {false_positives}")
        self.logger.info(f"False negatives: {false_negatives}")
        
        # 2. Regression metrics (only for true presences)
        # Filter only where both true and predicted values are presences
        true_presence_mask = (valid_y != -2.0) & (valid_pred != -2.0)
        if sum(true_presence_mask) > 0:
            # Extract non-NaN values for regression metrics
            y_true_presence = valid_y[true_presence_mask]
            y_pred_presence = valid_pred[true_presence_mask]
            
            # Final check for any remaining NaN values
            final_valid_mask = ~(np.isnan(y_true_presence) | np.isnan(y_pred_presence))
            if not all(final_valid_mask):
                self.logger.warning(f"Removing {sum(~final_valid_mask)} samples with NaN values from regression metrics")
                y_true_presence = y_true_presence[final_valid_mask]
                y_pred_presence = y_pred_presence[final_valid_mask]
            
            if len(y_true_presence) > 0:
                rmse = np.sqrt(mean_squared_error(y_true_presence, y_pred_presence))
                r2 = r2_score(y_true_presence, y_pred_presence)
                self.logger.info(f"Regression RMSE (presence only): {rmse:.4f}")
                self.logger.info(f"Regression R2 (presence only): {r2:.4f}")
            else:
                self.logger.warning("No valid samples for regression metrics")
        
        # 3. Overall performance
        # Calculate metrics ignoring misclassifications
        correctly_classified = ((valid_y == -2.0) & (valid_pred == -2.0)) | ((valid_y != -2.0) & (valid_pred != -2.0))
        if sum(correctly_classified) > 0:
            # Extract values for correctly classified samples
            y_correct = valid_y[correctly_classified]
            pred_correct = valid_pred[correctly_classified]
            
            # Final check for any remaining NaN values
            final_valid_mask = ~(np.isnan(y_correct) | np.isnan(pred_correct))
            if not all(final_valid_mask):
                self.logger.warning(f"Removing {sum(~final_valid_mask)} samples with NaN values from overall metrics")
                y_correct = y_correct[final_valid_mask]
                pred_correct = pred_correct[final_valid_mask]
            
            if len(y_correct) > 0:
                rmse_correct = np.sqrt(mean_squared_error(y_correct, pred_correct))
                self.logger.info(f"Overall RMSE (correctly classified): {rmse_correct:.4f}")
            else:
                self.logger.warning("No valid samples for overall metrics")
        
        # Create visualizations
        self.visualize_predictions()
        
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
            
            ax.set_xlabel('Predicted Values')
            ax.set_ylabel('Residuals')
            ax.set_title('Residuals vs Predicted Values')
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
            ax.set_xlabel('Residuals')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of Residuals')
            
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
            
        self.logger.info(f"Saving predictions to CSV file...")
        
        # There are two approaches we can take:
        # 1. Save predictions only for valid rows (those without NaN values)
        # 2. Save predictions for all rows, with NaN for rows that were dropped
        
        # Approach 1: Save predictions only for valid rows
        pred_df = self.df.copy()
        pred_df['PREDICTED_CLASS'] = self.y_pred_class
        pred_df['PREDICTED_VALUE'] = self.y_pred_combined
        
        # Add error columns (only for valid rows)
        pred_df['ERROR'] = pred_df[KrillClassify.targetColumn] - pred_df['PREDICTED_VALUE']
        pred_df['ABS_ERROR'] = np.abs(pred_df['ERROR'])
        
        # Save to CSV
        output_file = f"{self.outputPath}/hurdle_predictions_valid_only.csv"
        pred_df.to_csv(output_file, index=False)
        self.logger.info(f"Predictions for valid rows saved to {output_file}")
        
        # Approach 2: Save predictions for all rows, with NaN for rows that were dropped
        full_pred_df = self.original_df.copy()
        
        # Initialize prediction columns with NaN
        full_pred_df['PREDICTED_CLASS'] = np.nan
        full_pred_df['PREDICTED_VALUE'] = np.nan
        full_pred_df['ERROR'] = np.nan
        full_pred_df['ABS_ERROR'] = np.nan
        
        # Fill in predictions for valid rows
        full_pred_df.loc[self.valid_indices, 'PREDICTED_CLASS'] = self.y_pred_class
        full_pred_df.loc[self.valid_indices, 'PREDICTED_VALUE'] = self.y_pred_combined
        
        # Calculate error metrics for valid rows
        valid_mask = self.valid_indices & ~full_pred_df[KrillClassify.targetColumn].isna()
        full_pred_df.loc[valid_mask, 'ERROR'] = (
            full_pred_df.loc[valid_mask, KrillClassify.targetColumn] - 
            full_pred_df.loc[valid_mask, 'PREDICTED_VALUE']
        )
        full_pred_df.loc[valid_mask, 'ABS_ERROR'] = np.abs(full_pred_df.loc[valid_mask, 'ERROR'])
        
        # Save complete dataframe to CSV
        output_file_full = f"{self.outputPath}/hurdle_predictions_all.csv"
        full_pred_df.to_csv(output_file_full, index=False)
        self.logger.info(f"Predictions for all rows saved to {output_file_full}")
        
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