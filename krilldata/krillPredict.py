from joblib import load
import json
import numpy as np
import pandas as pd
import xarray as xr
import logging
import matplotlib.pyplot as plt
import os
from scipy.stats import gaussian_kde

#todo: predict
class KrillPredict:
    # logger description:
    loggerDescription = "\nKrillPredict class description:\n\
        predicts and maps krill density from bathymetry and SST data\n\
        visualises predictions\n\
        saves predictions\n"

    # filenames:
    bathymetryFilename = "bathymetry.nc"
    sstFilename = "sst.nc"
    sshFilename = "ssh.nc"
    fusedFilename = "krillFusedData.csv"

    def __init__(self, inputPath, outputPath, modelType, scenario='southGeorgia'):
        # Instance variables
        self.inputPath = inputPath
        self.outputPath = outputPath
        self.modelType = modelType
        self.scenario = scenario
        self.model = load(f"{inputPath}/{self.modelType}Model.joblib")
        self.mapParams = self.loadParameters()

        # Main methods
        self.initLogger()
        self.X = self.loadFeatures()
        self.y = self.predict()
        self.plotPredictions()
        return

    def initLogger(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"================={self.__class__.__name__}=====================")
        self.logger.info(f"{KrillPredict.loggerDescription}")
        self.logger.info(f"Loaded parameters for: {self.modelType}")
        self.logger.info(f"Loaded scenario parameters for: {self.scenario}")
        self.logger.info(f"Loaded model: {self.model}")
        return

    def loadParameters(self):
        """Load map parameters from JSON file"""
        with open("krilldata/map_params.json", "r") as f:
            params = json.load(f)
        return params[self.scenario]

    def loadFeatures(self):
        # Get spatial and temporal extents
        lon_grid, lat_grid = self.spatialExtent()
        time_range = self.temporalExtent()
        
        # Load bathymetry and SST data
        bathymetry_ds = xr.open_dataset(f"{self.inputPath}/{KrillPredict.bathymetryFilename}")
        sst_ds = xr.open_dataset(f"{self.inputPath}/{KrillPredict.sstFilename}")
        ssh_ds = xr.open_dataset(f"{self.inputPath}/{KrillPredict.sshFilename}")
        
        # Create feature matrix
        n_points = len(lon_grid.flatten())
        n_times = len(time_range)
        X = np.zeros((n_points * n_times, 9))  # year, lon, lat, bathymetry, sst, ssh, ugeo, vgeo, net_vel
        
        # Flatten spatial grids for vectorized operations
        lons_flat = lon_grid.flatten()
        lats_flat = lat_grid.flatten()
        
        # Find nearest bathymetry points
        lat_bath = bathymetry_ds["lat"].data
        lon_bath = bathymetry_ds["lon"].data
        
        # Find nearest SST points
        lat_sst = sst_ds["latitude"].data
        lon_sst = sst_ds["longitude"].data
        time_sst = sst_ds["time"].data

        # Find nearest SSH points
        lat_ssh = ssh_ds["latitude"].data
        lon_ssh = ssh_ds["longitude"].data
        time_ssh = ssh_ds["time"].data

        
        # For each time point
        self.logger.info(f"Creating feature matrix")
        for t_idx, t in enumerate(time_range):
            # Find nearest SST time index
            t_sst_idx = np.abs(time_sst - np.datetime64(t)).argmin()

            # Find nearest SSH time index
            t_ssh_idx = np.abs(time_ssh - np.datetime64(t)).argmin()
            
            # Base index for this time slice
            base_idx = t_idx * n_points
            
            # Create mask for Weddell Sea region
            weddell_mask = ~(((-75 <= lats_flat) & (lats_flat <= -64)) & 
                           ((-60 <= lons_flat) & (lons_flat <= -20)))
            
            
            # Fill in coordinates for non-Weddell Sea points
            X[base_idx:base_idx + n_points, 0] = np.full(n_points, t.year)
            X[base_idx:base_idx + n_points, 1] = lons_flat
            X[base_idx:base_idx + n_points, 2] = lats_flat
            
            # Find nearest bathymetry and SST values
            for i, (lat, lon) in enumerate(zip(lats_flat, lons_flat)):
                if not weddell_mask[i]:
                    X[base_idx + i, 3] = np.nan
                    continue

                # Bathymetry (constant across time)
                lat_idx = np.abs(lat_bath - lat).argmin()
                lon_idx = np.abs(lon_bath - lon).argmin()
                X[base_idx + i, 3] = abs(bathymetry_ds["elevation"][lat_idx, lon_idx].data)
                
                # SST values
                lat_idx = np.abs(lat_sst - lat).argmin()
                lon_idx = np.abs(lon_sst - lon).argmin()
                init_val = sst_ds["analysed_sst"][t_sst_idx, lat_idx, lon_idx].data
                X[base_idx + i, 4] = init_val - 273.15

                #SSH values
                lat_idx = np.abs(lat_ssh - lat).argmin()
                lon_idx = np.abs(lon_ssh - lon).argmin()
                ssh_val = ssh_ds["adt"][t_ssh_idx, lat_idx, lon_idx].data
                ugeo_val = ssh_ds["ugos"][t_ssh_idx, lat_idx, lon_idx].data
                vgeo_val = ssh_ds["vgos"][t_ssh_idx, lat_idx, lon_idx].data
                net_vel_val = np.sqrt(ugeo_val**2 + vgeo_val**2)

                X[base_idx + i, 5] = ssh_val
                X[base_idx + i, 6] = ugeo_val
                X[base_idx + i, 7] = vgeo_val
                X[base_idx + i, 8] = net_vel_val

        self.logger.info(f"Finished creating feature matrix")
        
        # Store valid indices and grid shape for plotting
        self.grid_shape = lon_grid.shape
        self.n_points = n_points
        
        # Convert to DataFrame with feature names matching training data
        self.valid_mask = ~np.isnan(X).any(axis=1)
        X_valid = X[self.valid_mask]
        X_df = pd.DataFrame(X_valid, columns=['YEAR', 'LONGITUDE', 'LATITUDE', 'BATHYMETRY', 'SST', 'SSH', 'UGO', 'VGO', 'NET_VEL'])
        
        # Scale features to match training data
        self.logger.info(f"Scaling features...")
        for col in X_df.columns:
            X_df[col] = (X_df[col] - X_df[col].mean()) / (X_df[col].std() + 0.00001)
        return X_df

    def spatialExtent(self):
        # Create coordinate arrays based on map parameters
        lons = np.arange(self.mapParams['lon_min'], 
                        self.mapParams['lon_max'] + self.mapParams['lon_step'], 
                        self.mapParams['lon_step'])
        lats = np.arange(self.mapParams['lat_min'], 
                        self.mapParams['lat_max'] + self.mapParams['lat_step'], 
                        self.mapParams['lat_step'])
        
        # Create meshgrid for all combinations of coordinates
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        
        return lon_grid, lat_grid

    def temporalExtent(self):
        # Convert time_step string to pandas frequency string
        freq_map = {
            'daily': 'D',
            'weekly': 'W',
            'monthly': 'M'
        }
        # .get is a dictionary method that looks up lowercase time_step key and returns "D" as default value if not found
        freq = freq_map.get(self.mapParams['time_step'].lower(), 'D')
        
        # Create datetime range
        time_range = pd.date_range(start=self.mapParams['time_min'],
                                 end=self.mapParams['time_max'],
                                 freq=freq)
        
        return time_range

    def predict(self):
        # Predict only on valid points
        valid_predictions = self.model.predict(self.X)
        
        # Create full prediction array with NaN for invalid points
        y = np.full(len(self.valid_mask), np.nan)
        y[self.valid_mask] = valid_predictions
        return y

    def plotPredictions(self, time_idx=0, save_path=None):
        """Plot predictions on a map"""
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        import cmocean
        
        # Get spatial extent
        lon_grid, lat_grid = self.spatialExtent()
        
        # Get predictions for this time slice
        y_time = self.y[time_idx * self.n_points:(time_idx + 1) * self.n_points]
        
        # Transform from log10 back to original space
        y_time = np.power(10, y_time)
        
        # Reshape to grid
        pred_grid = y_time.reshape(self.grid_shape)
        
        # Create figure with map projection
        fig = plt.figure(figsize=(12, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.gridlines(draw_labels=True)
        ax.coastlines()
        
        # Create levels for contour plot
        #levels = np.linspace(0,np.nanmean(y_time)*1.5, 40)
        levels = np.linspace(0, 2.5, 40)
        
        # Plot bathymetry first
        bathymetry_ds = xr.open_dataset(f"{self.inputPath}/{KrillPredict.bathymetryFilename}")
        
        # Subset bathymetry data to match prediction grid extent
        bath_subset = bathymetry_ds.sel(
            lon=slice(lon_grid.min(), lon_grid.max()),
            lat=slice(lat_grid.min(), lat_grid.max())
        )
        
        bath_data = abs(bath_subset["elevation"].data)
        lon_bath = bath_subset["lon"].data
        lat_bath = bath_subset["lat"].data
        
        # Create proper meshgrid for bathymetry
        lon_bath_mesh, lat_bath_mesh = np.meshgrid(lon_bath, lat_bath)
        
        # Plot bathymetry
        bath_levels = np.linspace(bath_data.min(), bath_data.max(), 10)
        bath_mesh = ax.contour(lon_bath_mesh, lat_bath_mesh, bath_data,
                                transform=ccrs.PlateCarree(), colors='gray',
                                alpha=0.42, levels=bath_levels, linewidths=0.65)
        #plt.colorbar(bath_mesh, label='Depth (m)', shrink=0.75)
        
        # Plot predictions with pcolormesh
        mesh = ax.pcolormesh(lon_grid, lat_grid, pred_grid,
                           transform=ccrs.PlateCarree(),
                           cmap='Reds', vmin=min(levels), vmax=max(levels))
        plt.colorbar(mesh, label='Krill Density', shrink=0.75)
        
        # Add land on top of contours
        ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=100)
        
        # Set map extent with some padding
        ax.set_extent([
            self.mapParams['lon_min'] - 0.25,
            self.mapParams['lon_max'] + 0.25,
            self.mapParams['lat_min'] - 0.25,
            self.mapParams['lat_max'] + 0.25
        ], crs=ccrs.PlateCarree())
        
        # Add title with time information
        time_range = self.temporalExtent()
        plt.title(f'Krill Prediction for {time_range[time_idx].strftime("%Y-%m-%d")} ({self.scenario})')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        else:
            plt.show()
        
        plt.close()
        return

class ResponseCurves:
    # logger
    logging.basicConfig(level=logging.INFO)
    loggerDescription = "\nResponseCurves class description:\n\
        generates response curves for each feature while keeping others at their median values\n\
        calculates uncertainty bands using multiple predictions with sampled feature values\n\
        plots all response curves with standard deviation bands\n"

    # Define feature columns as class attribute
    # feature_cols = ["YEAR", "LONGITUDE", "LATITUDE", "BATHYMETRY", "SST", \
    #                "SSH", "UGO", "VGO", "NET_VEL", "CHL", "FE", "OXY"]
    feature_cols = ["BATHYMETRY", "SST", "FE", "SSH", "NET_VEL", "CHL", "YEAR", "LONGITUDE", "LATITUDE"]
    # Define display names for features
    display_names = {
        "BATHYMETRY": "DEPTH",
        # Add any other display name mappings here if needed
    }

    def __init__(self, inputPath, modelType='rfr', n_points=100, n_samples=50):
        """
        Initialize ResponseCurves class to generate and plot model response curves.
        """
        self.inputPath = inputPath
        self.modelType = modelType
        self.n_points = n_points
        self.n_samples = n_samples
        
        # Load model directly using joblib
        self.model = load(f"{inputPath}/{modelType}Model.joblib")
        
        # Load the fused data and remove any rows with NaN values
        self.fusedData = pd.read_csv(f"{inputPath}/{KrillPredict.fusedFilename}")
        self.fusedData = self.fusedData.dropna(subset=self.feature_cols)
        
        if len(self.fusedData) == 0:
            raise ValueError("No valid data remains after removing NaN values")
        
        # Initialize logger
        self.initLogger()
        
        # Calculate statistics for each feature
        self.feature_stats = {}
        self.logger.info("\nFeature statistics used for predictions:")
        for col in self.feature_cols:
            col_data = self.fusedData[col].dropna()
            self.feature_stats[col] = {
                'median': col_data.median(),
                'std': col_data.std(),
                'min': col_data.min(),
                'max': col_data.max()
            }
            self.logger.info(f"\n{col}:")
            self.logger.info(f"  Median: {self.feature_stats[col]['median']:.3f}")
            self.logger.info(f"  Std Dev: {self.feature_stats[col]['std']:.3f}")
            self.logger.info(f"  Range: [{self.feature_stats[col]['min']:.3f}, {self.feature_stats[col]['max']:.3f}]")

    def initLogger(self):
        """Initialize the logger for the ResponseCurves class."""
        self.logger = logging.getLogger("ResponseCurves")
        self.logger.setLevel(logging.INFO)
        
        # Add class description to logger
        self.logger.info(self.loggerDescription)
        
        # Log initialization parameters
        self.logger.info("\nInitialization parameters:")
        self.logger.info(f"Input path: {self.inputPath}")
        self.logger.info(f"Model type: {self.modelType}")
        self.logger.info(f"Number of evaluation points: {self.n_points}")
        self.logger.info(f"Number of samples for uncertainty: {self.n_samples}")

    def generate_response_curve(self, feature_name):
        """
        Generate response curve for a specific feature while keeping others at median values.
        
        Args:
            feature_name (str): Name of the feature to vary
            
        Returns:
            tuple: (x_values, mean_predictions, std_devs) where x_values are the feature values,
                  mean_predictions are the mean predictions across samples, and std_devs are the 
                  standard deviations of predictions at each point
        """
        if feature_name not in self.feature_cols:
            raise ValueError(f"Feature {feature_name} not found in training data")
        
        # Drop any NaN values from the feature data before calculating range
        feature_data = self.fusedData[feature_name].dropna()
        if len(feature_data) == 0:
            raise ValueError(f"No valid data points found for feature {feature_name}")
            
        # Create array of values to evaluate for the target feature
        x_values = np.linspace(
            feature_data.min(),
            feature_data.max(),
            self.n_points
        )
        
        # Initialize arrays to store predictions
        all_predictions = np.zeros((self.n_samples, self.n_points))
        
        # Generate multiple predictions with different samples of other features
        for i in range(self.n_samples):
            # Create base prediction matrix
            X_pred = pd.DataFrame(index=range(self.n_points), columns=self.feature_cols)
            
            # For each feature, either use the target values or sample from a normal distribution
            for col in self.feature_cols:
                col_data = self.fusedData[col].dropna()
                if len(col_data) == 0:
                    raise ValueError(f"No valid data points found for feature {col}")
                    
                if col == feature_name:
                    X_pred[col] = x_values
                else:
                    # Sample from normal distribution around median with feature's std
                    sampled_values = np.random.normal(
                        loc=col_data.median(),
                        scale=col_data.std() * 0.1,
                        size=self.n_points
                    )
                    # Clip to min/max range
                    sampled_values = np.clip(
                        sampled_values,
                        col_data.min(),
                        col_data.max()
                    )
                    X_pred[col] = sampled_values
            
            # Get predictions based on model type
            if self.modelType == 'rfr':
                # Random Forest has built-in uncertainty estimation
                X_pred_array = X_pred.values
                tree_predictions = np.array([tree.predict(X_pred_array) for tree in self.model.estimators_])
                if np.isnan(tree_predictions).any():
                    raise ValueError(f"NaN values in tree predictions at sample {i}")
                all_predictions[i] = np.mean(tree_predictions, axis=0)
            else:
                # For other models (like GBR), just use the model's predict method
                predictions = self.model.predict(X_pred)
                if np.isnan(predictions).any():
                    raise ValueError(f"NaN values in predictions at sample {i}")
                all_predictions[i] = predictions
        
        # Calculate mean and std across samples
        mean_predictions = np.mean(all_predictions, axis=0)
        std_devs = np.std(all_predictions, axis=0)
        
        return x_values, mean_predictions, std_devs

    def plot_all_response_curves(self, save_path=None):
        """
        Create a single figure with subplots for all features showing response curves
        with standard deviation bands.
        
        Args:
            save_path (str, optional): Path to save the plot. If None, displays the plot.
        """
        # Calculate number of rows and columns for subplots
        n_features = len(self.feature_cols)
        n_cols = 3  # You can adjust this
        n_rows = (n_features + n_cols - 1) // n_cols
        
        # Create figure and subplots with increased width
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 4*n_rows))  # Increased from 20 to 24
        
        # Flatten axes array for easier iteration
        axes_flat = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
        
        # Generate and plot response curves for each feature
        for idx, feature in enumerate(self.feature_cols):
            ax = axes_flat[idx]
            x_values, mean_predictions, std_devs = self.generate_response_curve(feature)
            
            # Plot mean prediction
            ax.plot(x_values, mean_predictions, 'b-', label='Mean prediction', linewidth=2)
            
            # Plot standard deviation band if available
            if std_devs is not None:
                ax.fill_between(x_values, 
                              mean_predictions - std_devs,
                              mean_predictions + std_devs,
                              color='blue', alpha=0.2,
                              label='±1 std dev')
            
            # Use display name if available
            display_name = self.display_names.get(feature, feature)
            
            # Set labels and title with increased font size
            ax.set_xlabel(display_name, fontsize=20)
            ax.set_ylabel('Predicted Krill Density', fontsize=20)
            ax.grid(True)
            
            # Increase tick label font size
            ax.tick_params(axis='both', which='major', labelsize=20)
            
            # Add legend
            ax.legend(fontsize=16, loc='upper right')
            
            # Add subplot label (a, b, c, etc.)
            ax.text(-0.1, 1.1, f'({chr(97+idx)})', transform=ax.transAxes, 
                   fontsize=24, fontweight='bold')
        
        # Remove any empty subplots
        for idx in range(len(self.feature_cols), len(axes_flat)):
            fig.delaxes(axes_flat[idx])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()