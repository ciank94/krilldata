from joblib import load
import json
import numpy as np
import pandas as pd
import xarray as xr
import logging

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
        X = np.zeros((n_points * n_times, 8))  # lon, lat, bathymetry, sst, ssh, ugeo, vgeo, net_vel
        
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
            X[base_idx:base_idx + n_points, 0] = lons_flat
            X[base_idx:base_idx + n_points, 1] = lats_flat
            
            # Find nearest bathymetry and SST values
            for i, (lat, lon) in enumerate(zip(lats_flat, lons_flat)):
                if not weddell_mask[i]:
                    X[base_idx + i, 2] = np.nan
                    continue
                    
                # Bathymetry (constant across time)
                lat_idx = np.abs(lat_bath - lat).argmin()
                lon_idx = np.abs(lon_bath - lon).argmin()
                X[base_idx + i, 2] = abs(bathymetry_ds["elevation"][lat_idx, lon_idx].data)
                
                # SST values
                lat_idx = np.abs(lat_sst - lat).argmin()
                lon_idx = np.abs(lon_sst - lon).argmin()
                init_val = sst_ds["analysed_sst"][t_sst_idx, lat_idx, lon_idx].data
                X[base_idx + i, 3] = init_val - 273.15

                #SSH values
                lat_idx = np.abs(lat_ssh - lat).argmin()
                lon_idx = np.abs(lon_ssh - lon).argmin()
                ssh_val = ssh_ds["adt"][t_ssh_idx, lat_idx, lon_idx].data
                ugeo_val = ssh_ds["ugos"][t_ssh_idx, lat_idx, lon_idx].data
                vgeo_val = ssh_ds["vgos"][t_ssh_idx, lat_idx, lon_idx].data
                net_vel_val = np.sqrt(ugeo_val**2 + vgeo_val**2)

                X[base_idx + i, 4] = ssh_val
                X[base_idx + i, 5] = ugeo_val
                X[base_idx + i, 6] = vgeo_val
                X[base_idx + i, 7] = net_vel_val

        self.logger.info(f"Finished creating feature matrix")
        
        # Store valid indices and grid shape for plotting
        self.grid_shape = lon_grid.shape
        self.n_points = n_points
        
        # Convert to DataFrame with feature names matching training data
        self.valid_mask = ~np.isnan(X).any(axis=1)
        X_valid = X[self.valid_mask]
        X_df = pd.DataFrame(X_valid, columns=['LONGITUDE', 'LATITUDE', 'BATHYMETRY', 'SST', 'SSH', 'UGO', 'VGO', 'NET_VEL'])
        
        # Scale features to match training data
        self.logger.info(f"Scaling features...")
        for col in X_df.columns:
            X_df[col] = (X_df[col] - X_df[col].mean()) / X_df[col].std()
        
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
        import matplotlib.pyplot as plt
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
        levels = np.linspace(0, 2, 40)
        
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