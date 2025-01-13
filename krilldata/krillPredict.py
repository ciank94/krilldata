from joblib import load
import json
import numpy as np
import pandas as pd
import xarray as xr


#todo: predict
class KrillPredict:
    # filenames:
    bathymetryFilename = "bathymetry.nc"
    sstFilename = "sst.nc"
    fusedFilename = "krillFusedData.csv"

    def __init__(self, inputPath, outputPath, modelType):
        # Instance variables
        self.inputPath = inputPath
        self.outputPath = outputPath
        self.modelType = modelType
        self.model = load(f"{inputPath}/{self.modelType}Model.joblib")
        self.mapParams = self.loadParameters()

        # Main methods
        self.X = self.loadFeatures()
        self.y = self.predict()
        return

    def loadParameters(self):
        #todo: load features
        with open(f"{self.inputPath}/map_params.json", "r") as f:
            map_params = json.load(f)
        return map_params

    def loadFeatures(self):
        # Get spatial and temporal extents
        lon_grid, lat_grid = self.spatialExtent()
        time_range = self.temporalExtent()
        
        # Load bathymetry and SST data
        bathymetry_ds = xr.open_dataset(f"{self.inputPath}/{self.bathymetryFilename}")
        sst_ds = xr.open_dataset(f"{self.inputPath}/{self.sstFilename}")
        
        # Create feature matrix
        n_points = len(lon_grid.flatten())
        n_times = len(time_range)
        X = np.zeros((n_points * n_times, 4))  # lon, lat, bathymetry, sst
        
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
        
        # For each time point
        for t_idx, t in enumerate(time_range):
            # Find nearest SST time index
            t_sst_idx = np.abs(time_sst - np.datetime64(t)).argmin()
            
            # Base index for this time slice
            base_idx = t_idx * n_points
            
            # Fill in coordinates
            X[base_idx:base_idx + n_points, 0] = lons_flat
            X[base_idx:base_idx + n_points, 1] = lats_flat
            
            # Find nearest bathymetry and SST values
            for i, (lat, lon) in enumerate(zip(lats_flat, lons_flat)):
                # Bathymetry (constant across time)
                lat_idx = np.abs(lat_bath - lat).argmin()
                lon_idx = np.abs(lon_bath - lon).argmin()
                X[base_idx + i, 2] = abs(bathymetry_ds["elevation"][lat_idx, lon_idx].data)
                
                # SST values
                lat_idx = np.abs(lat_sst - lat).argmin()
                lon_idx = np.abs(lon_sst - lon).argmin()
                init_val = sst_ds["analysed_sst"][t_sst_idx, lat_idx, lon_idx].data
                X[base_idx + i, 3] = init_val - 273.15

        return X

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
        y = self.model.predict(self.X)
        return y