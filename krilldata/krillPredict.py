from joblib import load
import json
import numpy as np
import pandas as pd
import xarray as xr
import logging
import matplotlib.pyplot as plt
import os
from scipy.stats import gaussian_kde
from scipy.interpolate import RegularGridInterpolator
from sklearn.metrics import mean_squared_error
from krilldata import KrillTrain
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean
    

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
    chlFilename = "chl.nc"
    ironFilename = "iron.nc"
    fusedFilename = "krillFusedData.csv"
    featureColumns = ["BATHYMETRY", "SST", "FE","SSH", "NET_VEL", "CHL", "YEAR", "LONGITUDE", "LATITUDE"]
    targetColumn = "STANDARDISED_KRILL_UNDER_1M2"

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
        with open("config/map_params.json", "r") as f:
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
        chl_ds = xr.open_dataset(f"{self.inputPath}/{KrillPredict.chlFilename}")
        iron_ds = xr.open_dataset(f"{self.inputPath}/{KrillPredict.ironFilename}")
        
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

        # Find nearest CHL points
        lat_chl = chl_ds["latitude"].data
        lon_chl = chl_ds["longitude"].data
        time_chl = chl_ds["time"].data

        # Find nearest iron points
        lat_iron = iron_ds["latitude"].data
        lon_iron = iron_ds["longitude"].data
        time_iron = iron_ds["time"].data

        
        # For each time point
        self.logger.info(f"Creating feature matrix")
        for t_idx, t in enumerate(time_range):
            # Find nearest SST time index
            t_sst_idx = np.abs(time_sst - np.datetime64(t)).argmin()

            # Find nearest SSH time index
            t_ssh_idx = np.abs(time_ssh - np.datetime64(t)).argmin()

            # Find nearest CHL time index
            t_chl_idx = np.abs(time_chl - np.datetime64(t)).argmin()

            # Find nearest iron time index
            t_iron_idx = np.abs(time_iron - np.datetime64(t)).argmin()
            
            # Base index for this time slice
            base_idx = t_idx * n_points
            
            # Create mask for Weddell Sea region
            weddell_mask = ~(((-75 <= lats_flat) & (lats_flat <= -64)) & 
                           ((-60 <= lons_flat) & (lons_flat <= -20)))
            
            
            # Fill in coordinates for non-Weddell Sea points
            # Bathymetry (constant across time)
            X[base_idx:base_idx + n_points, 6] = np.full(n_points, t.year)
            X[base_idx:base_idx + n_points, 7] = lons_flat
            X[base_idx:base_idx + n_points, 8] = lats_flat
            
            # Find nearest bathymetry and SST values
            for i, (lat, lon) in enumerate(zip(lats_flat, lons_flat)):
                if not weddell_mask[i]:
                    X[base_idx + i, 0] = np.nan
                    continue

                # Bathymetry (constant across time)
                lat_idx = np.abs(lat_bath - lat).argmin()
                lon_idx = np.abs(lon_bath - lon).argmin()
                X[base_idx + i, 0] = abs(bathymetry_ds["elevation"][lat_idx, lon_idx].data)
                
                # SST values
                lat_idx = np.abs(lat_sst - lat).argmin()
                lon_idx = np.abs(lon_sst - lon).argmin()
                init_val = sst_ds["analysed_sst"][t_sst_idx, lat_idx, lon_idx].data
                X[base_idx + i, 1] = init_val - 273.15

                #SSH values
                lat_idx = np.abs(lat_ssh - lat).argmin()
                lon_idx = np.abs(lon_ssh - lon).argmin()
                ssh_val = ssh_ds["adt"][t_ssh_idx, lat_idx, lon_idx].data
                ugeo_val = ssh_ds["ugos"][t_ssh_idx, lat_idx, lon_idx].data
                vgeo_val = ssh_ds["vgos"][t_ssh_idx, lat_idx, lon_idx].data
                net_vel_val = np.sqrt(ugeo_val**2 + vgeo_val**2)

                X[base_idx + i, 4] = ssh_val
                X[base_idx + i, 5] = net_vel_val

                # CHL values
                lat_idx = np.abs(lat_chl - lat).argmin()
                lon_idx = np.abs(lon_chl - lon).argmin()
                X[base_idx + i, 2] = chl_ds["CHL"][t_chl_idx, lat_idx, lon_idx].data

                # Iron values
                lat_idx = np.abs(lat_iron - lat).argmin()
                lon_idx = np.abs(lon_iron - lon).argmin()
                X[base_idx + i, 3] = iron_ds["fe"][t_iron_idx, 0, lat_idx, lon_idx].data

        self.logger.info(f"Finished creating feature matrix")
        
        # Store valid indices and grid shape for plotting
        self.grid_shape = lon_grid.shape
        self.n_points = n_points
        
        # Convert to DataFrame with feature names matching training data
        self.valid_mask = ~np.isnan(X).any(axis=1)
        X_valid = X[self.valid_mask]
        X_df = pd.DataFrame(X_valid, columns=KrillPredict.featureColumns)
        
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

    def plotPredictions(self, time_idx=0, save_path="output/"):
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
        time_v = time_range[time_idx].strftime("%Y-%m-%d")
        plt.title(f'Krill Prediction for {time_v} ({self.scenario})')
        save_path = f"{save_path}Map_{time_v}.png"
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        else:
            plt.show()
        
        plt.close()
        return

class MapKrillDensity:
    def __init__(self, inputPath, outputPath, modelType='rfr'):
        self.bathymetryds = xr.open_dataset(f"{inputPath}/bathymetry.nc")
        self.sstds = xr.open_dataset(f"{inputPath}/sst.nc")
        self.sshds = xr.open_dataset(f"{inputPath}/ssh.nc")
        self.chlds = xr.open_dataset(f"{inputPath}/chl.nc")
        self.ironds = xr.open_dataset(f"{inputPath}/iron.nc")
        self.fusedFilename = "krillFusedData.csv"
        self.featureColumns = ["BATHYMETRY", "SST", "FE","SSH", "NET_VEL", "CHL", "YEAR", "LONGITUDE", "LATITUDE"]
        self.targetColumn = "STANDARDISED_KRILL_UNDER_1M2"
        self.inputPath = inputPath
        self.outputPath = outputPath
        self.modelType = modelType
        self.model = load(f"{inputPath}/{self.modelType}Model.joblib")
        with open("config/map_params.json", "r") as f:
            self.mapParams = json.load(f)

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"================={self.__class__.__name__}=====================")
        self.logger.info(f"Opening datasets: for {self.featureColumns} and {self.targetColumn}")
        self.logger.info(f"model type: {self.modelType}")
        return

    def plotExample(self):
        # algorithm
        self.spatialExtent()
        self.loadEnvFeatures(year=2014, region = "AP")
        self.reindexInterpolate()
        self.getFeatures()
        self.predictY()
        self.mapPredictions()
        return

    def plotRegion(self, region):
        # algorithm
        self.spatialExtent()
        y = []
        for year in range(2013, 2016+1):
            self.loadEnvFeatures(year=year, region = region)
            self.reindexInterpolate()
            self.getFeatures()
            self.predictY()
            y.append(self.y)
            self.logger.info(f"Finished year {year}")
        self.y_v = y
        self.mapArea()     
        return

    def spatialExtent(self):
        self.lonSGmin, self.lonSGmax = self.mapParams['southGeorgia']['lon_min'], self.mapParams['southGeorgia']['lon_max']
        self.latSGmin, self.latSGmax = self.mapParams['southGeorgia']['lat_min'], self.mapParams['southGeorgia']['lat_max']

        self.lonPeMin, self.lonPeMax = self.mapParams['peninsula']['lon_min'], self.mapParams['peninsula']['lon_max']
        self.latPeMin, self.latPeMax = self.mapParams['peninsula']['lat_min'], self.mapParams['peninsula']['lat_max']

        self.lonSoMin, self.latSoMin = self.mapParams['southOrkney']['lon_min'], self.mapParams['southOrkney']['lat_min']
        self.lonSoMax, self.latSoMax = self.mapParams['southOrkney']['lon_max'], self.mapParams['southOrkney']['lat_max']
        # Define the spatial extent of the map
        self.lonSG = np.arange(self.mapParams['southGeorgia']['lon_min'], 
                        self.mapParams['southGeorgia']['lon_max'] + self.mapParams['southGeorgia']['lon_step'], 
                        self.mapParams['southGeorgia']['lon_step'])
        self.latSG = np.arange(self.mapParams['southGeorgia']['lat_min'], 
                        self.mapParams['southGeorgia']['lat_max'] + self.mapParams['southGeorgia']['lat_step'], 
                        self.mapParams['southGeorgia']['lat_step'])

        # Define the spatial extent of the map
        self.lonPe = np.arange(self.mapParams['peninsula']['lon_min'], 
                        self.mapParams['peninsula']['lon_max'] + self.mapParams['peninsula']['lon_step'], 
                        self.mapParams['peninsula']['lon_step'])
        self.latPe = np.arange(self.mapParams['peninsula']['lat_min'], 
                        self.mapParams['peninsula']['lat_max'] + self.mapParams['peninsula']['lat_step'], 
                        self.mapParams['peninsula']['lat_step'])

        # Define the spatial extent of the map
        self.lonSo = np.arange(self.mapParams['southOrkney']['lon_min'], 
                        self.mapParams['southOrkney']['lon_max'] + self.mapParams['southOrkney']['lon_step'], 
                        self.mapParams['southOrkney']['lon_step'])
        self.latSo = np.arange(self.mapParams['southOrkney']['lat_min'], 
                        self.mapParams['southOrkney']['lat_max'] + self.mapParams['southOrkney']['lat_step'], 
                        self.mapParams['southOrkney']['lat_step'])
        
        
        # Create meshgrid for all combinations of coordinates
        self.lonSG_grid, self.latSG_grid = np.meshgrid(self.lonSG, self.latSG)
        self.lonPe_grid, self.latPe_grid = np.meshgrid(self.lonPe, self.latPe)
        self.lonSo_grid, self.latSo_grid = np.meshgrid(self.lonSo, self.latSo)
        return

    def loadEnvFeatures(self, year, region):
        # Load the fused data and remove any rows with NaN values
        self.region = region
        self.yearP = year
        time_slice = slice(f'{self.yearP}-01-01', f'{self.yearP}-03-31')
        if region == "AP":
            lat_slice = slice(self.latPeMin, self.latPeMax)
            lon_slice = slice(self.lonPeMin, self.lonPeMax)
        elif region == "SG":
            lat_slice = slice(self.latSGmin, self.latSGmax)
            lon_slice = slice(self.lonSGmin, self.lonSGmax)
        elif region == "SO":
            lat_slice = slice(self.latSoMin, self.latSoMax)
            lon_slice = slice(self.lonSoMin, self.lonSoMax)
        else:
            raise ValueError("Region must be either 'SG' or 'AP'")

        # Load data effectively
        self.sstdsSub = self.sstds.sel(time=time_slice, latitude=lat_slice, longitude=lon_slice)
        self.sshdsSub = self.sshds.sel(time=time_slice, latitude=lat_slice, longitude=lon_slice)
        self.chldsSub = self.chlds.sel(time=time_slice, latitude=lat_slice, longitude=lon_slice)
        self.irondsSub = self.ironds.sel(time=time_slice, latitude=lat_slice, longitude=lon_slice).isel(depth=0)
        self.bathymetrydsSub = self.bathymetryds.sel(lat=lat_slice, lon=lon_slice)

        # For the time period, compute the mean efficiently with dask
        self.sstdsMean = self.sstdsSub.mean(dim='time', skipna=True).compute()
        self.sshdsMean = self.sshdsSub.mean(dim='time', skipna=True).compute()
        self.chldsMean = self.chldsSub.mean(dim='time', skipna=True).compute()
        self.irondsMean = self.irondsSub.mean(dim='time', skipna=True).compute()
        
        return

    def reindexInterpolate(self):
        # here I should clean values required;
        if self.region == "AP":
            lat = self.latPe
            lon = self.lonPe
        elif self.region == "SG":
            lat = self.latSG
            lon = self.lonSG

        if self.region == "SG":
            self.lon = self.lonSG_grid
            self.lat = self.latSG_grid
        elif self.region == "SO":
            self.lon = self.lonSo_grid
            self.lat = self.latSo_grid
        else:
            self.lon = self.lonPe_grid
            self.lat = self.latPe_grid


        interp_fe = RegularGridInterpolator((self.irondsMean["latitude"].values, self.irondsMean["longitude"].values), self.irondsMean["fe"].values, method='linear', bounds_error=False, fill_value=np.nan)
        irondsMean = interp_fe((self.lat, self.lon))
        interp_t = RegularGridInterpolator((self.sstdsMean["latitude"].values, self.sstdsMean["longitude"].values), self.sstdsMean["analysed_sst"].values, method='linear', bounds_error=False, fill_value=np.nan)
        sstdsMean = interp_t((self.lat, self.lon))
        interp_chl = RegularGridInterpolator((self.chldsMean["latitude"].values, self.chldsMean["longitude"].values), self.chldsMean["CHL"].values, method='linear', bounds_error=False, fill_value=np.nan)
        chldsMean = interp_chl((self.lat, self.lon))
        interp_ssh = RegularGridInterpolator((self.sshdsMean["latitude"].values, self.sshdsMean["longitude"].values), self.sshdsMean["adt"].values, method='linear', bounds_error=False, fill_value=np.nan)
        sshdsMean = interp_ssh((self.lat, self.lon))
        # todo: interpolate net_vel after finding values
        interp_net = RegularGridInterpolator((self.sshdsMean["latitude"].values, self.sshdsMean["longitude"].values), np.sqrt(self.sshdsMean["ugos"].values**2 + self.sshdsMean["vgos"].values**2), method='linear', bounds_error=False, fill_value=np.nan)
        net_vel = interp_net((self.lat, self.lon))
        interp_bath = RegularGridInterpolator((self.bathymetryds["lat"].values, self.bathymetryds["lon"].values), self.bathymetryds["elevation"].values, method='linear', bounds_error=False, fill_value=np.nan)
        bathymetryds = interp_bath((self.lat, self.lon))
        #bathymetryds = self.bathymetryds.sel(lat=lat, lon=lon, method='nearest')
        #self.bvals = bathymetryds["elevation"].values
        self.bvals = bathymetryds
        self.bvals[self.bvals > 0] = 0

        # get all the relevant data
        self.bathymetry = abs(self.bvals)
        # self.fe = irondsMean["fe"].values
        # self.sst = sstdsMean["analysed_sst"].values - 273.15
        # self.ssh = sshdsMean["adt"].values
        # self.net_vel = np.sqrt(np.power(sshdsMean["ugos"].values,2) + np.power(sshdsMean["vgos"].values,2))
        # self.chl = chldsMean["CHL"].values
        self.fe = irondsMean
        self.sst = sstdsMean - 273.15
        self.ssh = sshdsMean
        self.net_vel = net_vel
        self.chl = chldsMean
        self.year = self.yearP*np.ones(self.bathymetry.shape)
        return


    def getFeatures(self):
        X = self.bathymetry.flatten(), self.fe.flatten(), self.sst.flatten(), self.ssh.flatten(), self.net_vel.flatten(), self.chl.flatten(), self.year.flatten(), self.lon.flatten(), self.lat.flatten()
        self.X = np.column_stack(X)
        self.X_df = pd.DataFrame(self.X, columns=self.featureColumns)
        for col in self.X_df.columns:
            self.X_df[col] = (self.X_df[col] - self.X_df[col].mean()) / (self.X_df[col].std() + 0.00001)
        return

    def predictY(self):
        self.y = self.model.predict(self.X_df)
        self.y = self.y.reshape(self.bathymetry.shape)
        return

    def mapArea(self, save_path = "output/"):
        # Create levels for contour plot
        #levels = np.linspace(0,np.nanmean(y_time)*1.5, 40)
        levels = np.linspace(0, 1.5, 30)
        
        # Plot bathymetry first
        bath_data = self.bathymetry
        lon_bath = self.lon
        lat_bath = self.lat

        # Create subplots
        fig, axs = plt.subplots(figsize=(20,12), nrows=2, ncols=2, 
        subplot_kw={'projection': ccrs.PlateCarree()}, layout="constrained")

        # Plot bathymetry on each subplot
        f2_size=20
        counter = -1
        for ax in axs.flatten():
            counter+=1
            gl = ax.gridlines(draw_labels=True, alpha=0.4)
            gl.top_labels = False
            gl.right_labels = False
            gl.xlabel_style = {'size': 20, 'color': 'black'}
            gl.ylabel_style = {'size': 20, 'color': 'black'}
            ax.coastlines()
            self.plot_bathymetry(ax, lon_bath, lat_bath, bath_data)
            # Plot predictions with pcolormesh
            y_v = np.power(10, self.y_v[counter])
            mesh = ax.pcolormesh(self.lon, self.lat, y_v,
                           transform=ccrs.PlateCarree(),
                           cmap='Reds', vmin=min(levels), vmax=max(levels))
            cbar = plt.colorbar(mesh, shrink=0.75, pad=0.01)
            cbar.ax.set_ylabel("individuals/m$^{2}$", loc="center", size=f2_size, weight="bold")
            cbar.ax.tick_params(labelsize=f2_size, rotation=0)

            # Add land on top of contours
            ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=100)
            self.logger.info(f"Plot {counter + 1} of 4")
        
        
        save_path = f"{save_path}Map_{self.region}.png"
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        return


    def plot_bathymetry(self, ax, lon_bath, lat_bath, bath_data):
        # Create levels for contour plot
        levels = np.linspace(0, 1.5, 30)
        
        # Plot bathymetry
        bath_levels = np.linspace(bath_data.min(), bath_data.max(), 10)
        bath_mesh = ax.contour(lon_bath, lat_bath, bath_data,
                            transform=ccrs.PlateCarree(), colors='gray',
                            alpha=0.42, levels=bath_levels, linewidths=0.65)
        return bath_mesh

    def mapPredictions(self, save_path="output/"):
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        import cmocean
        # Create figure with map projection
        fig = plt.figure(figsize=(12, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.gridlines(draw_labels=True)
        ax.coastlines()
        
        # Create levels for contour plot
        #levels = np.linspace(0,np.nanmean(y_time)*1.5, 40)
        self.y = np.power(10, self.y)
        levels = np.linspace(0, 1.5, 30)
        
        # Plot bathymetry first
        bath_data = self.bathymetry
        lon_bath = self.lon
        lat_bath = self.lat
        
        # Create proper meshgrid for bathymetry
        #lon_bath_mesh, lat_bath_mesh = np.meshgrid(lon_bath, lat_bath)
        
        # Plot bathymetry
        bath_levels = np.linspace(bath_data.min(), bath_data.max(), 10)
        bath_mesh = ax.contour(lon_bath, lat_bath, bath_data,
                                transform=ccrs.PlateCarree(), colors='gray',
                                alpha=0.42, levels=bath_levels, linewidths=0.65)
        #plt.colorbar(bath_mesh, label='Depth (m)', shrink=0.75)

        
        # Plot predictions with pcolormesh
        mesh = ax.pcolormesh(self.lon, self.lat, self.y,
                           transform=ccrs.PlateCarree(),
                           cmap='Reds', vmin=min(levels), vmax=max(levels))
        plt.colorbar(mesh, label='Krill Density', shrink=0.75)
        
        # Add land on top of contours
        ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=100)
        
        # Set map extent with some padding
        # ax.set_extent([
        #     self.mapParams['lon_min'] - 0.25,
        #     self.mapParams['lon_max'] + 0.25,
        #     self.mapParams['lat_min'] - 0.25,
        #     self.mapParams['lat_max'] + 0.25
        # ], crs=ccrs.PlateCarree())
        
        # Add title with time information
        plt.title(f'Krill Prediction for {self.yearP}')
        save_path = f"{save_path}Map_{self.yearP}.png"
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        else:
            plt.show()

        pass

    

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
        return

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
        Generate response curve for a specific feature while keeping others at their median values.
        
        Args:
            feature_name (str): Name of the feature to vary
            
        Returns:
            tuple: (x_values, predictions) where x_values are the feature values
                  and predictions are the model predictions
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
        
        # Number of Monte Carlo iterations
        n_iterations = 100
        predictions = np.zeros((n_iterations, self.n_points))
        
        # For each iteration, sample from the distribution of other features
        for i in range(n_iterations):
            X_pred = pd.DataFrame(index=range(self.n_points), columns=self.feature_cols)
            
            # For each feature, either use the target values or random sample
            for col in self.feature_cols:
                col_data = self.fusedData[col].dropna()
                if len(col_data) == 0:
                    raise ValueError(f"No valid data points found for feature {col}")
                
                if col == feature_name:
                    X_pred[col] = x_values
                else:
                    # Sample random values from the feature distribution
                    X_pred[col] = np.random.choice(col_data, size=self.n_points)
            
            # Make predictions for this iteration
            predictions[i, :] = self.model.predict(X_pred)
        
        # Calculate mean and standard deviation of predictions
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        # Store results for plotting
        self.response_data = {
            'x_values': x_values,
            'mean': mean_pred,
            'lower': mean_pred - 2*std_pred,  # 95% confidence interval
            'upper': mean_pred + 2*std_pred
        }
        return self.response_data

    def plot_all_response_curves(self, save_path=None):
        """
        Create a single figure with subplots for all features showing response curves.
        
        Args:
            save_path (str, optional): Path to save the plot. If None, displays the plot.
        """
        # Set default font sizes
        plt.rcParams.update({'font.size': 14})
        
        # Create figure with subplots
        n_features = len(self.feature_cols)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 7*n_rows))
        axes = axes.flatten()
        
        # Generate and plot response curves for each feature
        for i, feature in enumerate(self.feature_cols):
            response_data = self.generate_response_curve(feature)
            
            # Get display name if available, otherwise use feature name
            display_name = self.display_names.get(feature, feature)
            
            # Plot response curve
            axes[i].plot(response_data['x_values'], response_data['mean'], 'b-', linewidth=2)
            axes[i].fill_between(response_data['x_values'], response_data['lower'], response_data['upper'], alpha=0.3, color='b')
            axes[i].set_xlabel(display_name, fontsize=16)
            axes[i].set_ylabel('Predicted Krill Density', fontsize=16)
            axes[i].tick_params(axis='both', which='major', labelsize=14)
            axes[i].grid(True, alpha=0.3)
            
            # Rotate x-axis labels if they're too long
            if len(str(max(response_data['x_values']))) > 6:
                axes[i].tick_params(axis='x', rotation=45)
        
        # Remove empty subplots
        for i in range(n_features, len(axes)):
            fig.delaxes(axes[i])
        
        # Adjust layout with more space
        plt.tight_layout(h_pad=0.5, w_pad=0.5)
        
        # Save or show the plot
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()
        return

class PerformancePlot:
    """Class for creating prediction vs observation plots"""
    
    loggerDescription = "\nPerformancePlot class description:\n\
        loads model predictions and observations from fused data\n\
        creates scatter plot of predicted vs observed values\n\
        calculates performance metrics (R², RMSE)\n\
        adds uncertainty bands using density estimation\n"
    
    def __init__(self, inputPath, modelType='rfr'):
        """
        Initialize PerformancePlot class to generate prediction vs observation plots.
        
        Args:
            inputPath (str): Path to the directory containing the model and data
            modelType (str): Type of model to evaluate (default: 'rfr')
        """
        self.inputPath = inputPath
        self.modelType = modelType
        
        # Load model directly using joblib
        self.model = load(f"{inputPath}/{modelType}Model.joblib")
        
        # Load the fused data and remove any rows with NaN values
        self.fusedData = pd.read_csv(f"{inputPath}/{KrillPredict.fusedFilename}")
        self.fusedData = self.fusedData.dropna(subset=KrillTrain.featureColumns + ["STANDARDISED_KRILL_UNDER_1M2"])
        
        if len(self.fusedData) == 0:
            raise ValueError("No valid data remains after removing NaN values")
        
        # Initialize logger
        self.initLogger()
        
        # Prepare data
        self.X = self.fusedData[KrillTrain.featureColumns]
        self.y_true = self.fusedData["STANDARDISED_KRILL_UNDER_1M2"]
        
        # Standardize features
        self.logger.info("Standardizing features...")
        for col in self.X.columns:
            self.X[col] = (self.X[col] - self.X[col].mean()) / (self.X[col].std() + 0.00001)
        
        # Make predictions with standardized features
        self.y_pred = self.model.predict(self.X)
        
        # Calculate performance metrics
        self.r2 = np.corrcoef(self.y_true, self.y_pred)[0,1]**2
        self.rmse = np.sqrt(mean_squared_error(self.y_true, self.y_pred))
        
        self.logger.info(f"\nModel Performance Metrics:")
        self.logger.info(f"  R²: {self.r2:.3f}")
        self.logger.info(f"  RMSE: {self.rmse:.3f}")
    
    def initLogger(self):
        """Initialize the logger for the PerformancePlot class."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"================={self.__class__.__name__}=====================")
        self.logger.info(f"{PerformancePlot.loggerDescription}")
        return
    
    def plot_performance(self, save_path=None):
        """
        Create a scatter plot of predicted vs observed values with density estimation.
        
        Args:
            save_path (str, optional): Path to save the plot. If None, displays the plot.
        """
        # Convert to numpy arrays
        y_true_np = self.y_true.to_numpy()
        y_pred_np = self.y_pred
        
        # Log data statistics
        self.logger.info(f"\nData statistics:")
        self.logger.info(f"  Number of points: {len(y_true_np)}")
        self.logger.info(f"  True range: [{y_true_np.min():.3f}, {y_true_np.max():.3f}]")
        self.logger.info(f"  Predicted range: [{y_pred_np.min():.3f}, {y_pred_np.max():.3f}]")
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Calculate point density for coloring
        xy = np.vstack([y_true_np, y_pred_np])
        z = gaussian_kde(xy)(xy)
        
        # Sort points by density for better visualization
        idx = z.argsort()
        x, y, z = y_true_np[idx], y_pred_np[idx], z[idx]
        
        # Create scatter plot colored by density
        scatter = plt.scatter(x, y, c=z, s=50, alpha=0.5, cmap='inferno')
        plt.colorbar(scatter, label='Point density')
        
        # Add perfect prediction line
        min_val = min(min(y_true_np), min(y_pred_np))
        max_val = max(max(y_true_np), max(y_pred_np))
        
        # Plot 1:1 line and add empty line for stats
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1')
        plt.plot([], [], ' ', label=f'R² = {self.r2:.3f}\nRMSE = {self.rmse:.3f}')
        
        # Add labels and title
        plt.xlabel('Observed log10(Krill density)')
        plt.ylabel('Predicted log10(Krill density)')
        plt.title('Predicted vs Observed Krill Density')
        plt.legend()
        
        # Add grid
        plt.grid(True, alpha=0.3)
        
        # Equal aspect ratio
        plt.axis('equal')
        
        # Save or show the plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved performance plot to: {save_path}")
            plt.close()
        else:
            plt.show()
        return

