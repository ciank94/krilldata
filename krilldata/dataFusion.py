import logging
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import cmocean
import copernicusmarine as cop
import cartopy.crs as ccrs
import cartopy.feature as cfeature

class DataFusion:
    # logger
    logging.basicConfig(level=logging.INFO)
    loggerDescription = "\nDataFusion class description:\n\
        reads fileDataSubset from ReadKrillBase class\n\
        preprocesses bathymetry data from GEBCO\n\
        preprocesses SST data from CMEMS\n\
        preprocesses SSH data from Copernicus\n\
        fuses data\n"

    fusedFilename = "krillFusedData.csv"
    bathymetrySaveFig = "bathymetryVerification.png"
    sstSaveFig = "sstVerification.png"
    fusedSaveFig = "fusedDistributions.png"
    bioticSaveFig = "bioticDistributions.png"
    environmentalSaveFig = "environmentalData.png"
    doImputation = True

    def __init__(self, krillData, inputPath, outputPath):
        # Instance variables
        self.krillData = krillData
        self.inputPath = inputPath
        self.outputPath = outputPath
        self.bathymetryPath = f"{inputPath}/bathymetry.nc"
        self.sstPath = f"{inputPath}/sst.nc"
        self.sshPath = f"{inputPath}/ssh.nc"
        self.chlPath = f"{inputPath}/chl.nc"
        self.ironPath = f"{inputPath}/iron.nc"
        self.oxyPath = f"{inputPath}/oxygen.nc"

        # Main methods
        self.initLogger()
        #self.fuseData()
        self.fuseDynamic()
        return

    def initLogger(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"================={self.__class__.__name__}=====================")
        self.logger.info(f"{DataFusion.loggerDescription}")
        return

    def fuseData(self):
        if os.path.exists(os.path.join(self.inputPath, DataFusion.fusedFilename)):
            self.logger.info(f"File already exists: {DataFusion.fusedFilename}")
        else:
            self.fuseYear()
            self.fuseBathymetry()
            self.fuseSST()
            self.fuseSSH()
            self.fuseCHL()
            self.fuseIRON()
            self.fuseOXY()
            if DataFusion.doImputation:
                self.logger.info("Imputing missing values...")
                self.imputeMissingValues()
            else:
                self.logger.info("Not imputing missing values...")
            self.fuseSave()

        self.krillData = pd.read_csv(os.path.join(self.inputPath, DataFusion.fusedFilename))
        self.checkPlot()
        return

    def fuseDynamic(self):
        sstDataset = xr.open_dataset(self.sstPath)
        sshDataset = xr.open_dataset(self.sshPath)
        chlDataset = xr.open_dataset(self.chlPath)
        ironDataset = xr.open_dataset(self.ironPath)
        lat = np.array(self.krillData['LATITUDE'])
        lon = np.array(self.krillData['LONGITUDE'])
        
        time_slice = slice(f'-01-01', f'-03-31')
        sst = sstDataset.analysed_sst.sel(time=sstDataset.time.dt.month.isin([1,2,3]))
        breakpoint()
        yearly_mean = sst.groupby("time.year").mean(dim="time", skipna=True).compute()

        # calculate time:
        start_time = time.time()
        sstDataset.analysed_sst.mean(dim='time', skipna=True).compute()
        end_time = time.time()
        print(f"Time: {end_time - start_time}")
        for i in range(len(lat)):
            sst = sstDataset.analysed_sst.sel(latitude=lat[i], longitude=lon[i], method='nearest')
            sstmean = sst.mean(dim='time', skipna=True).compute()
            ssh = sshDataset.adt.sel(latitude=lat[i], longitude=lon[i], method='nearest')
            sshmean = ssh.mean(dim='time', skipna=True).compute()
            chl = chlDataset.CHL.sel(latitude=lat[i], longitude=lon[i], method='nearest')
            chlmean = chl.mean(dim='time', skipna=True).compute()
            iron = ironDataset.fe.sel(latitude=lat[i], longitude=lon[i], method='nearest')
            ironmean = iron.mean(dim='time', skipna=True).compute()
            self.logger.info(f"{i}")
        breakpoint()

        return

    def fuseYear(self):
        self.logger.info(f"Adding year column...")
        self.krillData["YEAR"] = self.krillData["DATE"].dt.year
        self.logger.info(f"Finished adding year column")
        return

    def fuseBathymetry(self):
        bathymetryDataset = xr.open_dataset(self.bathymetryPath)
        latBath = bathymetryDataset["lat"].data
        lonBath = bathymetryDataset["lon"].data
        elevation = bathymetryDataset["elevation"].data
        latKrill = self.krillData.LATITUDE
        lonKrill = self.krillData.LONGITUDE
        latNearest, lonNearest = self.findNearestPoints(latBath, lonBath, latKrill, lonKrill)
        for idx, (latIdx, lonIdx) in enumerate(zip(latNearest, lonNearest)):
            self.krillData.loc[idx, "BATHYMETRY"] = abs(elevation[latIdx, lonIdx])
        bathymetryDataset.close()
        self.logger.info(f"Finished fusing bathymetry data, closing dataset")
        self.logger.info(f"{self.krillData.head()}")
        return

    def fuseSST(self):
        self.logger.info(f"Checking for SST data...")
        if not os.path.exists(self.sstPath):
            self.logger.warning(f"SST data does not exist, must be downloaded: {self.sstPath}")
            raise FileNotFoundError(f"File does not exist: {self.sstPath}")
        else:
            self.logger.info(f"File already exists: {self.sstPath}")

        sstDataset = xr.open_dataset(self.sstPath)
        lonSST = sstDataset["longitude"].data
        latSST = sstDataset["latitude"].data
        timeSST = sstDataset["time"].data
        latKrill = self.krillData.LATITUDE
        lonKrill = self.krillData.LONGITUDE
        timeKrill = self.krillData.DATE

        # Split data into pre-Oct 1982 and post-Oct 1982
        cutoff_date = np.datetime64('1982-10-01')
        pre_1982_mask = timeKrill < cutoff_date
        post_1982_mask = ~pre_1982_mask

        # Initialize SST column with NaN
        self.krillData["SST"] = np.nan

        # Process post-1982 data
        if np.any(post_1982_mask):
            latKrill_post = latKrill[post_1982_mask]
            lonKrill_post = lonKrill[post_1982_mask]
            timeKrill_post = timeKrill[post_1982_mask]
            
            latNearest, lonNearest = self.findNearestPoints(latSST, lonSST, latKrill_post, lonKrill_post)
            timeIndices = self.findNearestTime(timeSST, timeKrill_post)
            
            post_1982_indices = self.krillData.index[post_1982_mask]
            
            self.logger.info(f"Processing post-Oct 1982 SST data")
            for i, (idx, latIdx, lonIdx, timeIdx) in enumerate(zip(post_1982_indices, latNearest, lonNearest, timeIndices)):
                init_val = sstDataset["analysed_sst"][timeIdx, latIdx, lonIdx].data
                self.krillData.loc[idx, "SST"] = init_val - 273.15  # Convert to Celsius
            sstDataset.close()
            self.logger.info(f"Finished processing post-Oct 1982 SST data, closing dataset")

        # Impute pre-1982 data using mean values per location
        if np.any(pre_1982_mask):
            self.logger.info(f"Imputing pre-Oct 1982 SST data")
            pre_1982_indices = self.krillData.index[pre_1982_mask]
            
            # For each pre-1982 point, find the mean of post-1982 values within a spatial window
            for idx in pre_1982_indices:
                lat = self.krillData.loc[idx, "LATITUDE"]
                lon = self.krillData.loc[idx, "LONGITUDE"]

                # Define spatial window (±2 degrees)
                lat_window = 2.0
                lon_window = 2.0
                
                # Find post-1982 points within the window
                spatial_mask = (
                    (self.krillData.LATITUDE >= lat - lat_window) &
                    (self.krillData.LATITUDE <= lat + lat_window) &
                    (self.krillData.LONGITUDE >= lon - lon_window) &
                    (self.krillData.LONGITUDE <= lon + lon_window) &
                    post_1982_mask
                )
                
                if np.any(spatial_mask):
                    # Impute using mean values from the spatial window
                    self.krillData.loc[idx, "SST"] = self.krillData.loc[spatial_mask, "SST"].mean(skipna=True)
                else:
                    # If no points in window, use global means
                    self.krillData.loc[idx, "SST"] = self.krillData.loc[post_1982_mask, "SST"].mean(skipna=True)

        self.logger.info(f"Finished fusing SST data")
        self.logger.info(f"{self.krillData.head()}")
        return

    def fuseSSH(self):
        if not os.path.exists(self.sshPath):
           self.logger.warning(f"SSH data does not exist, must be downloaded: {self.sshPath}")
           raise FileNotFoundError(f"File does not exist: {self.sshPath}")
        else:
            self.logger.info(f"File already exists: {self.sshPath}")
        sshDataset = xr.open_dataset(self.sshPath)
        lonSSH = sshDataset["longitude"].data
        latSSH = sshDataset["latitude"].data
        timeSSH = sshDataset["time"].data
        latKrill = self.krillData.LATITUDE
        lonKrill = self.krillData.LONGITUDE
        timeKrill = self.krillData.DATE

        # Split data into pre-1993 and post-1993
        cutoff_date = np.datetime64('1993-01-01')
        pre_1993_mask = timeKrill < cutoff_date
        post_1993_mask = ~pre_1993_mask

        # Initialize columns with NaN
        self.krillData["SSH"] = np.nan
        self.krillData["UGO"] = np.nan
        self.krillData["VGO"] = np.nan
        self.krillData["NET_VEL"] = np.nan

        # Process post-1993 data
        if np.any(post_1993_mask):
            latKrill_post = latKrill[post_1993_mask]
            lonKrill_post = lonKrill[post_1993_mask]
            timeKrill_post = timeKrill[post_1993_mask]
            
            latNearest, lonNearest = self.findNearestPoints(latSSH, lonSSH, latKrill_post, lonKrill_post)
            timeIndices = self.findNearestTime(timeSSH, timeKrill_post)
            
            post_1993_indices = self.krillData.index[post_1993_mask]
            
            self.logger.info(f"Processing post-1993 SSH data")
            for i, (idx, latIdx, lonIdx, timeIdx) in enumerate(zip(post_1993_indices, latNearest, lonNearest, timeIndices)):
                ugos_val = sshDataset["ugos"][timeIdx, latIdx, lonIdx]
                vgos_val = sshDataset["vgos"][timeIdx, latIdx, lonIdx]
                adt_val = sshDataset["adt"][timeIdx, latIdx, lonIdx]
                self.krillData.loc[idx, "SSH"] = adt_val.data
                self.krillData.loc[idx, "UGO"] = ugos_val.data
                self.krillData.loc[idx, "VGO"] = vgos_val.data
                self.krillData.loc[idx, "NET_VEL"] = np.sqrt(ugos_val.data ** 2 + vgos_val.data ** 2)
            sshDataset.close()
            self.logger.info(f"Finished processing post-1993 SSH data, closing dataset")

        # Impute pre-1993 data using mean values per location
        if np.any(pre_1993_mask):
            self.logger.info(f"Imputing pre-1993 SSH data")
            pre_1993_indices = self.krillData.index[pre_1993_mask]
            
            # For each pre-1993 point, find the mean of post-1993 values within a spatial window
            for idx in pre_1993_indices:
                lat = self.krillData.loc[idx, "LATITUDE"]
                lon = self.krillData.loc[idx, "LONGITUDE"]
                
                # Define spatial window (±2 degrees)
                lat_window = 2.0
                lon_window = 2.0
                
                # Find post-1993 points within the window
                spatial_mask = (
                    (self.krillData.LATITUDE >= lat - lat_window) &
                    (self.krillData.LATITUDE <= lat + lat_window) &
                    (self.krillData.LONGITUDE >= lon - lon_window) &
                    (self.krillData.LONGITUDE <= lon + lon_window) &
                    post_1993_mask
                )
                
                if np.any(spatial_mask):
                    # Impute using mean values from the spatial window
                    self.krillData.loc[idx, "SSH"] = self.krillData.loc[spatial_mask, "SSH"].mean(skipna=True)
                    self.krillData.loc[idx, "UGO"] = self.krillData.loc[spatial_mask, "UGO"].mean(skipna=True)
                    self.krillData.loc[idx, "VGO"] = self.krillData.loc[spatial_mask, "VGO"].mean(skipna=True)
                    self.krillData.loc[idx, "NET_VEL"] = self.krillData.loc[spatial_mask, "NET_VEL"].mean(skipna=True)
                else:
                    # If no points in window, use global means
                    self.krillData.loc[idx, "SSH"] = self.krillData.loc[post_1993_mask, "SSH"].mean(skipna=True)
                    self.krillData.loc[idx, "UGO"] = self.krillData.loc[post_1993_mask, "UGO"].mean(skipna=True)
                    self.krillData.loc[idx, "VGO"] = self.krillData.loc[post_1993_mask, "VGO"].mean(skipna=True)
                    self.krillData.loc[idx, "NET_VEL"] = self.krillData.loc[post_1993_mask, "NET_VEL"].mean(skipna=True)

        self.logger.info(f"Finished fusing SSH data")
        self.logger.info(f"{self.krillData.head()}")
        return

    def fuseCHL(self):
        self.logger.info(f"Adding CHL column...")
        self.logger.info(f"Fuse after September 1997...")
        if not os.path.exists(self.chlPath):
           self.logger.warning(f"CHL data does not exist, must be downloaded: {self.chlPath}")
           raise FileNotFoundError(f"File does not exist: {self.chlPath}")
        else:
            self.logger.info(f"File already exists: {self.chlPath}")
        chlDataset = xr.open_dataset(self.chlPath)
        lonCHL = chlDataset["longitude"].data
        latCHL = chlDataset["latitude"].data
        timeCHL = chlDataset["time"].data
        latKrill = self.krillData.LATITUDE
        lonKrill = self.krillData.LONGITUDE
        timeKrill = self.krillData.DATE

        # Split data into pre-Sept 1997 and post-Sept 1997
        cutoff_date = np.datetime64('1997-09-01')
        pre_1997_mask = timeKrill < cutoff_date
        post_1997_mask = ~pre_1997_mask
        if np.any(post_1997_mask):
            latKrill_post = latKrill[post_1997_mask]
            lonKrill_post = lonKrill[post_1997_mask]
            timeKrill_post = timeKrill[post_1997_mask]
            
            latNearest, lonNearest = self.findNearestPoints(latCHL, lonCHL, latKrill_post, lonKrill_post)
            timeIndices = self.findNearestTime(timeCHL, timeKrill_post)
            
            post_1997_indices = self.krillData.index[post_1997_mask]
            self.logger.info(f"Processing post-1997 CHL data")
            for i, (idx, latIdx, lonIdx, timeIdx) in enumerate(zip(post_1997_indices, latNearest, lonNearest, timeIndices)):
                chl = chlDataset["CHL"][timeIdx, latIdx, lonIdx]
                self.krillData.loc[idx, "CHL"] = chl.data

            chlDataset.close()
            self.logger.info(f"Finished processing post-1997 CHL data, closing dataset")
            
        # Impute pre-Sept 1997 data using mean values per location
        if np.any(pre_1997_mask):
            self.logger.info(f"Imputing pre-1997 CHL data")
            pre_1997_indices = self.krillData.index[pre_1997_mask]
            
            # For each pre-1997 point, find the mean of post-1997 values within a spatial window
            for idx in pre_1997_indices:
                lat = self.krillData.loc[idx, "LATITUDE"]
                lon = self.krillData.loc[idx, "LONGITUDE"]

                # Define spatial window (±2 degrees)
                lat_window = 2.0
                lon_window = 2.0
                
                # Find post-1997 points within the window
                spatial_mask = (
                    (self.krillData.LATITUDE >= lat - lat_window) &
                    (self.krillData.LATITUDE <= lat + lat_window) &
                    (self.krillData.LONGITUDE >= lon - lon_window) &
                    (self.krillData.LONGITUDE <= lon + lon_window) &
                    post_1997_mask
                )
                
                if np.any(spatial_mask):
                    # Impute using mean values from the spatial window
                    self.krillData.loc[idx, "CHL"] = self.krillData.loc[spatial_mask, "CHL"].mean(skipna=True)
                else:
                    # If no points in window, use global means
                    self.krillData.loc[idx, "CHL"] = self.krillData.loc[post_1997_mask, "CHL"].mean(skipna=True)

        self.logger.info(f"Finished adding CHL column")
        return

    def fuseIRON(self):
        self.logger.info(f"Adding IRON column...")
        self.logger.info(f"Fuse after January 1993...")
        if not os.path.exists(self.ironPath):
           self.logger.warning(f"IRON data does not exist, must be downloaded: {self.ironPath}")
           raise FileNotFoundError(f"File does not exist: {self.ironPath}")
        else:
            self.logger.info(f"File already exists: {self.ironPath}")
        ironDataset = xr.open_dataset(self.ironPath)
        lonIRON = ironDataset["longitude"].data
        latIRON = ironDataset["latitude"].data
        timeIRON = ironDataset["time"].data
        latKrill = self.krillData.LATITUDE
        lonKrill = self.krillData.LONGITUDE
        timeKrill = self.krillData.DATE

        # Split data into pre-Jan 1993 and post-Jan 1993
        cutoff_date = np.datetime64('1993-01-01')
        pre_1993_mask = timeKrill < cutoff_date
        post_1993_mask = ~pre_1993_mask
        if np.any(post_1993_mask):
            latKrill_post = latKrill[post_1993_mask]
            lonKrill_post = lonKrill[post_1993_mask]
            timeKrill_post = timeKrill[post_1993_mask]
            
            latNearest, lonNearest = self.findNearestPoints(latIRON, lonIRON, latKrill_post, lonKrill_post)
            timeIndices = self.findNearestTime(timeIRON, timeKrill_post)
            
            post_1993_indices = self.krillData.index[post_1993_mask]
            self.logger.info(f"Processing post-1993 IRON data")
            for i, (idx, latIdx, lonIdx, timeIdx) in enumerate(zip(post_1993_indices, latNearest, lonNearest, timeIndices)):
                iron = ironDataset["fe"][timeIdx, 0, latIdx, lonIdx] #depth = 0 for first index
                self.krillData.loc[idx, "FE"] = iron.data
            ironDataset.close()
            self.logger.info(f"Finished processing post-1993 IRON data, closing dataset")
            
        # Impute pre-Jan 1993 data using mean values per location
        if np.any(pre_1993_mask):
            self.logger.info(f"Imputing pre-1993 IRON data")
            pre_1993_indices = self.krillData.index[pre_1993_mask]

            # Get pre-1993 coordinates
            lat = self.krillData.loc[pre_1993_indices, "LATITUDE"]
            lon = self.krillData.loc[pre_1993_indices, "LONGITUDE"]

            # Define spatial window (±2 degrees)
            lat_window = 2.0
            lon_window = 2.0

            # Get post-1993 data and create a mapping of filtered indices to original indices
            post_1993_data = self.krillData[post_1993_mask].copy()
            post_1993_data['original_index'] = post_1993_data.index
            post_1993_data = post_1993_data.reset_index(drop=True)
            
            # Initialize dictionary to store masks for each pre-1993 point
            spatial_masks = {}
            
            # For each pre-1993 point, find the mean of post-1993 values within a spatial window
            self.logger.info(f"Creating masks for each pre-1993 point")
            for lon_val, lat_val, idx in zip(lon, lat, pre_1993_indices):
    
                # Find post-1993 points within the window using filtered dataset
                spatial_mask = (
                    (post_1993_data.LATITUDE >= lat_val - lat_window) &
                    (post_1993_data.LATITUDE <= lat_val + lat_window) &
                    (post_1993_data.LONGITUDE >= lon_val - lon_window) &
                    (post_1993_data.LONGITUDE <= lon_val + lon_window)
                )
                spatial_masks[idx] = spatial_mask
            
            # Calculate global mean for points with no neighbors
            global_mean = self.krillData.loc[post_1993_mask, "FE"].mean(skipna=True)
            
            # Now assign all values at once
            self.logger.info("Assigning imputed values...")
            for idx, mask in spatial_masks.items():
                if np.any(mask):
                    matching_indices = post_1993_data[mask]['original_index']
                    self.krillData.loc[idx, "FE"] = self.krillData.loc[matching_indices, "FE"].mean(skipna=True)
                else:
                    self.krillData.loc[idx, "FE"] = global_mean
        self.logger.info(f"Finished adding FE column")
        return

    def fuseOXY(self):
        self.logger.info(f"Adding OXY column...")
        self.logger.info(f"Fuse after January 1993...")
        if not os.path.exists(self.oxyPath):
           self.logger.warning(f"OXY data does not exist, must be downloaded: {self.oxyPath}")
           raise FileNotFoundError(f"File does not exist: {self.oxyPath}")
        else:
            self.logger.info(f"File already exists: {self.oxyPath}")
        oxyDataset = xr.open_dataset(self.oxyPath)
        lonOXY = oxyDataset["longitude"].data
        latOXY = oxyDataset["latitude"].data
        timeOXY = oxyDataset["time"].data
        latKrill = self.krillData.LATITUDE
        lonKrill = self.krillData.LONGITUDE
        timeKrill = self.krillData.DATE

        # Split data into pre-Jan 1993 and post-Jan 1993
        cutoff_date = np.datetime64('1993-01-01')
        pre_1993_mask = timeKrill < cutoff_date
        post_1993_mask = ~pre_1993_mask
        if np.any(post_1993_mask):
            latKrill_post = latKrill[post_1993_mask]
            lonKrill_post = lonKrill[post_1993_mask]
            timeKrill_post = timeKrill[post_1993_mask]
            
            latNearest, lonNearest = self.findNearestPoints(latOXY, lonOXY, latKrill_post, lonKrill_post)
            timeIndices = self.findNearestTime(timeOXY, timeKrill_post)
            
            post_1993_indices = self.krillData.index[post_1993_mask]
            self.logger.info(f"Processing post-1993 OXY data")
            for i, (idx, latIdx, lonIdx, timeIdx) in enumerate(zip(post_1993_indices, latNearest, lonNearest, timeIndices)):
                oxy = oxyDataset["o2"][timeIdx, 0, latIdx, lonIdx] #depth = 0 for first index
                self.krillData.loc[idx, "OXY"] = oxy.data

        # Impute pre-Jan 1993 data using mean values per location
        if np.any(pre_1993_mask):
            self.logger.info(f"Imputing pre-1993 OXY data")
            pre_1993_indices = self.krillData.index[pre_1993_mask]
            
            # For each pre-1993 point, find the mean of post-1993 values within a spatial window
            for idx in pre_1993_indices:
                lat = self.krillData.loc[idx, "LATITUDE"]
                lon = self.krillData.loc[idx, "LONGITUDE"]
                
                # Define spatial window (±2 degrees)
                lat_window = 2.0
                lon_window = 2.0
                
                # Find post-1993 points within the window
                spatial_mask = (
                    (self.krillData.LATITUDE >= lat - lat_window) &
                    (self.krillData.LATITUDE <= lat + lat_window) &
                    (self.krillData.LONGITUDE >= lon - lon_window) &
                    (self.krillData.LONGITUDE <= lon + lon_window) &
                    post_1993_mask
                )
                
                if np.any(spatial_mask):
                    # Impute using mean values from the spatial window
                    self.krillData.loc[idx, "OXY"] = self.krillData.loc[spatial_mask, "OXY"].mean(skipna=True)
                else:
                    # If no points in window, use global means
                    self.krillData.loc[idx, "OXY"] = self.krillData.loc[post_1993_mask, "OXY"].mean(skipna=True)

        self.logger.info(f"Finished adding OXY column")
        return
   

    def imputeMissingValues(self):
        """
        Impute missing values in all columns using the median of each column
        """
        self.logger.info("Imputing missing values for non-target columns...")
        
        # Get numerical columns (excluding date and string columns)
        numerical_cols = self.krillData.select_dtypes(include=['float64', 'int64']).columns
        numerical_cols = [col for col in numerical_cols if col != "STANDARDISED_KRILL_UNDER_1M2"]
         
        # Log number of missing values before imputation
        for col in numerical_cols:
            missing_count = self.krillData[col].isna().sum()
            if missing_count > 0:
                self.logger.info(f"Column {col}: {missing_count} missing values")
        
        # Impute missing values with median for each column
        medians = {col: self.krillData[col].median() for col in numerical_cols}
        self.krillData = self.krillData.fillna(medians)
        
        # Verify no missing values remain
        remaining_missing = self.krillData[numerical_cols].isna().sum().sum()
        if remaining_missing == 0:
            self.logger.info("Successfully imputed all missing values")
        else:
            self.logger.warning(f"There are still {remaining_missing} missing values after imputation")
        
        return

    def fuseSave(self):
        self.krillData.to_csv(os.path.join(self.inputPath, DataFusion.fusedFilename), index=False)
        self.logger.info(f"Saved fused data to: {DataFusion.fusedFilename}")
        return

    def checkPlot(self):
        # Plot SST at krill locations
        if os.path.exists(os.path.join(self.outputPath, DataFusion.sstSaveFig)):
            self.logger.info(f"File already exists: {DataFusion.sstSaveFig}")
        else:
            self.logger.info(f"File does not exist: {DataFusion.sstSaveFig}")
            self.logger.info(f"File will be created: {DataFusion.sstSaveFig}")
            bathymetryDataset = xr.open_dataset(self.bathymetryPath)
            self.plotSST(bathymetryDataset)
            bathymetryDataset.close()

        # Plot bathymetry at krill locations
        if os.path.exists(os.path.join(self.outputPath, DataFusion.bathymetrySaveFig)):
            self.logger.info(f"File already exists: {DataFusion.bathymetrySaveFig}")
        else:
            self.logger.info(f"File does not exist: {DataFusion.bathymetrySaveFig}")
            self.logger.info(f"File will be created: {DataFusion.bathymetrySaveFig}")
            bathymetryDataset = xr.open_dataset(self.bathymetryPath)
            self.plotBathymetry(bathymetryDataset)
            bathymetryDataset.close()

        # Plot fused data
        if os.path.exists(os.path.join(self.outputPath, DataFusion.fusedSaveFig)):
            self.logger.info(f"File already exists: {DataFusion.fusedSaveFig}")
        else:
            self.logger.info(f"File does not exist: {DataFusion.fusedSaveFig}")
            self.logger.info(f"File will be created: {DataFusion.fusedSaveFig}")
            self.fusePlot()

        if os.path.exists(os.path.join(self.outputPath, DataFusion.bioticSaveFig)):
            self.logger.info(f"File already exists: {DataFusion.bioticSaveFig}")
        else:
            self.logger.info(f"File does not exist: {DataFusion.bioticSaveFig}")
            self.logger.info(f"File will be created: {DataFusion.bioticSaveFig}")
            self.bioticPlot()

        if os.path.exists(os.path.join(self.outputPath, DataFusion.environmentalSaveFig)):
            self.logger.info(f"File already exists: {DataFusion.environmentalSaveFig}")
        else:
            self.logger.info(f"File does not exist: {DataFusion.environmentalSaveFig}")
            self.logger.info(f"File will be created: {DataFusion.environmentalSaveFig}")
            bathymetryDataset = xr.open_dataset(self.bathymetryPath)
            self.plotEnvironmentalData(bathymetryDataset)
            bathymetryDataset.close()
        return

    def findNearestPoints(self, latGrid, lonGrid, latPoints, lonPoints):
        """
        Find nearest lat/lon pairs in a grid for given points using direct distance calculation
        """
        # Print debug information
        self.logger.info(f"Latitude grid range: {latGrid.min():.2f} to {latGrid.max():.2f}")
        self.logger.info(f"Points latitude range: {latPoints.min():.2f} to {latPoints.max():.2f}")
        
        latNearest = np.zeros_like(latPoints, dtype=int)
        lonNearest = np.zeros_like(lonPoints, dtype=int)
        
        # Iterate through each point
        for i, (lat, lon) in enumerate(zip(latPoints, lonPoints)):
            # Find nearest latitude
            lat_distances = np.abs(latGrid - lat)
            lat_idx = np.argmin(lat_distances)
            # Clip latitude index to valid range
            lat_idx = min(max(0, lat_idx), len(latGrid) - 1)
            latNearest[i] = lat_idx
            
            # Find nearest longitude
            lon_distances = np.abs(lonGrid - lon)
            lon_idx = np.argmin(lon_distances)
            # Clip longitude index to valid range
            lon_idx = min(max(0, lon_idx), len(lonGrid) - 1)
            lonNearest[i] = lon_idx
            
            # Debug output for first few points
            if i < 5:
                self.logger.info(f"Point {i}: lat={lat:.2f}, closest lat={latGrid[lat_idx]:.2f} at index {lat_idx}")
                self.logger.info(f"Point {i}: lon={lon:.2f}, closest lon={lonGrid[lon_idx]:.2f} at index {lon_idx}")
        
        return latNearest, lonNearest

    def findNearestTime(self, timeSST, timeKrill):
        """Find nearest time in SST dataset for each krill observation.
        
        Args:
            timeSST: Array of datetime values from SST dataset
            timeKrill: Array of datetime values from krill observations
            
        Returns:
            Array of indices of nearest SST times for each krill observation
        """
        # Convert pandas Timestamp to numpy datetime64
        timeKrill_np = np.array([np.datetime64(t) for t in timeKrill])
        timeSST_np = np.array([np.datetime64(t) for t in timeSST])
        
        # Initialize array to store indices
        timeIndices = np.zeros(len(timeKrill), dtype=int)
        
        for i in range(len(timeKrill)):
            timeDiff = np.abs(timeSST_np - timeKrill_np[i])
            timeIndices[i] = np.argmin(timeDiff)
            
        return timeIndices

    def plotBathymetry(self, bathymetryDataset):
        """Create a figure showing bathymetry data and krill locations"""
        self.logger.info("Plotting bathymetry data...")
        
        # Create masked array for bathymetry where elevation <= 0
        bathymetry = abs(bathymetryDataset.elevation.values)
        masked_bathymetry = np.ma.masked_where(bathymetry < 0, bathymetry)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot bathymetry with ocean-focused colormap
        im = ax.pcolormesh(bathymetryDataset.lon, bathymetryDataset.lat, 
                          masked_bathymetry, shading='auto', 
                          cmap='Blues', vmin=0)  # Blues_r gives darker blues for deeper water
        im.set_clim(0, None)
        im.cmap.set_bad('grey')
        
        # Plot krill locations
        scatter = ax.scatter(self.krillData.LONGITUDE, self.krillData.LATITUDE, 
                           c=self.krillData.BATHYMETRY, cmap='Blues', 
                           s=10, edgecolor='black', linewidth=0.5)
        
        plt.colorbar(im, ax=ax, label='Ocean Depth (m)')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        
        plt.tight_layout()
        plotName = DataFusion.bathymetrySaveFig
        plt.savefig(os.path.join(self.outputPath, plotName), dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Saved bathymetry plot to: {plotName}")
        return

    def plotSST(self, bathymetryDataset):
        """Create a figure showing bathymetry data and krill locations"""
        self.logger.info("Plotting sst data...")
        
        # Create masked array for bathymetry where elevation <= 0
        bathymetry = abs(bathymetryDataset.elevation.values)
        # Mask both land (elevation > 0) and invalid points
        masked_bathymetry = np.ma.masked_where((bathymetry <= 0) | (bathymetry > 10000), bathymetry)
        
        # Create figure with extra width to accommodate colorbars
        fig, ax = plt.subplots(figsize=(11, 8))
        
        # Plot bathymetry with ocean-focused colormap
        im = ax.pcolormesh(bathymetryDataset.lon, bathymetryDataset.lat, 
                          masked_bathymetry, shading='auto', 
                          cmap='Blues')  # Blues_r gives darker blues for deeper water

        im.set_clim(0, 5000)  # Set bathymetry limits between 0-5000m
        im.cmap.set_bad('lightgrey')  # Set land points to grey
        
        # Plot krill locations
        scatter = ax.scatter(self.krillData.LONGITUDE, self.krillData.LATITUDE, 
                           c=self.krillData.SST, cmap='hot', 
                           s=10, edgecolor='black', linewidth=0.5)
        
        # Add colorbars on opposite sides with consistent size
        cbar1 = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar2 = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.08)
        
        # Set colorbar labels with increased font size
        cbar1.set_label('Ocean Depth (m)', fontsize=12)
        cbar2.set_label('SST (°C)', fontsize=12)
        cbar1.ax.tick_params(labelsize=11)
        cbar2.ax.tick_params(labelsize=11)
        
        # Increase font sizes for axis labels and ticks
        ax.set_xlabel('Longitude', fontsize=14)
        ax.set_ylabel('Latitude', fontsize=14)
        ax.tick_params(axis='both', labelsize=12)
        
        plt.tight_layout()
        
        plotName = DataFusion.sstSaveFig
        plt.savefig(os.path.join(self.outputPath, plotName), dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Saved sst plot to: {plotName}")
        return

    def fusePlot(self):
        """Create a figure with histograms showing distributions of bathymetry and SST"""
        # Define colors
        barColor = '#7FB3D5'  # Soft blue
        lineColor = '#E74C3C'  # Bright red
        barAlpha = 0.7  # Transparency for bars
        lineWidth = 2.5  # Thicker line for better visibility
        
        plt.rcParams.update({'font.size': 12})
        fig = plt.figure(figsize=(12, 10))
        gs = plt.GridSpec(2, 2)
        
        # Bathymetry histogram
        ax1 = fig.add_subplot(gs[0, 0])
        n1, bins1, _ = ax1.hist(self.krillData.BATHYMETRY, bins=30, color=barColor, 
                               edgecolor='white', alpha=barAlpha, linewidth=1)
        ax1Twin = ax1.twinx()
        ax1Twin.plot(bins1[:-1], np.cumsum(n1)/np.sum(n1)*100, color=lineColor, linewidth=lineWidth)
        ax1.set_xlabel('Depth (m)', fontsize=14)
        ax1.set_ylabel('Count', fontsize=14)
        ax1Twin.set_ylabel('Cumulative %', fontsize=14, color=lineColor)
        ax1.tick_params(axis='both', which='major', labelsize=12)
        ax1Twin.tick_params(axis='y', colors=lineColor, labelsize=12)
        ax1Twin.set_ylim(0, 105)
        
        # SST histogram
        ax2 = fig.add_subplot(gs[0, 1])
        n2, bins2, _ = ax2.hist(self.krillData.SST, bins=30, color=barColor, 
                               edgecolor='white', alpha=barAlpha, linewidth=1)
        ax2Twin = ax2.twinx()
        ax2Twin.plot(bins2[:-1], np.cumsum(n2)/np.sum(n2)*100, color=lineColor, linewidth=lineWidth)
        ax2.set_xlabel('Sea Surface Temperature (°C)', fontsize=14)
        ax2.set_ylabel('Count', fontsize=14)
        ax2Twin.set_ylabel('Cumulative %', fontsize=14, color=lineColor)
        ax2.tick_params(axis='both', which='major', labelsize=12)
        ax2Twin.tick_params(axis='y', colors=lineColor, labelsize=12)
        ax2Twin.set_ylim(0, 105)
        
        # SSH histogram
        ax3 = fig.add_subplot(gs[1, 0])
        n3, bins3, _ = ax3.hist(self.krillData.SSH, bins=30, color=barColor, 
                               edgecolor='white', alpha=barAlpha, linewidth=1)
        ax3Twin = ax3.twinx()
        ax3Twin.plot(bins3[:-1], np.cumsum(n3)/np.sum(n3)*100, color=lineColor, linewidth=lineWidth)
        ax3.set_xlabel('Sea Surface Height (m)', fontsize=14)
        ax3.set_ylabel('Count', fontsize=14)
        ax3Twin.set_ylabel('Cumulative %', fontsize=14, color=lineColor)
        ax3.tick_params(axis='both', which='major', labelsize=12)
        ax3Twin.tick_params(axis='y', colors=lineColor, labelsize=12)
        ax3Twin.set_ylim(0, 105)

        # Velocity magnitude histogram
        ax4 = fig.add_subplot(gs[1, 1])
        velocity_mag = np.sqrt(self.krillData.NET_VEL)
        n4, bins4, _ = ax4.hist(velocity_mag, bins=30, color=barColor, 
                               edgecolor='white', alpha=barAlpha, linewidth=1)
        ax4Twin = ax4.twinx()
        ax4Twin.plot(bins4[:-1], np.cumsum(n4)/np.sum(n4)*100, color=lineColor, linewidth=lineWidth)
        ax4.set_xlabel('Velocity Magnitude (m/s)', fontsize=14)
        ax4.set_ylabel('Count', fontsize=14)
        ax4Twin.set_ylabel('Cumulative %', fontsize=14, color=lineColor)
        ax4.tick_params(axis='both', which='major', labelsize=12)
        ax4Twin.tick_params(axis='y', colors=lineColor, labelsize=12)
        ax4Twin.set_ylim(0, 105)
        
        plt.tight_layout()        
        # Save figure with high DPI
        plotName = DataFusion.fusedSaveFig
        fig.savefig(os.path.join(self.outputPath, plotName), dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Saved fused distributions plot to: {plotName}")
        return

    def bioticPlot(self):
        """Create figure with histograms showing CHL, Iron, Oxygen"""
        # Define colors
        barColor = '#7FB3D5'  # Soft blue
        lineColor = '#E74C3C'  # Bright red
        barAlpha = 0.7  # Transparency for bars
        lineWidth = 2.5  # Thicker line for better visibility
        
        plt.rcParams.update({'font.size': 12})
        fig = plt.figure(figsize=(18, 6))  # Wider figure for three columns
        gs = plt.GridSpec(1, 3)  # One row, three columns
        
        # Chl histogram
        ax1 = fig.add_subplot(gs[0, 0])
        n1, bins1, _ = ax1.hist(self.krillData.CHL, bins=30, color=barColor, 
                               edgecolor='white', alpha=barAlpha, linewidth=1)
        ax1Twin = ax1.twinx()
        ax1Twin.plot(bins1[:-1], np.cumsum(n1)/np.sum(n1)*100, color=lineColor, linewidth=lineWidth)
        ax1.set_xlabel('Chlorophyll (mg/m3)', fontsize=14)
        ax1.set_ylabel('Count', fontsize=14)
        ax1Twin.set_ylabel('Cumulative %', fontsize=14, color=lineColor)
        ax1.tick_params(axis='both', which='major', labelsize=12)
        ax1Twin.tick_params(axis='y', colors=lineColor, labelsize=12)
        ax1Twin.set_ylim(0, 105)
        
        # Iron histogram
        ax2 = fig.add_subplot(gs[0, 1])
        n2, bins2, _ = ax2.hist(self.krillData.FE, bins=30, color=barColor, 
                               edgecolor='white', alpha=barAlpha, linewidth=1)
        ax2Twin = ax2.twinx()
        ax2Twin.plot(bins2[:-1], np.cumsum(n2)/np.sum(n2)*100, color=lineColor, linewidth=lineWidth)
        ax2.set_xlabel('Iron (mmol/m3)', fontsize=14)
        ax2.set_ylabel('Count', fontsize=14)
        ax2Twin.set_ylabel('Cumulative %', fontsize=14, color=lineColor)
        ax2.tick_params(axis='both', which='major', labelsize=12)
        ax2Twin.tick_params(axis='y', colors=lineColor, labelsize=12)
        ax2Twin.set_ylim(0, 105)
        
        # Oxygen histogram
        ax3 = fig.add_subplot(gs[0, 2])  # Changed from [1, 0] to [0, 2]
        n3, bins3, _ = ax3.hist(self.krillData.OXY, bins=30, color=barColor, 
                               edgecolor='white', alpha=barAlpha, linewidth=1)
        ax3Twin = ax3.twinx()
        ax3Twin.plot(bins3[:-1], np.cumsum(n3)/np.sum(n3)*100, color=lineColor, linewidth=lineWidth)
        ax3.set_xlabel('Oxygen (mmol/m3)', fontsize=14)
        ax3.set_ylabel('Count', fontsize=14)
        ax3Twin.set_ylabel('Cumulative %', fontsize=14, color=lineColor)
        ax3.tick_params(axis='both', which='major', labelsize=12)
        ax3Twin.tick_params(axis='y', colors=lineColor, labelsize=12)
        ax3Twin.set_ylim(0, 105)

        plt.tight_layout()        
        # Save figure with high DPI
        plotName = DataFusion.bioticSaveFig
        fig.savefig(os.path.join(self.outputPath, plotName), dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Saved biotic distributions plot to: {plotName}")
        return

    def plotEnvironmentalData(self, bathymetryDataset):
        """Create a figure showing bathymetry with krill locations and environmental variables"""
        self.logger.info("Plotting environmental data...")
        
        # Convert DATE to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(self.krillData.DATE):
            self.krillData['DATE'] = pd.to_datetime(self.krillData['DATE'])
        
        # Filter data for January 2016
        jan_2016_mask = (self.krillData.DATE.dt.year == 2016) & (self.krillData.DATE.dt.month == 1)
        
        # Load SST data
        sstDataset = xr.open_dataset(self.sstPath)
        # Find January 2016 in SST data
        sst_time_idx = np.where((sstDataset.time.dt.year == 2016) & (sstDataset.time.dt.month == 1))[0][0]
        sst_data = sstDataset["analysed_sst"][sst_time_idx].values - 273.15  # Convert to Celsius
        
        # Load SSH data
        sshDataset = xr.open_dataset(self.sshPath)
        # Find January 2016 in SSH data
        ssh_time_idx = np.where((sshDataset.time.dt.year == 2016) & (sshDataset.time.dt.month == 1))[0][0]
        ssh_data = sshDataset["adt"][ssh_time_idx].values
        net_vel_data = np.sqrt(sshDataset["ugos"][ssh_time_idx].values**2 + 
                              sshDataset["vgos"][ssh_time_idx].values**2)
        
        # Load CHL data
        chlDataset = xr.open_dataset(self.chlPath)
        chl_time_idx = np.where((chlDataset.time.dt.year == 2016) & (chlDataset.time.dt.month == 1))[0][0]
        chl_data = chlDataset["CHL"][chl_time_idx].values
        
        # Load Iron data
        ironDataset = xr.open_dataset(self.ironPath)
        iron_time_idx = np.where((ironDataset.time.dt.year == 2016) & (ironDataset.time.dt.month == 1))[0][0]
        iron_data = ironDataset["fe"][iron_time_idx, 0].values  # Select the first depth level (depth=0)
        
        # Create masked array for bathymetry where elevation <= 0
        bathymetry = abs(bathymetryDataset.elevation.values)
        # Mask both land (elevation > 0) and invalid points
        masked_bathymetry = np.ma.masked_where((bathymetry <= 0) | (bathymetry > 10000), bathymetry)
        
        # Create contour levels every 500m
        contour_levels = np.arange(0, 3000, 400)
        
        # Create figure with 3x2 subplots
        plt.rcParams.update({'font.size': 20})  # Set default font size to 20
        fig = plt.figure(figsize=(28, 24))  # Increased height for better spacing
        gs = fig.add_gridspec(3, 2, hspace=0.001, wspace=0.25)  # Increased hspace from 0.005 to 0.4
        
        projection = ccrs.PlateCarree()
        
        # Plot 1: Bathymetry with krill locations
        ax1 = plt.subplot(gs[0, 0], projection=projection)
        im1 = ax1.pcolormesh(bathymetryDataset.lon, bathymetryDataset.lat, 
                          masked_bathymetry, shading='auto', 
                          cmap='Blues', transform=projection, zorder=1)
        im1.set_clim(0, 5000)
        ax1.add_feature(cfeature.LAND, facecolor='lightgrey', zorder=100)  # Increased zorder
        ax1.coastlines(zorder=101)  # Coastlines on top
        ax1.scatter(self.krillData.LONGITUDE, 
                   self.krillData.LATITUDE,
                   c='red', s=10, edgecolor='black', linewidth=0.3,  # Reduced size
                   transform=projection, zorder=102)  # Points on top of everything
        cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.025, pad=0.04)
        cbar1.set_label('Depth (m)', fontsize=20)
        cbar1.ax.tick_params(labelsize=20)
        ax1.set_xlabel('Longitude', fontsize=20)
        ax1.set_ylabel('Latitude', fontsize=20)
        ax1.tick_params(axis='both', labelsize=20)
        cs1 = ax1.contour(bathymetryDataset.lon, bathymetryDataset.lat, 
                       masked_bathymetry, levels=contour_levels,
                       colors='grey', alpha=0.3, transform=projection)
        self.logger.info("Completed bathymetry subplot (1/6)")

        # Plot 2: SST
        ax2 = plt.subplot(gs[0, 1], projection=projection)
        im2 = ax2.pcolormesh(sstDataset.longitude, sstDataset.latitude, 
                          sst_data, shading='auto',
                          cmap=cmocean.cm.thermal, transform=projection)
        ax2.add_feature(cfeature.LAND, facecolor='lightgrey', zorder=100)
        ax2.coastlines(zorder=101)
        cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.025, pad=0.04)
        cbar2.set_label('Temperature (°C)', fontsize=20)
        cbar2.ax.tick_params(labelsize=20)
        ax2.set_xlabel('Longitude', fontsize=20)
        ax2.set_ylabel('Latitude', fontsize=20)
        ax2.tick_params(axis='both', labelsize=20)
        cs2 = ax2.contour(bathymetryDataset.lon, bathymetryDataset.lat, 
                       masked_bathymetry, levels=contour_levels,
                       colors='grey', alpha=0.3, transform=projection)
        self.logger.info("Completed SST subplot (2/6)")

        # Plot 3: SSH
        ax3 = plt.subplot(gs[1, 0], projection=projection)
        im3 = ax3.pcolormesh(sshDataset.longitude, sshDataset.latitude, 
                          ssh_data, shading='auto',
                          cmap=cmocean.cm.balance, transform=projection)
        ax3.add_feature(cfeature.LAND, facecolor='lightgrey', zorder=100)
        ax3.coastlines(zorder=101)
        cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.025, pad=0.04)
        cbar3.set_label('Height (m)', fontsize=20)
        cbar3.ax.tick_params(labelsize=20)
        ax3.set_xlabel('Longitude', fontsize=20)
        ax3.set_ylabel('Latitude', fontsize=20)
        ax3.tick_params(axis='both', labelsize=20)
        cs3 = ax3.contour(bathymetryDataset.lon, bathymetryDataset.lat, 
                       masked_bathymetry, levels=contour_levels,
                       colors='grey', alpha=0.3, transform=projection)
        self.logger.info("Completed SSH subplot (3/6)")

        # Plot 4: Net velocity
        ax4 = plt.subplot(gs[1, 1], projection=projection)
        im4 = ax4.pcolormesh(sshDataset.longitude, sshDataset.latitude, 
                          net_vel_data, shading='auto',
                          cmap=cmocean.cm.speed, transform=projection)
        ax4.add_feature(cfeature.LAND, facecolor='lightgrey', zorder=100)
        ax4.coastlines(zorder=101)
        cbar4 = plt.colorbar(im4, ax=ax4, fraction=0.025, pad=0.04)
        cbar4.set_label('Velocity (m/s)', fontsize=20)
        cbar4.ax.tick_params(labelsize=20)
        ax4.set_xlabel('Longitude', fontsize=20)
        ax4.set_ylabel('Latitude', fontsize=20)
        ax4.tick_params(axis='both', labelsize=20)
        cs4 = ax4.contour(bathymetryDataset.lon, bathymetryDataset.lat, 
                       masked_bathymetry, levels=contour_levels,
                       colors='grey', alpha=0.3, transform=projection)
        self.logger.info("Completed velocity subplot (4/6)")

        # Plot 5: CHL
        ax5 = plt.subplot(gs[2, 0], projection=projection)
        im5 = ax5.pcolormesh(chlDataset.longitude, chlDataset.latitude, 
                          chl_data, shading='auto',
                          cmap=cmocean.cm.algae, transform=projection, zorder=1)
        im5.set_clim(0, 2)  # Set max to 2 for CHL
        ax5.add_feature(cfeature.LAND, facecolor='lightgrey', zorder=100)
        ax5.coastlines(zorder=101)
        cbar5 = plt.colorbar(im5, ax=ax5, fraction=0.025, pad=0.04)
        cbar5.set_label('Chlorophyll (mg/m$^3$)', fontsize=20)
        cbar5.ax.tick_params(labelsize=20)
        ax5.set_xlabel('Longitude', fontsize=20)
        ax5.set_ylabel('Latitude', fontsize=20)
        ax5.tick_params(axis='both', labelsize=20)
        cs5 = ax5.contour(bathymetryDataset.lon, bathymetryDataset.lat, 
                       masked_bathymetry, levels=contour_levels,
                       colors='grey', alpha=0.3, transform=projection)
        self.logger.info("Completed chlorophyll subplot (5/6)")

        # Plot 6: Iron
        ax6 = plt.subplot(gs[2, 1], projection=projection)
        im6 = ax6.pcolormesh(ironDataset.longitude, ironDataset.latitude, 
                          iron_data, shading='auto',
                          cmap=cmocean.cm.matter, transform=projection, zorder=1)
        im6.set_clim(0, 0.001)  # Set color limits on the plot object
        ax6.add_feature(cfeature.LAND, facecolor='lightgrey', zorder=100)
        ax6.coastlines(zorder=101)
        cbar6 = plt.colorbar(im6, ax=ax6, fraction=0.025, pad=0.04)
        cbar6.set_label('Iron (mmol/m$^3$)', fontsize=20)
        cbar6.ax.tick_params(labelsize=20)
        ax6.set_xlabel('Longitude', fontsize=20)
        ax6.set_ylabel('Latitude', fontsize=20)
        ax6.tick_params(axis='both', labelsize=20)
        cs6 = ax6.contour(bathymetryDataset.lon, bathymetryDataset.lat, 
                       masked_bathymetry, levels=contour_levels,
                       colors='grey', alpha=0.3, transform=projection)
        self.logger.info("Completed iron subplot (6/6)")

        # Set common gridlines for all subplots
        for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
            gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
            gl.top_labels = False
            gl.right_labels = False
            gl.xlabel_style = {'size': 20}
            gl.ylabel_style = {'size': 20}
        
        plotName = DataFusion.environmentalSaveFig
        plt.savefig(os.path.join(self.outputPath, plotName), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Close datasets
        sstDataset.close()
        sshDataset.close()
        chlDataset.close()
        ironDataset.close()
        
        self.logger.info(f"Saved environmental data plot to: {plotName}")
        return
