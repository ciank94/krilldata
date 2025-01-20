import logging
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import copernicusmarine as cop

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
        self.fuseData()
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

        # Plot bathymetry at krill locations
        if os.path.exists(os.path.join(self.outputPath, DataFusion.bathymetrySaveFig)):
            self.logger.info(f"File already exists: {DataFusion.bathymetrySaveFig}")
        else:
            self.logger.info(f"File does not exist: {DataFusion.bathymetrySaveFig}")
            self.logger.info(f"File will be created: {DataFusion.bathymetrySaveFig}")
            self.plotBathymetry(bathymetryDataset)

        # Plot fused data
        if os.path.exists(os.path.join(self.outputPath, DataFusion.fusedSaveFig)):
            self.logger.info(f"File already exists: {DataFusion.fusedSaveFig}")
        else:
            self.logger.info(f"File does not exist: {DataFusion.fusedSaveFig}")
            self.logger.info(f"File will be created: {DataFusion.fusedSaveFig}")
            self.fusePlot()
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
                           s=20, edgecolor='black', linewidth=0.5)
        
        plt.colorbar(im, ax=ax, label='Ocean Depth (m)')
        ax.set_title('Ocean Bathymetry and Krill Locations')
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
        masked_bathymetry = np.ma.masked_where(bathymetry < 0, bathymetry)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot bathymetry with ocean-focused colormap
        im = ax.pcolormesh(bathymetryDataset.lon, bathymetryDataset.lat, 
                          masked_bathymetry, shading='auto', 
                          cmap='Blues')  # Blues_r gives darker blues for deeper water

        im.set_clim(0, None)
        im.cmap.set_bad('grey')
        
        # Plot krill locations
        scatter = ax.scatter(self.krillData.LONGITUDE, self.krillData.LATITUDE, 
                           c=self.krillData.SST, cmap='hot', 
                           s=20, edgecolor='black', linewidth=0.5)
        
        plt.colorbar(scatter, ax=ax, label='SST (°C)')
        ax.set_title('Ocean Bathymetry, sst and Krill Locations')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        
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
