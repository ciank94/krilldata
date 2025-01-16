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

    # filenames:
    bathymetryFilename = "bathymetry.nc"
    sstFilename = "sst.nc"
    sshFilename = "ssh.nc"
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
        self.bathymetryPath = f"{inputPath}/{DataFusion.bathymetryFilename}"
        self.sstPath = f"{inputPath}/{DataFusion.sstFilename}"
        self.sshPath = f"{inputPath}/{DataFusion.sshFilename}"

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
        
        self.logger.info(f"Finished fusing bathymetry data")
        self.logger.info(f"{self.krillData.head()}")
        return

    def fuseSST(self):
        self.logger.info(f"Downloading SST data...")
        if not os.path.exists(self.sstPath):
            self.logger.info(f"File does not exist: {DataFusion.sstFilename}")
            self.logger.info(f"LARGE FILE WILL BE DOWNLOADED: {DataFusion.sstFilename}")
            DownloadSST()
        else:
            self.logger.info(f"File already exists: {DataFusion.sstFilename}")

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
            self.logger.info(f"File does not exist: {DataFusion.sshFilename}")
            self.logger.info(f"LARGE FILE WILL BE DOWNLOADED: {DataFusion.sshFilename}")
            DownloadSSH()
        else:
            self.logger.info(f"File already exists: {DataFusion.sshFilename}")
        
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
        Find nearest lat/lon pairs in a grid for given points
        """
        # Ensure arrays are sorted for searchsorted
        latSorted = np.sort(latGrid)
        lonSorted = np.sort(lonGrid)
        
        # Get the insertion points
        latIdx = np.searchsorted(latSorted, latPoints)
        lonIdx = np.searchsorted(lonSorted, lonPoints)
        
        # Clip to valid range
        latIdx = np.clip(latIdx, 1, len(latGrid)-1)
        lonIdx = np.clip(lonIdx, 1, len(lonGrid)-1)
        
        # Find nearest point by comparing distances
        latPrev = latGrid[latIdx-1]
        latNext = latGrid[latIdx]
        latNearest = np.where(np.abs(latPoints - latPrev) < np.abs(latPoints - latNext), 
                            latIdx-1, latIdx)
        
        lonPrev = lonGrid[lonIdx-1]
        lonNext = lonGrid[lonIdx]
        lonNearest = np.where(np.abs(lonPoints - lonPrev) < np.abs(lonPoints - lonNext), 
                            lonIdx-1, lonIdx)
        
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

class DownloadSST:
    defaultRequest = {
            'dataID': 'METOFFICE-GLO-SST-L4-REP-OBS-SST',
            'configurePath': 'C:/Users/ciank/.copernicusmarine/.copernicusmarine-credentials',
            'outputFilename': 'input/sst.nc',
            'startDate': "1981-10-01T00:00:00",
            'endDate': "2016-10-01T00:00:00",
            'lonBounds': [-70, -31],
            'latBounds': [-73, -50],
            'variables': ['analysed_sst']
        }

    def __init__(self):
        self.dataID = DownloadSST.defaultRequest['dataID']
        self.configurePath = DownloadSST.defaultRequest['configurePath']
        self.outputFilename = DownloadSST.defaultRequest['outputFilename']
        self.startDate = DownloadSST.defaultRequest['startDate']
        self.endDate = DownloadSST.defaultRequest['endDate']
        self.lonBounds = DownloadSST.defaultRequest['lonBounds']
        self.latBounds = DownloadSST.defaultRequest['latBounds']
        self.variables = DownloadSST.defaultRequest['variables']

        # log request details and process request;
        self.initLogger()
        self.downloadSST()
        return

    def initLogger(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"================={self.__class__.__name__}=====================")
        self.logger.info(f"Initializing {self.__class__.__name__}")
        self.logger.info(f"=================Request parameters=====================")
        self.logger.info(f"DataID: {self.dataID}")    
        self.logger.info(f"Configure path: {self.configurePath}")
        self.logger.info(f"Output path: {self.outputFilename}")
        self.logger.info(f"Start date: {self.startDate}")
        self.logger.info(f"End date: {self.endDate}")
        self.logger.info(f"Longitude bounds: {self.lonBounds}")
        self.logger.info(f"Latitude bounds: {self.latBounds}")
        self.logger.info(f"Variables: {self.variables}")   
        return

    def downloadSST(self):
        self.logger.info(f"Downloading SST data...")
        cop.subset(
            dataset_id=self.dataID,
            variables=self.variables,
            start_datetime=self.startDate,
            end_datetime=self.endDate,
            minimum_longitude=self.lonBounds[0],
            maximum_longitude=self.lonBounds[1],
            minimum_latitude=self.latBounds[0],
            maximum_latitude=self.latBounds[1],
            output_filename=self.outputFilename,
            credentials_file=self.configurePath,
            force_download=True,
        )
        return

class DownloadSSH:
    defaultRequest = {
            'dataID': 'cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D',
            'configurePath': 'C:/Users/ciank/.copernicusmarine/.copernicusmarine-credentials',
            'outputFilename': 'input/ssh.nc',
            'startDate': "1992-01-01T00:00:00",
            'endDate': "2016-12-31T00:00:00",
            'lonBounds': [-70, -31],
            'latBounds': [-73, -50],
            'variables': ['adt', 'ugos', 'vgos']
        }

    def __init__(self):
        self.dataID = DownloadSSH.defaultRequest['dataID']
        self.configurePath = DownloadSSH.defaultRequest['configurePath']
        self.outputFilename = DownloadSSH.defaultRequest['outputFilename']
        self.startDate = DownloadSSH.defaultRequest['startDate']
        self.endDate = DownloadSSH.defaultRequest['endDate']
        self.lonBounds = DownloadSSH.defaultRequest['lonBounds']
        self.latBounds = DownloadSSH.defaultRequest['latBounds']
        self.variables = DownloadSSH.defaultRequest['variables']

        # log request details and process request;
        self.initLogger()
        self.downloadSSH()
        return

    def initLogger(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"================={self.__class__.__name__}=====================")
        self.logger.info(f"Initializing {self.__class__.__name__}")
        self.logger.info(f"=================Request parameters=====================")
        self.logger.info(f"DataID: {self.dataID}")    
        self.logger.info(f"Configure path: {self.configurePath}")
        self.logger.info(f"Output path: {self.outputFilename}")
        self.logger.info(f"Start date: {self.startDate}")
        self.logger.info(f"End date: {self.endDate}")
        self.logger.info(f"Longitude bounds: {self.lonBounds}")
        self.logger.info(f"Latitude bounds: {self.latBounds}")
        self.logger.info(f"Variables: {self.variables}")   
        return

    def downloadSSH(self):
        self.logger.info(f"Downloading SSH data...")
        cop.subset(
            dataset_id=self.dataID,
            variables=self.variables,
            start_datetime=self.startDate,
            end_datetime=self.endDate,
            minimum_longitude=self.lonBounds[0],
            maximum_longitude=self.lonBounds[1],
            minimum_latitude=self.latBounds[0],
            maximum_latitude=self.latBounds[1],
            output_filename=self.outputFilename,
            credentials_file=self.configurePath,
            force_download=True,
        )
        return