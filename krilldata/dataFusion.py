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
        fuses data\n"

    # filenames:
    bathymetryFilename = "bathymetry.nc"
    sstFilename = "sst.nc"
    fusedFilename = "krillFusedData.csv"
    bathymetrySaveFig = "bathymetryVerification.png"
    sstSaveFig = "sstVerification.png"
    fusedSaveFig = "fusedDistributions.png"

    def __init__(self, krillData, inputPath, outputPath):
        # Instance variables
        self.krillData = krillData
        self.inputPath = inputPath
        self.outputPath = outputPath
        self.bathymetryPath = f"{inputPath}/{DataFusion.bathymetryFilename}"
        self.sstPath = f"{inputPath}/{DataFusion.sstFilename}"

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
            self.fuseBathymetry()
            self.fuseSST()
            self.fuseSave()

        self.krillData = pd.read_csv(os.path.join(self.inputPath, DataFusion.fusedFilename))

        if os.path.exists(os.path.join(self.outputPath, DataFusion.fusedSaveFig)):
            self.logger.info(f"File already exists: {DataFusion.fusedSaveFig}")
        else:
            self.logger.info(f"File does not exist: {DataFusion.fusedSaveFig}")
            self.logger.info(f"File will be created: {DataFusion.fusedSaveFig}")
            self.fusePlot()
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
        
        # Plot bathymetry at krill locations
        if os.path.exists(os.path.join(self.outputPath, DataFusion.bathymetrySaveFig)):
            self.logger.info(f"File already exists: {DataFusion.bathymetrySaveFig}")
        else:
            self.logger.info(f"File does not exist: {DataFusion.bathymetrySaveFig}")
            self.logger.info(f"File will be created: {DataFusion.bathymetrySaveFig}")
            self.plotBathymetry(bathymetryDataset)
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
        latNearest, lonNearest = self.findNearestPoints(latSST, lonSST, latKrill, lonKrill)
        timeIndices = self.findNearestTime(timeSST, timeKrill)
        self.logger.info(f"Finding nearest SST time for each krill observation")
        for idx, (latIdx, lonIdx, timeIdx) in enumerate(zip(latNearest, lonNearest, timeIndices)):
             init_val = sstDataset["analysed_sst"][timeIdx, latIdx, lonIdx]
             self.krillData.loc[idx, "SST"] = init_val.data - 273.15  # Convert Kelvin to Celsius
             #self.logger.info(f"SST value at index {idx} is {init_val.data}")

        
        self.logger.info(f"Finished fusing SST data")
        self.logger.info(f"{self.krillData.head()}")
        if os.path.exists(os.path.join(self.outputPath, DataFusion.sstSaveFig)):
            self.logger.info(f"File already exists: {DataFusion.sstSaveFig}")
        else:
            self.logger.info(f"File does not exist: {DataFusion.sstSaveFig}")
            self.logger.info(f"File will be created: {DataFusion.sstSaveFig}")
            bathymetryDataset = xr.open_dataset(self.bathymetryPath)
            self.plotSST(bathymetryDataset)
        return

    def fuseSave(self):
        self.krillData.to_csv(os.path.join(self.inputPath, DataFusion.fusedFilename), index=False)
        self.logger.info(f"Saved fused data to: {DataFusion.fusedFilename}")
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
        bathymetry = bathymetryDataset.elevation.values
        masked_bathymetry = np.ma.masked_where(bathymetry > 0, bathymetry)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot bathymetry with ocean-focused colormap
        im = ax.pcolormesh(bathymetryDataset.lon, bathymetryDataset.lat, 
                          masked_bathymetry, shading='auto', 
                          cmap='Blues_r')  # Blues_r gives darker blues for deeper water
        
        # Plot krill locations
        scatter = ax.scatter(self.krillData.LONGITUDE, self.krillData.LATITUDE, 
                           c=self.krillData.BATHYMETRY, cmap='Blues_r', 
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
        bathymetry = bathymetryDataset.elevation.values
        masked_bathymetry = np.ma.masked_where(bathymetry > 0, bathymetry)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot bathymetry with ocean-focused colormap
        im = ax.pcolormesh(bathymetryDataset.lon, bathymetryDataset.lat, 
                          masked_bathymetry, shading='auto', 
                          cmap='Blues_r')  # Blues_r gives darker blues for deeper water
        
        # Plot krill locations
        scatter = ax.scatter(self.krillData.LONGITUDE, self.krillData.LATITUDE, 
                           c=self.krillData.SST, cmap='hot', 
                           s=20, edgecolor='black', linewidth=0.5)
        
        plt.colorbar(im, ax=ax, label='Ocean Depth (m)')
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
        fig = plt.figure(figsize=(12, 5))
        gs = plt.GridSpec(1, 2)
        
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