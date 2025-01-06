import logging
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os

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
    bathymetrySaveFig = "bathymetryVerification.png"

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
        self.fuseBathymetry()
        self.fuseSST()
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
            self.krillData.loc[idx, "BATHYMETRY"] = elevation[latIdx, lonIdx]
        
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
        DownloadSST()
        breakpoint()
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

class DownloadSST:
    def __init__(self):
        pass