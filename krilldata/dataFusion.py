import logging
import xarray as xr

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

    def __init__(self, data, inputPath, outputPath):
        # Instance variables
        self.data = data
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

        self.bathymetryDataset = xr.open_dataset(self.bathymetryPath)
        breakpoint()

        return