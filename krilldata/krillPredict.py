import logging
import pandas as pd


class KrillPredict:
    # logger
    logging.basicConfig(level=logging.INFO)
    loggerDescription = "\nKrillPredict class description:\n\
        reads fusedData from output of DataFusion class\n\
        ML preprocess: feature scaling, train/test split\n\
        ML model: random forest\n\
        ML postprocess: predict\n\
        save trained model\n\
        load trained model\n\
        predict\n"

    fusedDataFilename = "krillFusedData.csv"

    def __init__(self, inputPath, outputPath):
        # Instance variables
        self.inputPath = inputPath
        self.outputPath = outputPath
        self.fusedDataPath = f"{inputPath}/{KrillPredict.fusedDataFilename}"
        self.fusedData = {}
        self.X = None
        self.y = None

        # Main methods
        self.initLogger()
        self.readData()
        self.scaleFeatures()
        self.loadXy()
        breakpoint()

        return

    def initLogger(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"================={self.__class__.__name__}=====================")
        self.logger.info(f"{KrillPredict.loggerDescription}")
        return

    def readData(self):
        data = pd.read_csv(self.fusedDataPath)
        self.fusedData = {"lon": data.LONGITUDE, 
                        "lat": data.LATITUDE, 
                        "depth": data.BATHYMETRY,
                        "sst": data.SST,
                        "krillDensity": data.STANDARDISED_KRILL_UNDER_1M2}
        self.logger.info(f"Read features from {self.fusedDataPath}")
        self.logger.info(f"\n{self.fusedData}")
        return

    def scaleFeatures(self):
        self.logger.info(f"Scaling features...")
        feature_columns = [key for key in self.fusedData.keys() if key != 'krillDensity']
        for key in feature_columns:
            self.fusedData[key] = (self.fusedData[key] - self.fusedData[key].mean()) / self.fusedData[key].std()
            self.logger.info(f"Feature {key} scaled, with mean {self.fusedData[key].mean()} and std {self.fusedData[key].std()}")
        self.logger.info(f"Finished scaling features")
        return

    def loadXy(self):
        self.logger.info(f"Loading data...")
        self.X = self.fusedData[['lon', 'lat', 'depth', 'sst']]
        self.y = self.fusedData['krillDensity']
        self.logger.info(f"Finished loading data")
        return