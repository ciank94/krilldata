import pandas as pd
import numpy as np
import logging


class readKrillBase:
    def __init__(self, file):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"================={self.__class__.__name__}=====================")
        self.logger.info(f"Initializing {self.__class__.__name__}")
        self.logger.info(f"Reading data from: {file}")
        self.filedata = pd.read_table(file, sep=',', encoding='unicode_escape')
        self.logger.info(f"Finished reading data from: {file}")
        self.data = None
        self.variable_subset()
        self.date_subset()
        self.transform_densities()
        self.geo_subset(lon_range=(-70, -31), lat_range=(-73, -50))
        self.data.reset_index(drop=True, inplace=True)
        self.logger.info(f"Head and tail of data:")
        self.logger.info(self.data.head())
        self.logger.info(self.data.tail())
        self.logger.info(f"Number of observations: {self.data.shape[0]}")
        return

    def variable_subset(self):
        variable_subset = [
            "DATE",
            "LONGITUDE", 
            "LATITUDE", 
            "CLIMATOLOGICAL_TEMPERATURE",
            "WATER_DEP_MEAN_WITHIN_10KM",
            "NUMBER_OF_KRILL_UNDER_1M2"
        ]
        self.data = self.filedata.loc[:, variable_subset]
        self.logger.info(f"Subset to variables: {variable_subset}")
        return

    def date_subset(self):
        # convert date to year-month-day
        self.data.DATE = pd.to_datetime(self.data.DATE, format='%d/%m/%Y')
        self.data = self.data[(self.data.DATE.dt.year >= 1980) & (self.data.DATE.dt.year <= 2016)]
        self.logger.info(f"Subset to date range 1980-2016")
        return

    def geo_subset(self, lon_range, lat_range):
        self.data = self.data[(self.data.LONGITUDE >= lon_range[0]) & (self.data.LONGITUDE <= lon_range[1]) & \
                              (self.data.LATITUDE >= lat_range[0]) & (self.data.LATITUDE <= lat_range[1])]
        self.logger.info(f"Subset to longitude range: {lon_range} and latitude range: {lat_range}")
        return

    def transform_densities(self):
        valid_data =self.data.NUMBER_OF_KRILL_UNDER_1M2 >= 0
        self.data.loc[valid_data, "NUMBER_OF_KRILL_UNDER_1M2"] = \
            np.log10(self.data.loc[valid_data, "NUMBER_OF_KRILL_UNDER_1M2"] + 0.01)
        return

    