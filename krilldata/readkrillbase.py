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
        #self.date_subset()
        self.transform_densities()
        return

    def variable_subset(self):
        variable_subset = ["DATE","LONGITUDE", "LATITUDE", "NUMBER_OF_KRILL_UNDER_1M2"]
        self.data = self.filedata.loc[:, variable_subset]
        self.logger.info(f"Subset to variables: {variable_subset}")
        return

    def date_subset(self):
        # convert date to year-month-day
        self.data.DATE = pd.to_datetime(self.data.DATE, format='%d/%m/%Y')
        self.data = self.data[(self.data.DATE.dt.year >= 2000) & (self.data.DATE.dt.year <= 2019)]
        self.logger.info(f"Subset to date range 2000-2019")
        return

    def transform_densities(self):
        valid_data =self.data.NUMBER_OF_KRILL_UNDER_1M2 >= 0
        self.data.loc[valid_data, "NUMBER_OF_KRILL_UNDER_1M2"] = \
            np.log10(self.data.loc[valid_data, "NUMBER_OF_KRILL_UNDER_1M2"] + 0.01)
        return