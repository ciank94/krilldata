import pandas as pd
import numpy as np
import logging
import os
import matplotlib.pyplot as plt
import sys


class ReadKrillBase:
    def __init__(self, file, outputPath):
        self.outputPath = outputPath
        self.file = file
        self.fileData = None
        self.fileDataSubset = None
        self.processData()
        return

    def processData(self):
        os.makedirs(self.outputPath, exist_ok=True)
        self.initLogger()  
        self.variableSubset()
        self.dateSubset()
        self.transformDensities()
        self.geoSubset(lonRange=(-70, -31), latRange=(-73, -50))
        self.fileDataSubset.reset_index(drop=True, inplace=True)
        self.checkPlot()
        self.printDataHead()
        return

    def initLogger(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"================={self.__class__.__name__}=====================")
        self.logger.info(f"Initializing {self.__class__.__name__}")
        self.logger.info(f"Reading data from: {self.file}")
        return

    def variableSubset(self):
        variableSubset = [
            "DATE",
            "LONGITUDE",
            "LATITUDE",
            "CLIMATOLOGICAL_TEMPERATURE",
            "WATER_DEP_MEAN_WITHIN_10KM",
            "NUMBER_OF_KRILL_UNDER_1M2"
        ]
        self.fileData = pd.read_table(self.file, sep=',', encoding='unicode_escape')
        self.fileDataSubset = self.fileData.loc[:, variableSubset]
        self.logger.info(f"Finished reading data from: {self.file}")
        self.logger.info(f"Subset to variables: {variableSubset}")
        return

    def dateSubset(self):
        # convert date to year-month-day
        self.fileDataSubset.DATE = pd.to_datetime(self.fileDataSubset.DATE, format='%d/%m/%Y')
        self.fileDataSubset = self.fileDataSubset[(self.fileDataSubset.DATE.dt.year >= 1980) & (self.fileDataSubset.DATE.dt.year <= 2016)]
        self.logger.info(f"Subset to date range 1980-2016")
        return

    def geoSubset(self, lonRange=(-70, -31), latRange=(-73, -50)):
        self.fileDataSubset = self.fileDataSubset[(self.fileDataSubset.LONGITUDE >= lonRange[0]) & (self.fileDataSubset.LONGITUDE <= lonRange[1]) & \
                              (self.fileDataSubset.LATITUDE >= latRange[0]) & (self.fileDataSubset.LATITUDE <= latRange[1])]
        self.logger.info(f"Subset to longitude range: {lonRange} and latitude range: {latRange}")
        return

    def transformDensities(self):
        validData = self.fileDataSubset.NUMBER_OF_KRILL_UNDER_1M2 >= 0
        self.fileDataSubset.loc[validData, "NUMBER_OF_KRILL_UNDER_1M2"] = \
            np.log10(self.fileDataSubset.loc[validData, "NUMBER_OF_KRILL_UNDER_1M2"] + 0.01)
        return

    def printDataHead(self):
        self.logger.info(f"Head and tail of data:")
        self.logger.info(self.fileDataSubset.head())
        self.logger.info(self.fileDataSubset.tail())
        self.logger.info(f"Number of observations: {self.fileDataSubset.shape[0]}")
        return

    def checkPlot(self):
        self.logger.info(f"Plotting data if file is not empty")
        self.savename = 'krillbaseDistributions.png'
        if os.path.exists(os.path.join(self.outputPath, self.savename)):
            self.logger.info(f"File already exists: {self.savename}")
        else:
            self.logger.info(f"File does not exist: {self.savename}")
            self.logger.info(f"File will be created: {self.savename}")
            self.plotKrillBase()
            self.logger.info(f"Finished plotting data")
        return

    def plotKrillBase(self):
        """
        Create a figure with multiple subplots showing distributions of:
        - Krill densities (log10 transformed)
        - Years
        - Months
        - Latitude
        - Longitude
        - Climatological temperature
        """
        # Define colors
        barColor = '#7FB3D5'  # Soft blue
        lineColor = '#E74C3C'  # Bright red
        barAlpha = 0.7  # Transparency for bars
        lineWidth = 2.5  # Thicker line for better visibility
        
        plt.rcParams.update({'font.size': 12})
        fig = plt.figure(figsize=(20, 8))  # Wider layout
        gs = plt.GridSpec(2, 3, figure=fig)
        
        # Krill densities histogram
        ax1 = fig.add_subplot(gs[0, 0])
        n1, bins1, _ = ax1.hist(self.fileDataSubset.NUMBER_OF_KRILL_UNDER_1M2, bins=30, color=barColor, 
                               edgecolor='white', alpha=barAlpha, linewidth=1)
        ax1Twin = ax1.twinx()
        ax1Twin.plot(bins1[:-1], np.cumsum(n1)/np.sum(n1)*100, color=lineColor, linewidth=lineWidth)
        ax1.set_xlabel('Log10 krill density', fontsize=14)
        ax1.set_ylabel('Count', fontsize=14)
        ax1Twin.set_ylabel('Cumulative %', fontsize=14, color=lineColor)
        ax1.tick_params(axis='both', which='major', labelsize=12)
        ax1Twin.tick_params(axis='y', colors=lineColor, labelsize=12)
        ax1Twin.set_ylim(0, 105)
        
        # Years histogram
        ax2 = fig.add_subplot(gs[0, 1])
        n2, bins2, _ = ax2.hist(self.fileDataSubset.DATE.dt.year, bins=np.arange(1980, 2017), color=barColor, 
                               edgecolor='white', alpha=barAlpha, linewidth=1)
        ax2Twin = ax2.twinx()
        ax2Twin.plot(bins2[:-1], np.cumsum(n2)/np.sum(n2)*100, color=lineColor, linewidth=lineWidth)
        ax2.set_xlabel('Year', fontsize=14)
        ax2.set_ylabel('Count', fontsize=14)
        ax2Twin.set_ylabel('Cumulative %', fontsize=14, color=lineColor)
        ax2.tick_params(axis='both', which='major', labelsize=12)
        ax2Twin.tick_params(axis='y', colors=lineColor, labelsize=12)
        ax2Twin.set_ylim(0, 105)

        # Months histogram
        ax3 = fig.add_subplot(gs[0, 2])
        n3, bins3, _ = ax3.hist(self.fileDataSubset.DATE.dt.month, bins=np.arange(1, 14), color=barColor, 
                               edgecolor='white', alpha=barAlpha, linewidth=1)
        ax3Twin = ax3.twinx()
        ax3Twin.plot(bins3[:-1], np.cumsum(n3)/np.sum(n3)*100, color=lineColor, linewidth=lineWidth)
        ax3.set_xlabel('Month', fontsize=14)
        ax3.set_ylabel('Count', fontsize=14)
        ax3Twin.set_ylabel('Cumulative %', fontsize=14, color=lineColor)
        ax3.tick_params(axis='both', which='major', labelsize=12)
        ax3Twin.tick_params(axis='y', colors=lineColor, labelsize=12)
        ax3.set_xticks(np.arange(1, 13))
        ax3Twin.set_ylim(0, 105)
        
        # Latitude histogram
        ax4 = fig.add_subplot(gs[1, 0])
        n4, bins4, _ = ax4.hist(self.fileDataSubset.LATITUDE, bins=30, color=barColor, 
                               edgecolor='white', alpha=barAlpha, linewidth=1)
        ax4Twin = ax4.twinx()
        ax4Twin.plot(bins4[:-1], np.cumsum(n4)/np.sum(n4)*100, color=lineColor, linewidth=lineWidth)
        ax4.set_xlabel('Latitude', fontsize=14)
        ax4.set_ylabel('Count', fontsize=14)
        ax4Twin.set_ylabel('Cumulative %', fontsize=14, color=lineColor)
        ax4.tick_params(axis='both', which='major', labelsize=12)
        ax4Twin.tick_params(axis='y', colors=lineColor, labelsize=12)
        ax4Twin.set_ylim(0, 105)
        
        # Longitude histogram
        ax5 = fig.add_subplot(gs[1, 1])
        n5, bins5, _ = ax5.hist(self.fileDataSubset.LONGITUDE, bins=30, color=barColor, 
                               edgecolor='white', alpha=barAlpha, linewidth=1)
        ax5Twin = ax5.twinx()
        ax5Twin.plot(bins5[:-1], np.cumsum(n5)/np.sum(n5)*100, color=lineColor, linewidth=lineWidth)
        ax5.set_xlabel('Longitude', fontsize=14)
        ax5.set_ylabel('Count', fontsize=14)
        ax5Twin.set_ylabel('Cumulative %', fontsize=14, color=lineColor)
        ax5.tick_params(axis='both', which='major', labelsize=12)
        ax5Twin.tick_params(axis='y', colors=lineColor, labelsize=12)
        ax5Twin.set_ylim(0, 105)
        
        # Temperature histogram
        ax6 = fig.add_subplot(gs[1, 2])
        n6, bins6, _ = ax6.hist(self.fileDataSubset.CLIMATOLOGICAL_TEMPERATURE, bins=30, color=barColor, 
                               edgecolor='white', alpha=barAlpha, linewidth=1)
        ax6Twin = ax6.twinx()
        ax6Twin.plot(bins6[:-1], np.cumsum(n6)/np.sum(n6)*100, color=lineColor, linewidth=lineWidth)
        ax6.set_xlabel('Temperature (°C)', fontsize=14)
        ax6.set_ylabel('Count', fontsize=14)
        ax6Twin.set_ylabel('Cumulative %', fontsize=14, color=lineColor)
        ax6.tick_params(axis='both', which='major', labelsize=12)
        ax6Twin.tick_params(axis='y', colors=lineColor, labelsize=12)
        ax6Twin.set_ylim(0, 105)
        
        plt.tight_layout()
        # Save figure with high DPI
        fig.savefig(os.path.join(self.outputPath, self.savename), dpi=300, bbox_inches='tight')
        self.logger.info(f"Saved figure to: {os.path.join(self.outputPath, self.savename )}")
        return 