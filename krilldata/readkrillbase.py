import pandas as pd
import numpy as np
import logging
import os
import matplotlib.pyplot as plt

class ReadKrillBase:
    loggerDescription = "\nReadKrillBase class description:\n\
        reads data from the krillbase.csv from input files\n\
        subsets key variables: `[date, lon, lat, temp, depth, krillDensity]`\n\
        subsets years: `1980:2016`\n\
        log10 transform krillDensity\n\
        subsets lon and lat ranges\n\
        visualises: `output/krillDistributions.png`\n\
        prints head and tail of data to terminal\n"
        
    defaultVariableSubset = [
        "DATE",
        "LONGITUDE",
        "LATITUDE",
        "STANDARDISED_KRILL_UNDER_1M2"
    ]
    defaultLonRange = (-70, -31)
    defaultLatRange = (-73, -50)
    defaultTimeLimits = {
        'startYear': 1976,
        'endYear': 2016
    }
    
    krillDistributionsFigName = "krillDistributions.png"
    krillFusedDataFilename = "krillFusedData.csv"
    logging.basicConfig(level=logging.INFO) # set logging level for class
    
    def __init__(self, inputPath, outputPath):
        self.inputPath = inputPath
        self.outputPath = outputPath
        self.file = f"{inputPath}/krillbase.csv"
        self.fileData = None
        self.fileDataSubset = None
        self.initLogger() 
        if not os.path.exists(os.path.join(self.inputPath, ReadKrillBase.krillFusedDataFilename)):
            self.processData()
            self.checkPlot()
        else:
            self.logger.info(f"Fused krillbase file already exists, no need to process data")
            self.fileDataSubset = pd.read_csv(os.path.join(self.inputPath, ReadKrillBase.krillFusedDataFilename))
            self.fileDataSubset.DATE = pd.to_datetime(self.fileDataSubset.DATE, format='%Y-%m-%d')
            self.checkPlot()
        
        return

    def processData(self):
        os.makedirs(self.inputPath, exist_ok=True)
        os.makedirs(self.outputPath, exist_ok=True)
        self.variableSubset()
        self.dateSubset()
        self.transformDensities()
        self.geoSubset(lonRange=ReadKrillBase.defaultLonRange, latRange=ReadKrillBase.defaultLatRange)
        self.fileDataSubset.reset_index(drop=True, inplace=True)
        self.printDataHead()
        return

    def initLogger(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"================={self.__class__.__name__}=====================")
        self.logger.info(f"{ReadKrillBase.loggerDescription}")
        self.logger.info(f"Reading data from: {self.file}")
        return

    def variableSubset(self):
        variableSubset = ReadKrillBase.defaultVariableSubset
        self.fileData = pd.read_table(self.file, sep=',', encoding='unicode_escape')
        self.fileDataSubset = self.fileData.loc[:, variableSubset]
        self.logger.info(f"Finished reading data from: {self.file}")
        self.logger.info(f"Subset to variables: {variableSubset}")
        return

    def dateSubset(self):
        # convert date to year-month-day
        self.fileDataSubset.DATE = pd.to_datetime(self.fileDataSubset.DATE, format='%d/%m/%Y')
        self.fileDataSubset = self.fileDataSubset[(self.fileDataSubset.DATE.dt.year >= ReadKrillBase.defaultTimeLimits['startYear']) & (self.fileDataSubset.DATE.dt.year <= ReadKrillBase.defaultTimeLimits['endYear'])]
        self.logger.info(f"Subset to date range {ReadKrillBase.defaultTimeLimits['startYear']}-{ReadKrillBase.defaultTimeLimits['endYear']}")
        return

    def geoSubset(self, lonRange=(-70, -31), latRange=(-73, -50)):
        self.fileDataSubset = self.fileDataSubset[(self.fileDataSubset.LONGITUDE >= lonRange[0]) & (self.fileDataSubset.LONGITUDE <= lonRange[1]) & \
                              (self.fileDataSubset.LATITUDE >= latRange[0]) & (self.fileDataSubset.LATITUDE <= latRange[1])]
        self.logger.info(f"Subset to longitude range: {lonRange} and latitude range: {latRange}")
        return

    def transformDensities(self):
        validData = self.fileDataSubset.STANDARDISED_KRILL_UNDER_1M2 >= 0
        self.fileDataSubset.loc[validData, "STANDARDISED_KRILL_UNDER_1M2"] = np.log10(self.fileDataSubset.loc[validData, "STANDARDISED_KRILL_UNDER_1M2"] + 0.01)
        #self.fileDataSubset.loc[validData, "STANDARDISED_KRILL_UNDER_1M2"] = np.log1p(self.fileDataSubset.loc[validData, "STANDARDISED_KRILL_UNDER_1M2"]) # exp(x) - 1 is the inverse
        return

    def printDataHead(self):
        self.logger.info(f"Head and tail of data:")
        self.logger.info(self.fileDataSubset.head())
        self.logger.info(self.fileDataSubset.tail())
        self.logger.info(f"Number of observations: {self.fileDataSubset.shape[0]}")
        return

    def checkPlot(self):
        self.logger.info(f"Plotting data if file is not empty")
        self.saveName = ReadKrillBase.krillDistributionsFigName
        if os.path.exists(os.path.join(self.outputPath, self.saveName)):
            self.logger.info(f"File already exists: {self.saveName}")
        else:
            self.logger.info(f"File does not exist: {self.saveName}")
            self.logger.info(f"File will be created: {self.saveName}")
            self.plotKrillBase()
            self.logger.info(f"Finished plotting data")
        return

    def plotKrillBase(self):
        """
        Create a figure with multiple subplots showing distributions of:
        - Krill densities (log10 transformed)
        - Years
        - Latitude
        - Longitude
        """
        # Define colors
        barColor = '#7FB3D5'  # Soft blue
        lineColor = '#E74C3C'  # Bright red
        barAlpha = 0.7  # Transparency for bars
        lineWidth = 2.5  # Thicker line for better visibility
        
        plt.rcParams.update({'font.size': 12})
        fig = plt.figure(figsize=(12, 10))
        gs = plt.GridSpec(2, 2)
        
        # Krill densities histogram
        ax1 = fig.add_subplot(gs[0, 0])
        n1, bins1, _ = ax1.hist(self.fileDataSubset.STANDARDISED_KRILL_UNDER_1M2, bins=30, color=barColor, 
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
        n2, bins2, _ = ax2.hist(self.fileDataSubset.DATE.dt.year, bins=np.arange(1976, 2017), color=barColor, 
                               edgecolor='white', alpha=barAlpha, linewidth=1)
        ax2Twin = ax2.twinx()
        ax2Twin.plot(bins2[:-1], np.cumsum(n2)/np.sum(n2)*100, color=lineColor, linewidth=lineWidth)
        ax2.set_xlabel('Year', fontsize=14)
        ax2.set_ylabel('Count', fontsize=14)
        ax2Twin.set_ylabel('Cumulative %', fontsize=14, color=lineColor)
        ax2.tick_params(axis='both', which='major', labelsize=12)
        ax2Twin.tick_params(axis='y', colors=lineColor, labelsize=12)
        ax2Twin.set_ylim(0, 105)
        
        # Latitude histogram
        ax3 = fig.add_subplot(gs[1, 0])
        n4, bins4, _ = ax3.hist(self.fileDataSubset.LATITUDE, bins=30, color=barColor, 
                               edgecolor='white', alpha=barAlpha, linewidth=1)
        ax3Twin = ax3.twinx()
        ax3Twin.plot(bins4[:-1], np.cumsum(n4)/np.sum(n4)*100, color=lineColor, linewidth=lineWidth)
        ax3.set_xlabel('Latitude', fontsize=14)
        ax3.set_ylabel('Count', fontsize=14)
        ax3Twin.set_ylabel('Cumulative %', fontsize=14, color=lineColor)
        ax3.tick_params(axis='both', which='major', labelsize=12)
        ax3Twin.tick_params(axis='y', colors=lineColor, labelsize=12)
        ax3Twin.set_ylim(0, 105)
        
        # Longitude histogram
        ax4 = fig.add_subplot(gs[1, 1])
        n5, bins5, _ = ax4.hist(self.fileDataSubset.LONGITUDE, bins=30, color=barColor, 
                               edgecolor='white', alpha=barAlpha, linewidth=1)
        ax4Twin = ax4.twinx()
        ax4Twin.plot(bins5[:-1], np.cumsum(n5)/np.sum(n5)*100, color=lineColor, linewidth=lineWidth)
        ax4.set_xlabel('Longitude', fontsize=14)
        ax4.set_ylabel('Count', fontsize=14)
        ax4Twin.set_ylabel('Cumulative %', fontsize=14, color=lineColor)
        ax4.tick_params(axis='both', which='major', labelsize=12)
        ax4Twin.tick_params(axis='y', colors=lineColor, labelsize=12)
        ax4Twin.set_ylim(0, 105)
        
        plt.tight_layout()
        # Save figure with high DPI
        fig.savefig(os.path.join(self.outputPath, self.saveName), dpi=300, bbox_inches='tight')
        self.logger.info(f"Saved figure to: {os.path.join(self.outputPath, self.saveName )}")
        return 