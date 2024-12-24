import pandas as pd
import numpy as np
import logging
import os
import matplotlib.pyplot as plt
import sys


class readKrillBase:
    def __init__(self, file, output_path):
        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)
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
        self.check_plot()
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

    def check_plot(self):
        self.logger.info(f"Plotting data if file is not empty")
        self.savename = 'krillbase_distributions.png'
        if os.path.exists(os.path.join(self.output_path, self.savename)):
            self.logger.info(f"File already exists: {self.savename}")
        else:
            self.logger.info(f"File does not exist: {self.savename}")
            self.logger.info(f"File will be created: {self.savename}")
            self.plotkrillbase()
            self.logger.info(f"Finished plotting data")
        return

    def plotkrillbase(self):
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
        bar_color = '#7FB3D5'  # Soft blue
        line_color = '#E74C3C'  # Bright red
        bar_alpha = 0.7  # Transparency for bars
        line_width = 2.5  # Thicker line for better visibility
        
        plt.rcParams.update({'font.size': 12})
        fig = plt.figure(figsize=(20, 8))  # Wider layout
        gs = plt.GridSpec(2, 3, figure=fig)
        
        # Krill densities histogram
        ax1 = fig.add_subplot(gs[0, 0])
        n1, bins1, _ = ax1.hist(self.data.NUMBER_OF_KRILL_UNDER_1M2, bins=30, color=bar_color, 
                               edgecolor='white', alpha=bar_alpha, linewidth=1)
        ax1_twin = ax1.twinx()
        ax1_twin.plot(bins1[:-1], np.cumsum(n1)/np.sum(n1)*100, color=line_color, linewidth=line_width)
        ax1.set_xlabel('Log10 krill density', fontsize=14)
        ax1.set_ylabel('Count', fontsize=14)
        ax1_twin.set_ylabel('Cumulative %', fontsize=14, color=line_color)
        ax1.tick_params(axis='both', which='major', labelsize=12)
        ax1_twin.tick_params(axis='y', colors=line_color, labelsize=12)
        ax1_twin.set_ylim(0, 105)
        
        # Years histogram
        ax2 = fig.add_subplot(gs[0, 1])
        n2, bins2, _ = ax2.hist(self.data.DATE.dt.year, bins=np.arange(1980, 2017), color=bar_color, 
                               edgecolor='white', alpha=bar_alpha, linewidth=1)
        ax2_twin = ax2.twinx()
        ax2_twin.plot(bins2[:-1], np.cumsum(n2)/np.sum(n2)*100, color=line_color, linewidth=line_width)
        ax2.set_xlabel('Year', fontsize=14)
        ax2.set_ylabel('Count', fontsize=14)
        ax2_twin.set_ylabel('Cumulative %', fontsize=14, color=line_color)
        ax2.tick_params(axis='both', which='major', labelsize=12)
        ax2_twin.tick_params(axis='y', colors=line_color, labelsize=12)
        ax2_twin.set_ylim(0, 105)

        # Months histogram
        ax3 = fig.add_subplot(gs[0, 2])
        n3, bins3, _ = ax3.hist(self.data.DATE.dt.month, bins=np.arange(1, 14), color=bar_color, 
                               edgecolor='white', alpha=bar_alpha, linewidth=1)
        ax3_twin = ax3.twinx()
        ax3_twin.plot(bins3[:-1], np.cumsum(n3)/np.sum(n3)*100, color=line_color, linewidth=line_width)
        ax3.set_xlabel('Month', fontsize=14)
        ax3.set_ylabel('Count', fontsize=14)
        ax3_twin.set_ylabel('Cumulative %', fontsize=14, color=line_color)
        ax3.tick_params(axis='both', which='major', labelsize=12)
        ax3_twin.tick_params(axis='y', colors=line_color, labelsize=12)
        ax3.set_xticks(np.arange(1, 13))
        ax3_twin.set_ylim(0, 105)
        
        # Latitude histogram
        ax4 = fig.add_subplot(gs[1, 0])
        n4, bins4, _ = ax4.hist(self.data.LATITUDE, bins=30, color=bar_color, 
                               edgecolor='white', alpha=bar_alpha, linewidth=1)
        ax4_twin = ax4.twinx()
        ax4_twin.plot(bins4[:-1], np.cumsum(n4)/np.sum(n4)*100, color=line_color, linewidth=line_width)
        ax4.set_xlabel('Latitude', fontsize=14)
        ax4.set_ylabel('Count', fontsize=14)
        ax4_twin.set_ylabel('Cumulative %', fontsize=14, color=line_color)
        ax4.tick_params(axis='both', which='major', labelsize=12)
        ax4_twin.tick_params(axis='y', colors=line_color, labelsize=12)
        ax4_twin.set_ylim(0, 105)
        
        # Longitude histogram
        ax5 = fig.add_subplot(gs[1, 1])
        n5, bins5, _ = ax5.hist(self.data.LONGITUDE, bins=30, color=bar_color, 
                               edgecolor='white', alpha=bar_alpha, linewidth=1)
        ax5_twin = ax5.twinx()
        ax5_twin.plot(bins5[:-1], np.cumsum(n5)/np.sum(n5)*100, color=line_color, linewidth=line_width)
        ax5.set_xlabel('Longitude', fontsize=14)
        ax5.set_ylabel('Count', fontsize=14)
        ax5_twin.set_ylabel('Cumulative %', fontsize=14, color=line_color)
        ax5.tick_params(axis='both', which='major', labelsize=12)
        ax5_twin.tick_params(axis='y', colors=line_color, labelsize=12)
        ax5_twin.set_ylim(0, 105)
        
        # Temperature histogram
        ax6 = fig.add_subplot(gs[1, 2])
        n6, bins6, _ = ax6.hist(self.data.CLIMATOLOGICAL_TEMPERATURE, bins=30, color=bar_color, 
                               edgecolor='white', alpha=bar_alpha, linewidth=1)
        ax6_twin = ax6.twinx()
        ax6_twin.plot(bins6[:-1], np.cumsum(n6)/np.sum(n6)*100, color=line_color, linewidth=line_width)
        ax6.set_xlabel('Temperature (°C)', fontsize=14)
        ax6.set_ylabel('Count', fontsize=14)
        ax6_twin.set_ylabel('Cumulative %', fontsize=14, color=line_color)
        ax6.tick_params(axis='both', which='major', labelsize=12)
        ax6_twin.tick_params(axis='y', colors=line_color, labelsize=12)
        ax6_twin.set_ylim(0, 105)
        
        plt.tight_layout()
        # Save figure with high DPI
        fig.savefig(os.path.join(self.output_path, self.savename), dpi=300, bbox_inches='tight')
        self.logger.info(f"Saved figure to: {os.path.join(self.output_path, self.savename )}")
        return 