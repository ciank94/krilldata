from krilldata import PerformancePlot

# Initialize PerformancePlot
pp = PerformancePlot("input/", modelType="rfr")

# Create and save the plot
pp.plot_performance(save_path="output/model_performance.png")