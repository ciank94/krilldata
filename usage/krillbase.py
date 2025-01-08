from krilldata import ReadKrillBase, DataFusion, KrillTrain

#===============Input & Output Paths===============
inputPath = "input/"
outputPath = "output/"

# ================ReadKrillBase class===============
kb = ReadKrillBase(inputPath, outputPath) # subset krillbase data by key variables, years, lon & lat ranges

# ================DataFusion class================
DataFusion(kb.fileDataSubset, inputPath, outputPath) # fuse bathymetry, SST & krillbase data

# ================KrillTrain class===============
KrillTrain(inputPath, outputPath) # read fused data and train regressor