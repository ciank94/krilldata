from krilldata import (
    ReadKrillBase,
    DataFusion,
    KrillTrain
)

#===============IO Paths & model parameters===============
inputPath = "input/"
outputPath = "output/"
modelType = "rfr"

# ================ReadKrillBase class===============
kb = ReadKrillBase(inputPath, outputPath) # subset krillbase data by key variables, years, lon & lat ranges

# ================DataFusion class================
DataFusion(kb.fileDataSubset, inputPath, outputPath) # fuse bathymetry, SST & krillbase data

# ================KrillTrain class===============
KrillTrain(inputPath, outputPath, modelType, scenario="default") # read fused data and train regressor (scenario: default or random)