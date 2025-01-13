from krilldata import (
    ReadKrillBase,
    DataFusion,
    KrillTrain,
    KrillPredict
)

#===============IO Paths & model parameters===============
inputPath = "input/"
outputPath = "output/"
modelType = "gbr"

# ================ReadKrillBase class===============
kb = ReadKrillBase(inputPath, outputPath) # subset krillbase data by key variables, years, lon & lat ranges

# ================DataFusion class================
DataFusion(kb.fileDataSubset, inputPath, outputPath) # fuse bathymetry, SST & krillbase data

# ================KrillTrain class===============
KrillTrain(inputPath, outputPath, modelType) # read fused data and train regressor

# ================KrillPredict class===============
KrillPredict(inputPath, outputPath, modelType) # read trained model and make predictions