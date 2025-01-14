from krilldata import (
    ReadKrillBase,
    DataFusion,
    KrillTrain
)

#===============IO Paths & model parameters===============
inputPath = "input/"
outputPath = "output/"
modelTypes = ["svm", "rfr", "gbr", "dtr", "mlr", "nnr"]

for modelType in modelTypes:
    # ================ReadKrillBase class===============
    kb = ReadKrillBase(inputPath, outputPath) # subset krillbase data by key variables, years, lon & lat ranges

    # ================DataFusion class================
    DataFusion(kb.fileDataSubset, inputPath, outputPath) # fuse bathymetry, SST & krillbase data

    # ================KrillTrain class===============
    KrillTrain(inputPath, outputPath, modelType) # read fused data and train regressor
