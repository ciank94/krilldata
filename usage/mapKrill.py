from krilldata import (
    KrillPredict
)

#===============IO Paths & model parameters===============
inputPath = "input/"
outputPath = "output/"
modelType = "gbr"

# ================KrillPredict class===============
KrillPredict(inputPath, outputPath, modelType) # read trained model and make predictions