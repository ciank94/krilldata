from krilldata import (
    KrillPredict
)

#===============IO Paths & model parameters===============
inputPath = "input/"
outputPath = "output/"
modelType = "rfr"

# ================KrillPredict class===============
# Peninsula prediction
#KrillPredict(inputPath, outputPath, modelType, scenario='peninsula')

# South Georgia prediction
KrillPredict(inputPath, outputPath, modelType, scenario='southGeorgia')