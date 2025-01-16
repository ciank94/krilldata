from krilldata import (
    KrillPredict
)

#===============IO Paths & model parameters===============
inputPath = "input/"
outputPath = "output/"
modelType = "gbr"

# ================KrillPredict class===============
# Peninsula prediction
#KrillPredict(inputPath, outputPath, modelType, scenario='peninsula')

# South Georgia prediction
KrillPredict(inputPath, outputPath, modelType, scenario='southGeorgia')