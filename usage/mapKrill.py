from krilldata import (
    KrillPredict,
    MapKrillDensity
)

#===============IO Paths & model parameters===============
inputPath = "input/"
outputPath = "output/"
modelType = "rfr"

# ================KrillPredict class===============
# Peninsula prediction
#KrillPredict(inputPath, outputPath, modelType, scenario='peninsula')

# South Georgia prediction
#KrillPredict(inputPath, outputPath, modelType, scenario='southGeorgia')

# ================MapKrillDensity class===============
# Plots krill density over all scenarios
MapKrillDensity(inputPath, outputPath, modelType='rfr')