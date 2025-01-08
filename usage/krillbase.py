from krilldata import ReadKrillBase, DataFusion, KrillPredict

#===============Input & Output Paths===============
inputPath = "input/"
outputPath = "output/"

# ================ReadKrillBase class===============
kb = ReadKrillBase(inputPath, outputPath) # subset krillbase data by key variables, years, lon & lat ranges

# ================DataFusion class================
DataFusion(kb.fileDataSubset, inputPath, outputPath) # fuse bathymetry, SST & krillbase data

# ================KrillPredict class===============
KrillPredict(inputPath, outputPath) # read fused data and predict