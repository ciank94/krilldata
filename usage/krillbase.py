from krilldata import ReadKrillBase, DataFusion

inputPath = "input/"
outputPath = "output/"
kb = ReadKrillBase(inputPath, outputPath)
DataFusion(kb.fileDataSubset, inputPath, outputPath) # fuse bathymetry, SST & krillbase data