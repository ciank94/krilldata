from krilldata.readkrillbase import readKrillBase

file = f"input_files/krillbase.csv"
output_path = "output_files"
kb = readKrillBase(file, output_path)
kb.plotkrillbase()
breakpoint()