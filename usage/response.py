from krilldata import ResponseCurves

# Initialize ResponseCurves
rc = ResponseCurves("input/", modelType="rfr")

# Generate all response curves in a single plot with standard deviation bands
rc.plot_all_response_curves(save_path="output/response_curves.png")

