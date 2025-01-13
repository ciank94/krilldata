from joblib import load



#todo: predict
class KrillPredict:
    def __init__(self, inputPath, outputPath):
        self.inputPath = inputPath
        self.outputPath = outputPath
        
    def predict(self):
        model = load('path/to/gbrModel.joblib')  # or rfModel.joblib for random forest
        predictions = model.predict(new_data)