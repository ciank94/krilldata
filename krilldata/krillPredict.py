from joblib import load


#todo: predict
class KrillPredict:
    def __init__(self, inputPath, outputPath, modelType):
        self.inputPath = inputPath
        self.outputPath = outputPath
        self.modelType = modelType
        self.model = load(f"{inputPath}/{self.modelType}Model.joblib")
        return

    def predict(self):
        predictions = self.model.predict(new_data)
        return