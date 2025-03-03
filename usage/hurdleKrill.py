# usage/hurdleKrill.py
from krilldata import KrillHurdle

# Initialize hurdle model
hurdle_model = KrillHurdle("input", "output", "rfc", "rfr")

# Make predictions and evaluate
hurdle_model.predict()
metrics = hurdle_model.evaluate() 

# Save the predictions to CSV
hurdle_model.save_predictions()

print("\nHurdle Model Evaluation Complete!")
print(f"Classification Accuracy: {metrics['accuracy']:.4f}")
if metrics['rmse'] is not None:
    print(f"Regression RMSE (presence only): {metrics['rmse']:.4f}")
if metrics['r2'] is not None:
    print(f"Regression R2 (presence only): {metrics['r2']:.4f}")

print("\nSee output directory for detailed visualizations and predictions.")