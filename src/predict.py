import joblib
import pandas as pd
import numpy as np
import os

def predict_grid_demand(feature_dict):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model = joblib.load(os.path.join(base_dir, "models", "lightgbm_grid_model.joblib"))
    feature_cols = joblib.load(os.path.join(base_dir, "models", "feature_columns.joblib"))
    df = pd.DataFrame([feature_dict])[feature_cols]
    pred = model.predict(df)[0]
    return {"forecasted_load_mw": round(float(pred), 2)}
