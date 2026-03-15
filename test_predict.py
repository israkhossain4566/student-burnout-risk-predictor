import sys
import pickle
import json
import pandas as pd
from pathlib import Path
from src.demo.input_mapping import user_inputs_to_feature_row
from src.explain.shap_utils import get_local_shap_values

def main():
    model_path = Path("models/best_multimodal_model.pkl")
    names_path = Path("models/multimodal_feature_names.json")
    bg_path = Path("models/background_sample.csv")
    
    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)
    with open(names_path) as f:
        feature_names = json.load(f)
    background_df = pd.read_csv(bg_path)
    background_df = background_df[[c for c in feature_names if c in background_df.columns]]
        
    row = user_inputs_to_feature_row(
        sleep_hours=3,
        stress_level="High",
        physical_activity_min=120,
        academic_workload="High",
        social_interaction="Low",
        feature_names=feature_names,
    )
    prob = float(pipeline.predict_proba(row)[0, 1])
    print("Probability:", prob)
    
    base_val, shap_1d = get_local_shap_values(
        pipeline, background_df, row, feature_names, model_name="GradientBoosting"
    )
    
    name_val = list(zip(feature_names, shap_1d))
    name_val.sort(key=lambda x: x[1])
    
    print("\nTop Negative Impacts (Reduce Risk):")
    for n, v in name_val[:10]:
        print(f"{n}: {v:.4f}")
        
    print("\nTop Positive Impacts (Increase Risk):")
    for n, v in name_val[-10:]:
        print(f"{n}: {v:.4f}")

if __name__ == "__main__":
    main()
