import json
import pickle
import sys
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.demo.shap_utils import get_shap_explanation
from src.demo.input_mapping import user_inputs_to_feature_row
from config.paths import MODELS_PATH

def main():
    model_path = MODELS_PATH / "best_demo_model.pkl"
    names_path = MODELS_PATH / "demo_feature_names.json"
    
    if not model_path.exists() or not names_path.exists():
        print("Model artifacts not found.")
        return
        
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(names_path) as f:
        feature_names = json.load(f)
        
    # High risk inputs
    row = user_inputs_to_feature_row(
        sleep_hours=4.0,
        stress_level="High",
        physical_activity_min=5,
        academic_workload="High",
        social_interaction="Low",
        feature_names=feature_names
    )
    
    print("Testing SHAP with high-risk inputs...")
    shap_df = get_shap_explanation(model, row)
    print("\nTop SHAP Drivers:")
    print(shap_df.head(10))
    
    # Low risk inputs
    row_low = user_inputs_to_feature_row(
        sleep_hours=8.0,
        stress_level="Low",
        physical_activity_min=60,
        academic_workload="Low",
        social_interaction="High",
        feature_names=feature_names
    )
    
    print("\nTesting SHAP with low-risk inputs...")
    shap_df_low = get_shap_explanation(model, row_low)
    print("\nTop SHAP Drivers:")
    print(shap_df_low.head(10))

if __name__ == "__main__":
    main()
