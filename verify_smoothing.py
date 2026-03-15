
import json
import pickle
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Setup paths to import project modules
ROOT = Path(".").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.demo.input_mapping import user_inputs_to_feature_row

# Paths
MODELS_PATH = ROOT / "models"
model_path = MODELS_PATH / "best_multimodal_model.pkl"
names_path = MODELS_PATH / "multimodal_feature_names.json"

def test_smoothing():
    if not model_path.exists():
        print("Model not found.")
        return

    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)
    with open(names_path) as f:
        feature_names = json.load(f)

    # Test several variations of stress and sleep
    results = []
    
    stress_options = ["Low", "Medium", "High"]
    sleep_options = [4, 5, 6, 7, 8, 9]

    for stress in stress_options:
        for sleep in sleep_options:
            row = user_inputs_to_feature_row(
                sleep_hours=sleep,
                stress_level=stress,
                physical_activity_min=30,
                academic_workload="Medium",
                social_interaction="Medium",
                feature_names=feature_names,
            )
            prob = float(pipeline.predict_proba(row)[0, 1])
            results.append((stress, sleep, prob))
    
    print("\nDiagnostic: Probability variabilty check (Sigmoid)")
    print("-" * 50)
    print(f"{'Stress':<10} {'Sleep':<10} {'Probability'}")
    unique_probs = set()
    for s, sl, p in results:
        unique_probs.add(round(p, 4))
        print(f"{s:<10} {sl:<10} {p:.2%}")
    
    print("-" * 50)
    print(f"Unique probabilities found: {len(unique_probs)}")
    if len(unique_probs) > 10:
        print("SUCCESS: Probabilities are smooth and varied.")
    else:
        print("WARNING: Probabilities are still somewhat discrete.")

if __name__ == "__main__":
    test_smoothing()
