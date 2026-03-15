"""Quick diagnostic: print predicted probabilities for 3 known scenarios."""
import pickle
import json
import sys
sys.path.insert(0, '.')

import pandas as pd
from src.demo.input_mapping import user_inputs_to_feature_row

feat  = json.load(open('models/multimodal_feature_names.json'))
model = pickle.load(open('models/best_multimodal_model.pkl', 'rb'))

combos = [
    ("Extreme LOW  (Sleep=10, Stress=Low, Activity=120, Workload=Low, Social=High)",
     dict(sleep_hours=10, stress_level='Low', physical_activity_min=120,
          academic_workload='Low', social_interaction='High')),
    ("Medium       (Sleep=7,  Stress=Med, Activity=30,  Workload=Med, Social=Med)",
     dict(sleep_hours=7,  stress_level='Medium', physical_activity_min=30,
          academic_workload='Medium', social_interaction='Medium')),
    ("Extreme HIGH  (Sleep=3,  Stress=High, Activity=0,  Workload=High, Social=Low)",
     dict(sleep_hours=3,  stress_level='High', physical_activity_min=0,
          academic_workload='High', social_interaction='Low')),
]

print('=' * 65)
print(f"{'Scenario':<55} {'Prob':>6}  {'Level'}")
print('=' * 65)
for label, kw in combos:
    row  = user_inputs_to_feature_row(**kw, feature_names=feat)
    prob = float(model.predict_proba(row)[0, 1])
    tag  = 'LOW' if prob < 0.33 else ('HIGH' if prob > 0.66 else 'MEDIUM')
    print(f"{label:<55} {prob:>6.3f}  {tag}")
print('=' * 65)
