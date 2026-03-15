import sys
import pickle
import json
import pandas as pd

with open('models/best_multimodal_model.pkl', 'rb') as f:
    pipeline = pickle.load(f)
with open('models/multimodal_feature_names.json', 'rb') as f:
    cols = json.load(f)

stress_mean = 4.0; sleep_mean = 3.0; workload_mean = 5.0; recovery_mean = 0.0; social_mean = 1.0
row_dict = {}

for c in cols:
    prefix = c.split('_')[0]
    if prefix == 'stress': val = stress_mean
    elif prefix == 'sleep': val = sleep_mean
    elif prefix == 'workload': val = workload_mean
    elif prefix == 'recovery': val = recovery_mean
    elif prefix == 'social': val = social_mean
    else: val = 0.0

    if c == f'{prefix}_mean': row_dict[c] = val
    elif c in (f'{prefix}_std', f'{prefix}_max'): row_dict[c] = val if 'max' in c else 0.0
    elif c == f'{prefix}_count': row_dict[c] = 1.0
    elif c.startswith(f'{prefix}_z_'): row_dict[c] = 0.0
    elif c.startswith(f'{prefix}_') and ('diff' in c or 'roll' in c):
        if 'roll4_mean' in c and 'z_' not in c:
            row_dict[c] = val
        else:
            row_dict[c] = 0.0
    elif c == f'{prefix}_missing': row_dict[c] = 0
    elif '_is_low' in c: row_dict[c] = 0
    else: row_dict[c] = 0.0

df = pd.DataFrame([row_dict])[cols]
print('New Prob:', float(pipeline.predict_proba(df)[0,1]))
