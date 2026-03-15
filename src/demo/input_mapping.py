import pandas as pd
from src.features.build_weekly import DEMO_ALIGNED_COLS

# Stress: StudentLife scale 1-5. Map Low=1.5, Medium=3.0, High=4.5
STRESS_MAP = {"Low": 1.5, "Medium": 3.0, "High": 4.5}

# Workload: Low=1.5, Medium=3.0, High=4.5
WORKLOAD_MAP = {"Low": 1.5, "Medium": 3.0, "High": 4.5}

# Social: High=3, Medium=1.5, Low=0 (Ordinal for visible deficit calculation)
SOCIAL_MAP = {"Low": 0, "Medium": 1.5, "High": 3}

def _activity_to_recovery(activity_min: float) -> float:
    """Map minutes to 0-6 recovery scale."""
    return min(6.0, max(0.0, float(activity_min) / 20.0))


def user_inputs_to_feature_row(
    sleep_hours: float,
    stress_level: str,
    physical_activity_min: float,
    academic_workload: str,
    social_interaction: str,
    feature_names: list[str] | None = None,
) -> pd.DataFrame:
    """
    Map UI inputs to the engineered 'Strain' features for the demo-aligned model.
    """
    cols = feature_names if feature_names is not None else DEMO_ALIGNED_COLS

    # 1. Map raw metrics
    stress_mean = STRESS_MAP.get(stress_level, 3.0)
    sleep_mean = max(3.0, min(10.0, float(sleep_hours)))
    workload_mean = WORKLOAD_MAP.get(academic_workload, 3.0)
    recovery_mean = _activity_to_recovery(physical_activity_min)
    social_mean = SOCIAL_MAP.get(social_interaction, 2.5)

    # 2. Calculate Engineered Features
    stress_score = stress_mean
    workload_score = workload_mean
    sleep_deficit = max(0.0, 8.0 - sleep_mean)
    activity_deficit = max(0.0, 3.0 - recovery_mean)
    social_deficit = max(0.0, 3.0 - social_mean)
    stress_workload_int = stress_score * workload_score
    isolation_flag = 1 if social_mean == 0 else 0
    strain_index = (stress_score + workload_score) / (sleep_mean + social_mean + 1.0)

    # 3. Build row
    row_data = {
        "sleep_mean": sleep_mean,
        "stress_mean": stress_mean,
        "workload_mean": workload_mean,
        "recovery_mean": recovery_mean,
        "social_mean": social_mean,
        "sleep_deficit": sleep_deficit,
        "stress_score": stress_score,
        "workload_score": workload_score,
        "social_deficit": social_deficit,
        "stress_workload_int": stress_workload_int,
        "activity_deficit": activity_deficit,
        "isolation_flag": isolation_flag,
        "strain_index": strain_index
    }

    # Ensure we only return requested columns and in correct order
    row = {c: row_data.get(c, 0.0) for c in cols}
    return pd.DataFrame([row])[cols]
