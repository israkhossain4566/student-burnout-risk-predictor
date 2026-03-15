"""Clean StudentLife raw DataFrames (Stress, Activity, Sleep)."""
import numpy as np
import pandas as pd


def clean_stress(df: pd.DataFrame) -> pd.DataFrame:
    """Clean Stress modality: timestamp, student_id, stress_level; drop NaNs in stress_level."""
    out = df.copy()
    if "null" in out.columns:
        out = out.drop(columns=["null"])
    out["timestamp"] = pd.to_datetime(out["resp_time"], unit="s", errors="coerce")
    out["student_id"] = out["student_id"].astype(str).str.replace("Stress_", "", regex=False)
    out = out.rename(columns={"level": "stress_level"})
    out["stress_level"] = pd.to_numeric(out["stress_level"], errors="coerce")
    out = out.dropna(subset=["stress_level"])
    out["stress_level"] = out["stress_level"].astype(float)
    return out


def clean_activity(df: pd.DataFrame) -> pd.DataFrame:
    """Clean Activity modality: workload_score, recovery_score, social_score from EMA columns."""
    out = df.copy()
    if "null" in out.columns:
        out = out.drop(columns=["null"])
    out["timestamp"] = pd.to_datetime(out["resp_time"], unit="s", errors="coerce")
    out["student_id"] = out["student_id"].astype(str).str.replace("Activity_", "", regex=False)

    for col in ["Social2", "working", "other_working", "relaxing", "other_relaxing"]:
        if col not in out.columns:
            out[col] = np.nan
    for col in ["Social2", "working", "other_working", "relaxing", "other_relaxing"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["workload_score"] = out[["working", "other_working"]].sum(axis=1, min_count=1)
    out["recovery_score"] = out[["relaxing", "other_relaxing"]].sum(axis=1, min_count=1)
    out["social_score"] = out["Social2"]

    out = out[["student_id", "timestamp", "workload_score", "recovery_score", "social_score"]]
    out = out.dropna(subset=["workload_score", "recovery_score", "social_score"], how="all")
    return out


def clean_sleep(df: pd.DataFrame) -> pd.DataFrame:
    """Clean Sleep modality: sleep_hours from hour column; drop NaNs."""
    out = df.copy()
    if "null" in out.columns:
        out = out.drop(columns=["null"])
    out["timestamp"] = pd.to_datetime(out["resp_time"], unit="s", errors="coerce")
    out["student_id"] = out["student_id"].astype(str).str.replace("Sleep_", "", regex=False)
    out["sleep_hours"] = pd.to_numeric(out["hour"], errors="coerce")
    out = out[["student_id", "timestamp", "sleep_hours"]].dropna()
    return out


def clean_all(
    stress_raw: pd.DataFrame,
    activity_raw: pd.DataFrame,
    sleep_raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Clean all three modalities. Returns (stress_clean, activity_clean, sleep_clean)."""
    return (
        clean_stress(stress_raw),
        clean_activity(activity_raw),
        clean_sleep(sleep_raw),
    )
