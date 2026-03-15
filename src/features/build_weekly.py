"""Build weekly multimodal feature table from cleaned Stress, Activity, Sleep DataFrames."""
import numpy as np
import pandas as pd


def add_student_zscore(
    df: pd.DataFrame,
    id_col: str,
    value_col: str,
    new_col: str,
    time_col: str | None = "timestamp",
) -> pd.DataFrame:
    """Causal (leakage-safe) z-score per student using expanding mean/std in chronological order."""
    out = df.copy()
    if time_col and time_col in out.columns:
        out = out.sort_values([id_col, time_col]).copy()
    else:
        out = out.sort_values([id_col]).copy()

    def _exp_z(s: pd.Series) -> pd.Series:
        exp_mean = s.expanding(min_periods=1).mean()
        exp_std = s.expanding(min_periods=2).std().fillna(0.0)
        exp_std = exp_std.replace(0, np.nan).fillna(1.0)
        return (s - exp_mean) / exp_std

    out[new_col] = out.groupby(id_col)[value_col].transform(_exp_z)
    return out


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add interpretable strain and recovery features for demo responsiveness."""
    out = df.copy()
    
    # 1. Stress & Workload Scores (Higher = More pressure)
    out["stress_score"] = out["stress_mean"].fillna(2.5)
    out["workload_score"] = out["workload_mean"].fillna(3.0)
    
    # 2. Deficits (Higher = More strain/Less recovery)
    # Relative to healthy "ideal" values for students
    out["sleep_deficit"] = (8.0 - out["sleep_mean"]).clip(lower=0)
    out["activity_deficit"] = (3.0 - out["recovery_mean"]).clip(lower=0) # recovery_mean 0-6 scale (15min units)
    # social_mean 1-4 scale. Map to 0-3 score: (score-1). Deficit = 3 - score.
    social_score = (out.get("social_mean", 2.5) - 1.0).clip(0, 3)
    out["social_deficit"] = (3.0 - social_score).clip(lower=0)
    
    # 3. Intersections & Flags
    out["stress_workload_int"] = out["stress_score"] * out["workload_score"]
    out["isolation_flag"] = (out.get("social_mean", 2.5) < 2.0).astype(int)
    
    # 4. Strain Index (Combined Pressure / Combined Recovery)
    out["strain_index"] = (out["stress_score"] + out["workload_score"]) / (out["sleep_mean"] + out.get("social_mean", 2.5) + 1.0)
    
    return out


def build_weekly_multimodal(
    stress_clean: pd.DataFrame,
    activity_clean: pd.DataFrame,
    sleep_clean: pd.DataFrame,
) -> pd.DataFrame:
    """Build one weekly multimodal DataFrame with counts, missingness, and engineered features."""
    stress = stress_clean.copy()
    activity = activity_clean.copy()
    sleep = sleep_clean.copy()

    for df, time_col in [(stress, "timestamp"), (activity, "timestamp"), (sleep, "timestamp")]:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

    # Per-student z-scores
    stress = add_student_zscore(stress, "student_id", "stress_level", "stress_z", time_col="timestamp")
    for col, zcol in [("workload_score", "workload_z"), ("recovery_score", "recovery_z"), ("social_score", "social_z")]:
        if col in activity.columns:
            activity = add_student_zscore(activity, "student_id", col, zcol, time_col="timestamp")
    sleep = add_student_zscore(sleep, "student_id", "sleep_hours", "sleep_z", time_col="timestamp")

    stress["week"] = stress["timestamp"].dt.isocalendar().week.astype(int)
    activity["week"] = activity["timestamp"].dt.isocalendar().week.astype(int)
    sleep["week"] = sleep["timestamp"].dt.isocalendar().week.astype(int)

    stress_weekly = (
        stress.groupby(["student_id", "week"])
        .agg(
            stress_mean=("stress_level", "mean"),
            stress_std=("stress_level", "std"),
            stress_max=("stress_level", "max"),
            stress_count=("stress_level", "count"),
            stress_z_mean=("stress_z", "mean"),
            stress_z_std=("stress_z", "std"),
            stress_z_max=("stress_z", "max"),
        )
        .reset_index()
    )

    agg_dict = {}
    for prefix, val_col, z_col in [
        ("workload", "workload_score", "workload_z"),
        ("recovery", "recovery_score", "recovery_z"),
        ("social", "social_score", "social_z"),
    ]:
        if val_col in activity.columns:
            agg_dict.update({
                f"{prefix}_mean": (val_col, "mean"),
                f"{prefix}_std": (val_col, "std"),
                f"{prefix}_max": (val_col, "max"),
                f"{prefix}_count": (val_col, "count"),
            })
        if z_col in activity.columns:
            agg_dict.update({
                f"{prefix}_z_mean": (z_col, "mean"),
                f"{prefix}_z_std": (z_col, "std"),
                f"{prefix}_z_max": (z_col, "max"),
            })

    activity_weekly = activity.groupby(["student_id", "week"]).agg(**agg_dict).reset_index()

    sleep_weekly = (
        sleep.groupby(["student_id", "week"])
        .agg(
            sleep_mean=("sleep_hours", "mean"),
            sleep_std=("sleep_hours", "std"),
            sleep_count=("sleep_hours", "count"),
            sleep_z_mean=("sleep_z", "mean"),
            sleep_z_std=("sleep_z", "std"),
        )
        .reset_index()
    )

    multimodal_df = (
        stress_weekly
        .merge(activity_weekly, on=["student_id", "week"], how="left")
        .merge(sleep_weekly, on=["student_id", "week"], how="left")
    )

    count_cols = [
        "stress_count", "sleep_count",
        "workload_count", "recovery_count", "social_count",
    ]
    for c in count_cols:
        if c in multimodal_df.columns:
            multimodal_df[c] = multimodal_df[c].fillna(0).astype(float)

    if "stress_count" in multimodal_df.columns:
        multimodal_df["stress_missing"] = (multimodal_df["stress_count"] == 0).astype(int)
    if "sleep_count" in multimodal_df.columns:
        multimodal_df["sleep_missing"] = (multimodal_df["sleep_count"] == 0).astype(int)
    if "workload_count" in multimodal_df.columns:
        multimodal_df["workload_missing"] = (multimodal_df["workload_count"] == 0).astype(int)
    if "recovery_count" in multimodal_df.columns:
        multimodal_df["recovery_missing"] = (multimodal_df["recovery_count"] == 0).astype(int)
    if "social_count" in multimodal_df.columns:
        multimodal_df["social_missing"] = (multimodal_df["social_count"] == 0).astype(int)

    for prefix in ["stress", "sleep", "workload", "recovery", "social"]:
        ccount = f"{prefix}_count"
        if ccount not in multimodal_df.columns:
            continue
        mask0 = multimodal_df[ccount] == 0
        for stat in ["mean", "std", "max"]:
            cstat = f"{prefix}_{stat}"
            if cstat in multimodal_df.columns:
                multimodal_df.loc[mask0, cstat] = 0.0
        for zc in [f"{prefix}_z_mean", f"{prefix}_z_std", f"{prefix}_z_max"]:
            if zc in multimodal_df.columns:
                multimodal_df.loc[mask0, zc] = 0.0

    # Apply engineered features last
    multimodal_df = add_engineered_features(multimodal_df)

    return multimodal_df


# Feature lists for Demo responsiveness (Monotonic subset)
DEMO_ALIGNED_COLS = [
    "sleep_deficit", 
    "stress_score", 
    "workload_score",
    "activity_deficit",
    "social_deficit"
]


# Feature column lists for modeling (must match notebook)
BASELINE_COLS = [
    "stress_mean", "stress_std", "stress_max", "stress_count",
    "stress_z_mean", "stress_z_std", "stress_z_max",
    "stress_mean_diff1", "stress_mean_diff2",
    "stress_z_mean_diff1", "stress_z_mean_diff2",
    "stress_mean_roll4_mean", "stress_mean_roll4_std",
    "stress_z_mean_roll4_mean", "stress_z_mean_roll4_std",
    "stress_missing",
]

STRESS_SLEEP_COLS = BASELINE_COLS + [
    "sleep_mean", "sleep_std", "sleep_count",
    "sleep_z_mean", "sleep_z_std",
    "sleep_mean_diff1", "sleep_mean_diff2",
    "sleep_z_mean_diff1", "sleep_z_mean_diff2",
    "sleep_mean_roll4_mean", "sleep_mean_roll4_std",
    "sleep_z_mean_roll4_mean", "sleep_z_mean_roll4_std",
]

MULTIMODAL_COLS = STRESS_SLEEP_COLS + [
    "workload_mean", "workload_std", "workload_max", "workload_count",
    "workload_z_mean", "workload_z_std", "workload_z_max",
    "workload_mean_diff1", "workload_mean_diff2",
    "workload_z_mean_diff1", "workload_z_mean_diff2",
    "workload_mean_roll4_mean", "workload_mean_roll4_std",
    "workload_z_mean_roll4_mean", "workload_z_mean_roll4_std",
    "recovery_mean", "recovery_std", "recovery_max", "recovery_count",
    "recovery_z_mean", "recovery_z_std", "recovery_z_max",
    "recovery_mean_diff1", "recovery_mean_diff2",
    "recovery_z_mean_diff1", "recovery_z_mean_diff2",
    "recovery_mean_roll4_mean", "recovery_mean_roll4_std",
    "recovery_z_mean_roll4_mean", "recovery_z_mean_roll4_std",
    "social_mean", "social_std", "social_max", "social_count",
    "social_z_mean", "social_z_std", "social_z_max",
    "social_mean_diff1", "social_mean_diff2",
    "social_z_mean_diff1", "social_z_mean_diff2",
    "social_mean_roll4_mean", "social_mean_roll4_std",
    "social_z_mean_roll4_mean", "social_z_mean_roll4_std",
    "workload_missing", "recovery_missing", "social_missing",
]


def get_multimodal_feature_columns(multimodal_df: pd.DataFrame) -> list[str]:
    """Return list of multimodal feature columns that exist in the DataFrame."""
    return [c for c in MULTIMODAL_COLS if c in multimodal_df.columns]
