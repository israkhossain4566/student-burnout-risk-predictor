"""Add temporal features (diffs, rolling, _is_low) to weekly multimodal DataFrame."""
import pandas as pd


def add_temporal_features(multimodal_df: pd.DataFrame, window: int = 4) -> pd.DataFrame:
    """Add diff1, diff2, rollN mean/std, and *_is_low to multimodal_df. Modifies in place and returns."""
    df = multimodal_df.sort_values(["student_id", "week"]).reset_index(drop=True)

    base_cols = [
        c for c in [
            "stress_mean", "sleep_mean",
            "workload_mean", "recovery_mean", "social_mean",
            "stress_z_mean", "sleep_z_mean",
            "workload_z_mean", "recovery_z_mean", "social_z_mean",
        ]
        if c in df.columns
    ]

    for col in base_cols:
        df[f"{col}_diff1"] = df.groupby("student_id")[col].diff(1)
        df[f"{col}_diff2"] = df.groupby("student_id")[col].diff(2)

    for col in base_cols:
        df[f"{col}_roll{window}_mean"] = (
            df.groupby("student_id")[col]
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        df[f"{col}_roll{window}_std"] = (
            df.groupby("student_id")[col]
            .rolling(window=window, min_periods=2)
            .std()
            .reset_index(level=0, drop=True)
        )

    for c in ["stress_count", "sleep_count", "workload_count", "recovery_count", "social_count"]:
        if c in df.columns:
            df[f"{c}_is_low"] = (df[c] <= 1).astype(int)

    return df
