"""Create proxy burnout_risk target (2-week-ahead high stress + poor sleep)."""
import pandas as pd

# Stress > this quantile AND Sleep < this quantile => burnout_risk = 1
# Aim for ~25% positive rate (was 15% with strict AND, 74% with OR)
STRESS_Q_CANDIDATES = [0.55, 0.60, 0.65, 0.70]
SLEEP_Q_CANDIDATES  = [0.45, 0.40, 0.35, 0.30]
TARGET_RATE = 0.25            # aim for 25%
MIN_RATE, MAX_RATE = 0.18, 0.40  # accept anything in this window


def create_burnout_target(
    multimodal_df: pd.DataFrame,
    stress_q_candidates: list[float] | None = None,
    sleep_q_candidates: list[float] | None = None,
    target_rate: float = TARGET_RATE,
    min_rate: float = MIN_RATE,
    max_rate: float = MAX_RATE,
) -> tuple[pd.DataFrame, dict]:
    """
    Add burnout_risk column (proxy: 2-week-ahead high stress AND poor sleep).
    Returns (multimodal_df with burnout_risk and future_* dropped, info dict with thresholds).
    """
    stress_q_candidates = stress_q_candidates or STRESS_Q_CANDIDATES
    sleep_q_candidates = sleep_q_candidates or SLEEP_Q_CANDIDATES

    df = multimodal_df.copy()
    df["future_stress_2w"] = df.groupby("student_id")["stress_mean"].shift(-2)
    df["future_sleep_2w"] = df.groupby("student_id")["sleep_mean"].shift(-2)
    df = df.dropna(subset=["future_stress_2w", "future_sleep_2w"]).reset_index(drop=True)

    best = None
    for sq in stress_q_candidates:
        stress_th = df["future_stress_2w"].quantile(sq)
        for lq in sleep_q_candidates:
            sleep_th = df["future_sleep_2w"].quantile(lq)
            y = (
                (df["future_stress_2w"] >= stress_th) & (df["future_sleep_2w"] <= sleep_th)
            ).astype(int)
            pos_rate = float(y.mean())
            pos_count = int(y.sum())
            in_range = min_rate <= pos_rate <= max_rate
            score = abs(pos_rate - target_rate) + (0 if in_range else 0.5)
            if best is None or score < best[0]:
                best = (score, sq, lq, float(stress_th), float(sleep_th), pos_rate, pos_count)

    _, stress_q, sleep_q, stress_th, sleep_th, pos_rate, pos_count = best
    # AND-logic: high stress AND poor sleep (cleaner signal for the model)
    df["burnout_risk"] = (
        (df["future_stress_2w"] >= stress_th) & (df["future_sleep_2w"] <= sleep_th)
    ).astype(int)
    df = df.drop(columns=["future_stress_2w", "future_sleep_2w"])

    info = {
        "stress_quantile": stress_q,
        "sleep_quantile": sleep_q,
        "stress_threshold": stress_th,
        "sleep_threshold": sleep_th,
        "positive_rate": pos_rate,
        "positive_count": pos_count,
        "n_samples": len(df),
    }
    return df, info
