"""Generate simple lifestyle recommendations from SHAP feature names and values."""
from typing import List, Tuple


def get_recommendations_from_shap(
    feature_names: list[str],
    shap_values: list[float],
    top_k: int = 5,
) -> list[str]:
    """
    Map high positive-SHAP features (increase burnout risk) to canned tips.
    Returns a list of short recommendation strings.
    """
    # Features that push risk up (positive SHAP) -> suggest reducing or improving
    name_val: list[tuple[str, float]] = list(zip(feature_names, shap_values))
    name_val.sort(key=lambda x: -x[1])  # highest positive first

    tips = []
    seen = set()
    for name, val in name_val:
        if val <= 0 or len(tips) >= top_k:
            continue
        tip = _feature_to_tip(name, val)
        if tip and tip not in seen:
            tips.append(tip)
            seen.add(tip)
    return tips


def _feature_to_tip(feature_name: str, _shap_val: float) -> str | None:
    """One feature -> one recommendation string."""
    if "stress" in feature_name:
        return "Reduce stress (e.g. breaks, mindfulness, workload boundaries)"
    if "sleep" in feature_name:
        return "Aim for at least 7 hours of sleep per night"
    if "recovery" in feature_name or "relax" in feature_name:
        return "Add more time for relaxation and recovery"
    if "social" in feature_name:
        return "Increase social connection and support"
    if "workload" in feature_name:
        return "Balance academic workload and avoid overload"
    return None
