"""Centralized configuration and risk threshold definitions."""

RANDOM_SEED = 42

# Risk level thresholds (Stable calibrated range: ~0.10 - 0.80)
LOW_RISK_THRESHOLD = 0.40
HIGH_RISK_THRESHOLD = 0.70

def get_risk_label(prob: float) -> str:
    """Return risk category based on probability."""
    if prob < LOW_RISK_THRESHOLD:
        return "Low"
    if prob < HIGH_RISK_THRESHOLD:
        return "Moderate"
    return "High"

def get_risk_color(prob: float) -> str:
    """Return hex color for risk level visualization."""
    if prob < LOW_RISK_THRESHOLD:
        return "#2ecc71"  # Green
    if prob < HIGH_RISK_THRESHOLD:
        return "#f39c12"  # Amber/Orange
    return "#e74c3c"      # Red
