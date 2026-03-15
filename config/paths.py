"""Centralized path configuration for the Burnout Demo project."""
import os
from pathlib import Path

# Project root (directory containing config/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Optional: load from environment
def _get_env_path(key: str, default: str) -> Path:
    value = os.environ.get(key)
    if value:
        p = Path(value)
        return p if p.is_absolute() else PROJECT_ROOT / p
    return PROJECT_ROOT / default


STUDENTLIFE_PATH = _get_env_path("STUDENTLIFE_PATH", "data/raw/studentlife")
MODELS_PATH = _get_env_path("MODELS_PATH", "models")

# Ensure models dir exists when we write to it
def ensure_models_dir() -> None:
    MODELS_PATH.mkdir(parents=True, exist_ok=True)
