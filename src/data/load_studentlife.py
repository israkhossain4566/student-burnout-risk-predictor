"""Load StudentLife JSON data from Stress, Activity, and Sleep folders."""
import json
import os
from pathlib import Path

import pandas as pd


def load_studentlife_json(folder_path: str | Path) -> pd.DataFrame:
    """Load all JSON files from a StudentLife modality folder (e.g. Stress, Activity, Sleep).

    Each file is named like Stress_u00.json; student_id is derived from the filename.
    """
    folder_path = Path(folder_path)
    all_data = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".json"):
            student_id = file_name.replace(".json", "")
            with open(folder_path / file_name, "r", encoding="utf-8") as f:
                records = json.load(f)
                for r in records:
                    r = r.copy()
                    r["student_id"] = student_id
                    all_data.append(r)
    return pd.DataFrame(all_data)


def load_all_studentlife(base_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load Stress, Activity, and Sleep DataFrames from StudentLife base path."""
    base_path = Path(base_path)
    stress = load_studentlife_json(base_path / "Stress")
    activity = load_studentlife_json(base_path / "Activity")
    sleep = load_studentlife_json(base_path / "Sleep")
    return stress, activity, sleep
