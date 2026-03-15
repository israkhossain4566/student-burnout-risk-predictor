import json
import os
from pathlib import Path

notebook_path = Path(r"e:\MAC\Second Semester\COMP 8790\AI_demoDay\notebooks\Updated.ipynb")

def finalize_notebook():
    if not notebook_path.exists():
        print(f"Error: Notebook not found at {notebook_path}")
        return

    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # 1. Setup & Config
    setup_code = [
        "# Import libraries\n",
        "import os\n",
        "import json\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "import warnings\n",
        "from pathlib import Path\n",
        "\n",
        "warnings.filterwarnings('ignore')\n",
        "\n",
        "from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "from sklearn.impute import SimpleImputer\n",
        "from imblearn.pipeline import Pipeline\n",
        "from imblearn.over_sampling import SMOTE\n",
        "from sklearn.calibration import CalibratedClassifierCV\n",
        "from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier\n",
        "from sklearn.linear_model import LogisticRegression\n",
        "from sklearn.svm import SVC\n",
        "from sklearn.metrics import classification_report, roc_auc_score, f1_score, accuracy_score, average_precision_score, brier_score_loss\n",
        "from sklearn.calibration import calibration_curve # CORRECT IMPORT\n",
        "import shap\n",
        "\n",
        "print('\u2705 Libraries loaded')\n"
    ]

    config_code = [
        "# Configuration\n",
        "RANDOM_SEED = 42\n",
        "np.random.seed(RANDOM_SEED)\n",
        "\n",
        "# Path setup\n",
        "REPO_ROOT = Path(os.getcwd()).resolve().parent\n",
        "DATA_PATH = REPO_ROOT / \"data\" / \"raw\" / \"studentlife\"\n",
        "OUTPUT_PATH = REPO_ROOT / \"models\"\n",
        "os.makedirs(OUTPUT_PATH, exist_ok=True)\n",
        "\n",
        "print(f'\u2705 Data Path: {DATA_PATH}')\n",
        "print(f'\u2705 Output Path: {OUTPUT_PATH}')\n"
    ]

    # 2. Add Engineered Features to the pipeline in the notebook
    feature_engineering_code = [
        "def add_engineered_features(df):\n",
        "    out = df.copy()\n",
        "    out['sleep_deficit'] = (8.0 - out['sleep_mean']).clip(lower=0)\n",
        "    out['stress_workload_int'] = out['stress_mean'] * out['workload_mean']\n",
        "    out['isolation_flag'] = (out['social_mean'] < 2.0).astype(int)\n",
        "    out['strain_index'] = (out['stress_mean'] + out['workload_mean']) / (out['sleep_mean'] + out['social_mean'] + 1.0)\n",
        "    return out\n"
    ]

    # 3. 3-Way Splitting & Calibration GridSearch (updated for Demo-Aligned pass)
    gridsearch_code = [
        "def run_gridsearch(X, y, groups, model_dict, dataset_name, weeks=None):\n",
        "    \"\"\"\n",
        "    Methodologically clean GridSearch with 3-way Grouped Splitting and Calibration.\n",
        "    \"\"\"\n",
        "    print(f\"\\n{'='*70}\")\n",
        "    print(f\"GRIDSEARCH: {dataset_name}\")\n",
        "    print(f\"{'='*70}\")\n",
        "\n",
        "    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)\n",
        "    temp_idx, test_idx = next(gss1.split(X, y, groups=groups))\n",
        "    X_temp, X_test = X.iloc[temp_idx], X.iloc[test_idx]\n",
        "    y_temp, y_test = y.iloc[temp_idx], y.iloc[test_idx]\n",
        "    groups_temp, groups_test = groups.iloc[temp_idx], groups.iloc[test_idx]\n",
        "\n",
        "    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)\n",
        "    train_idx, val_idx = next(gss2.split(X_temp, y_temp, groups=groups_temp))\n",
        "    X_train, X_val = X_temp.iloc[train_idx], X_temp.iloc[val_idx]\n",
        "    y_train, y_val = y_temp.iloc[train_idx], y_temp.iloc[val_idx]\n",
        "    groups_train = groups_temp.iloc[train_idx]\n",
        "\n",
        "    print(f\"  [INFO] Splits: Train={len(y_train)}, Val={len(y_val)}, Test={len(y_test)}\")\n",
        "\n",
        "    results = {}\n",
        "    best_model = None\n",
        "    best_score = -1.0\n",
        "    best_name = None\n",
        "    cv = GroupKFold(n_splits=3)\n",
        "\n",
        "    for name, config in model_dict.items():\n",
        "        pipeline = Pipeline([\n",
        "            (\"imputer\", SimpleImputer(strategy='mean')),\n",
        "            (\"scaler\", StandardScaler()),\n",
        "            (\"smote\", SMOTE(random_state=42)),\n",
        "            (\"model\", config['model']),\n",
        "        ])\n",
        "        grid = GridSearchCV(pipeline, config['params'], cv=cv.split(X_train, y_train, groups=groups_train), scoring='roc_auc', n_jobs=-1)\n",
        "        grid.fit(X_train, y_train)\n",
        "\n",
        "        y_val_proba = grid.predict_proba(X_val)[:, 1]\n",
        "        val_roc = roc_auc_score(y_val, y_val_proba) if y_val.sum() > 0 else 0.5\n",
        "        print(f\"    {name} -> Val ROC: {val_roc:.4f}\")\n",
        "        if val_roc > best_score:\n",
        "            best_score = val_roc\n",
        "            best_model = grid.best_estimator_\n",
        "            best_name = name\n",
        "\n",
        "    if best_model:\n",
        "        calib = CalibratedClassifierCV(best_model, method='sigmoid', cv='prefit')\n",
        "        calib.fit(X_val, y_val)\n",
        "        best_model = calib\n",
        "\n",
        "    y_test_proba = best_model.predict_proba(X_test)[:, 1]\n",
        "    final_roc = roc_auc_score(y_test, y_test_proba) if y_test.sum() > 0 else 0.5\n",
        "    print(f\"  --- FINAL TEST SCORE ({best_name}): {final_roc:.4f} ---\")\n",
        "\n",
        "    return {best_name: {'model': best_model, 'roc_auc': final_roc, 'X_test': X_test, 'y_test': y_test}}, best_model, best_name\n"
    ]

    for i, cell in enumerate(nb['cells']):
        if 'outputs' in cell: cell['outputs'] = []
        if 'execution_count' in cell: cell['execution_count'] = None
        
        content = "".join(cell['source'])
        if "import libraries" in content.lower():
            cell['source'] = setup_code
        elif "# Configure paths (Google Colab)" in content:
            cell['source'] = config_code
        elif "def run_gridsearch" in content:
            cell['source'] = gridsearch_code
        elif "def build_weekly_multimodal" in content:
            # Insert feature engineering function before the aggregation
            cell['source'] = feature_engineering_code + cell['source']

    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

    print("Notebook Updated Successfully.")

if __name__ == "__main__":
    finalize_notebook()
