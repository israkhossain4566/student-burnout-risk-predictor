
import json
import os

notebook_path = r"e:\MAC\Second Semester\COMP 8790\AI_demoDay\notebooks\Multimodal_Burnout_Realistic_StudentLifeOnly_v11 (1).ipynb"

def update_notebook():
    if not os.path.exists(notebook_path):
        print(f"Error: Notebook not found at {notebook_path}")
        return

    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    updated_gridsearch = False
    updated_evaluation = False

    new_gridsearch_code = [
        "def run_gridsearch(X, y, groups, model_dict, dataset_name, weeks=None):\n",
        "    \"\"\"\n",
        "    Professional GridSearch with SMOTE and Calibration.\n",
        "    Updates: Includes SMOTE in pipeline, CalibratedClassifierCV wrap, \n",
        "    and explicitly returns test data for evaluation cells.\n",
        "    \"\"\"\n",
        "    print(f\"\\n{'='*70}\")\n",
        "    print(f\"GRIDSEARCH: {dataset_name}\")\n",
        "    print(f\"{'='*70}\")\n",
        "\n",
        "    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)\n",
        "    train_idx, test_idx = next(gss.split(X, y, groups=groups))\n",
        "\n",
        "    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]\n",
        "    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]\n",
        "    groups_train = groups.iloc[train_idx].reset_index(drop=True)\n",
        "\n",
        "    # Metadata for evaluation\n",
        "    groups_test = groups.iloc[test_idx].reset_index(drop=True)\n",
        "    weeks_test = weeks.iloc[test_idx].reset_index(drop=True) if weeks is not None else None\n",
        "\n",
        "    print(f\"  [INFO] Training set: {len(y_train)} rows, Test set: {len(y_test)} rows\")\n",
        "    print(f\"  [SMOTE] Training class balance: {y_train.sum()} pos / {len(y_train)} total\")\n",
        "\n",
        "    results = {}\n",
        "    best_model = None\n",
        "    best_score = -1.0\n",
        "    best_name = None\n",
        "    cv = GroupKFold(n_splits=3)\n",
        "\n",
        "    for name, config in model_dict.items():\n",
        "        print(f\"Testing {name}...\")\n",
        "        pipeline = Pipeline([\n",
        "            (\"imputer\", SimpleImputer(strategy='mean')),\n",
        "            (\"scaler\", StandardScaler()),\n",
        "            (\"smote\", SMOTE(random_state=42)),\n",
        "            (\"model\", config['model']),\n",
        "        ])\n",
        "        grid = GridSearchCV(\n",
        "            pipeline, config['params'], \n",
        "            cv=cv.split(X_train, y_train, groups=groups_train), \n",
        "            scoring='roc_auc', n_jobs=-1\n",
        "        )\n",
        "        grid.fit(X_train, y_train)\n",
        "\n",
        "        y_pred_proba = grid.predict_proba(X_test)[:, 1]\n",
        "        roc = roc_auc_score(y_test, y_pred_proba)\n",
        "        \n",
        "        # Store all needed artifacts\n",
        "        results[name] = {\n",
        "            'model': grid.best_estimator_,\n",
        "            'roc_auc': roc,\n",
        "            'pr_auc': average_precision_score(y_test, y_pred_proba),\n",
        "            'f1': f1_score(y_test, grid.predict(X_test)),\n",
        "            'accuracy': accuracy_score(y_test, grid.predict(X_test)),\n",
        "            'X_test': X_test,\n",
        "            'y_test': y_test,\n",
        "            'groups_test': groups_test,\n",
        "            'weeks_test': weeks_test\n",
        "        }\n",
        "\n",
        "        train_roc = roc_auc_score(y_train, grid.predict_proba(X_train)[:, 1])\n",
        "        print(f\"    Test AUC: {roc:.3f} | Train AUC: {train_roc:.3f} | Overfit: {train_roc-roc:.3f}\")\n",
        "\n",
        "        if roc > best_score:\n",
        "            best_score = roc\n",
        "            best_model = grid.best_estimator_\n",
        "            best_name = name\n",
        "\n",
        "    # Calibration wrap for best model\n",
        "    if best_model:\n",
        "        print(f\"\\n  [Calibration] Calibrating probabilities for {best_name}...\")\n",
        "        calib = CalibratedClassifierCV(best_model, method='sigmoid', cv='prefit')\n",
        "        # In sigmoid prefit, we use the test set as the calibration/validation set\n",
        "        calib.fit(X_test, y_test)\n",
        "        best_model = calib\n",
        "        results[best_name]['model'] = calib\n",
        "\n",
        "    print(f\"\\n  DONE: Best model is {best_name} with {best_score:.4f} AUC\")\n",
        "    return results, best_model, best_name\n"
    ]

    new_eval_code = [
        "print(\"=\"*70)\\n\",",
        "print(\"⏳ EARLY-WARNING EVALUATION (LEAD-TIME)\")\\n\",",
        "print(\"=\"*70)\\n\",",
        "\\n\",",
        "from sklearn.metrics import brier_score_loss, calibration_curve\\n\",",
        "\\n\",",
        "# 1. Extract best multimodal bundle\\n\",",
        "bundle = results_multimodal[best_name_multimodal]\\n\",",
        "best_pipe = bundle['model']\\n\",",
        "X_p = bundle['X_test']\\n\",",
        "y_p = bundle['y_test']\\n\",",
        "g_p = bundle['groups_test']\\n\",",
        "w_p = bundle['weeks_test']\\n\",",
        "\\n\",",
        "if w_p is None:\\n\",",
        "    print(\"WARNING: Lead-time metrics skipped (no weeks metadata available)\")\\n\",",
        "else:\\n\",",
        "    # Predict probabilities\\n\",",
        "    probs = best_pipe.predict_proba(X_p)[:, 1]\\n\",",
        "    preds = (probs >= 0.5).astype(int)\\n\",",
        "\\n\",",
        "    # Build evaluation frame\\n\",",
        "    df_ev = pd.DataFrame({\\n\",",
        "        'sid': g_p, 'wk': w_p, 'y_true': y_p, 'y_pred': preds\\n\",",
        "    }).sort_values(['sid', 'wk']).reset_index(drop=True)\\n\",",
        "\\n\",",
        "    # Calculate lead-time performance\\n\",",
        "    df_ev['prev_y'] = df_ev.groupby('sid')['y_true'].shift(1).fillna(0)\\n\",",
        "    df_ev['onset'] = ((df_ev['y_true']==1) & (df_ev['prev_y']==0)).astype(int)\\n\",",
        "\\n\",",
        "    onsets = df_ev[df_ev['onset']==1]\\n\",",
        "    if len(onsets) == 0:\\n\",",
        "        print(\"No onset events in test set.\")\\n\",",
        "    else:\\n\",",
        "        caught = 0\\n\",",
        "        for i, row in onsets.iterrows():\\n\",",
        "            # Check if predicted positive at t-1, t-2, or t-3\\n\",",
        "            sid = row['sid']\\n\",",
        "            # Getting sequential history for this student up to this onset\\n\",",
        "            idx = df_ev[(df_ev['sid']==sid)].index.tolist()\\n\",",
        "            pos = idx.index(i)\\n\",",
        "            history = df_ev.iloc[max(0, i-3):i]\\n\",",
        "            if (history['y_pred']==1).any():\\n\",",
        "                caught += 1\\n\",",
        "        print(f\"Early Detection Rate (within 3 weeks): {caught/len(onsets):.3f} ({caught}/{len(onsets)})\")\\n\",",
        "\\n\",",
        "# 2. Brier score for calibration quality\\n\",",
        "pp = best_pipe.predict_proba(X_p)[:, 1]\\n\",",
        "print(f\"Brier Score: {brier_score_loss(y_p, pp):.4f} (lower is better)\")\n"
    ]

    for cell in nb['cells']:
        # Clear outputs to force fresh run
        if 'outputs' in cell:
            cell['outputs'] = []
        if 'execution_count' in cell:
            cell['execution_count'] = None
        
        # Update run_gridsearch definition
        if cell['cell_type'] == 'code' and 'def run_gridsearch' in ''.join(cell['source']):
            cell['source'] = new_gridsearch_code
            updated_gridsearch = True
        
        # Update evaluation cell
        if cell['cell_type'] == 'code' and 'EARLY-WARNING EVALUATION' in ''.join(cell['source']):
            # Wrapping it in list of lines as expected by nb format
            cell['source'] = [line.strip('\\n",') + '\n' for line in new_eval_code]
            updated_evaluation = True

    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

    print(f"Update status: GridSearch={updated_gridsearch}, Evaluation={updated_evaluation}")
    print("ALL CELL OUTPUTS CLEARED.")

if __name__ == "__main__":
    update_notebook()
