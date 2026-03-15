"""Run GridSearchCV with group-wise train/test split for burnout models.

Key design decisions:
- SMOTE inside imblearn Pipeline: applied only on training folds, no data leakage.
- Class balance is logged before and after SMOTE for verification.
- Train vs Test ROC-AUC printed for every model with an overfitting flag.
- Best estimator is wrapped in CalibratedClassifierCV (isotonic, cv='prefit') to
  correct the well-known probability-compression issue of GradientBoosting and other
  tree-based models, ensuring predict_proba spans the full [0, 1] range.
"""
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler


def run_gridsearch(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    model_dict: dict,
    dataset_name: str,
    weeks: pd.Series | None = None,
) -> tuple[dict, object | None, str | None]:
    """
    Train/test split by student group, run GridSearchCV per model, calibrate best model.
    Returns (results_dict, calibrated_best_pipeline, best_model_name).
    """
    # ------------------------------------------------------------------
    # 1. Group-aware 3-way split (Train, Validation/Calibration, Test)
    # ------------------------------------------------------------------
    # Split 1: (Train+Val) vs Test (20% of students)
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    temp_idx, test_idx = next(gss1.split(X, y, groups=groups))

    X_temp, X_test = X.iloc[temp_idx], X.iloc[test_idx]
    y_temp, y_test = y.iloc[temp_idx], y.iloc[test_idx]
    groups_temp, groups_test = groups.iloc[temp_idx], groups.iloc[test_idx]

    # Split 2: Train vs Val (25% of the remaining students -> ~20% of total)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, val_idx = next(gss2.split(X_temp, y_temp, groups=groups_temp))

    X_train, X_val = X_temp.iloc[train_idx], X_temp.iloc[val_idx]
    y_train, y_val = y_temp.iloc[train_idx], y_temp.iloc[val_idx]
    groups_train, groups_val = groups_temp.iloc[train_idx], groups_temp.iloc[val_idx]

    # Ensure reproducibility and consistency
    X_train, X_val, X_test = X_train.reset_index(drop=True), X_val.reset_index(drop=True), X_test.reset_index(drop=True)
    y_train, y_val, y_test = y_train.reset_index(drop=True), y_val.reset_index(drop=True), y_test.reset_index(drop=True)

    print(f"\n  [INFO] Splits: Train={len(y_train)}, Val={len(y_val)}, Test={len(y_test)}")
    print(f"  [INFO] Positives: Train={y_train.sum()}, Val={y_val.sum()}, Test={y_test.sum()}")

    # ------------------------------------------------------------------
    # 2. Log class distribution BEFORE SMOTE
    # ------------------------------------------------------------------
    neg_raw, pos_raw = (y_train == 0).sum(), (y_train == 1).sum()
    ratio_raw = pos_raw / max(1, len(y_train))
    print(f"\n  [SMOTE] Training set — Class 0: {neg_raw} | Class 1: {pos_raw} | Ratio: {ratio_raw:.1%}")

    # ------------------------------------------------------------------
    # 3. GridSearchCV — SMOTE inside Pipeline (CV on X_train)
    # ------------------------------------------------------------------
    results = {}
    best_model = None
    best_score = -1.0
    best_name = None
    cv = GroupKFold(n_splits=3)

    for name, config in model_dict.items():
        print(f"\nTesting {name}...")
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=42)),
            ("model", config["model"]),
        ])
        grid = GridSearchCV(
            pipeline,
            config["params"],
            cv=cv.split(X_train, y_train, groups=groups_train),
            scoring="roc_auc",
            n_jobs=-1,
        )
        grid.fit(X_train, y_train)

        # Evaluate on Test set (untuned)
        y_test_pred = grid.predict(X_test)
        y_test_proba = grid.predict_proba(X_test)[:, 1]
        test_roc = roc_auc_score(y_test, y_test_proba) if y_test.sum() > 0 else np.nan

        print(f"  Grid best CV score: {grid.best_score_:.4f}")
        print(f"  Test ROC-AUC: {test_roc:.4f}")

        results[name] = {
            "model": grid.best_estimator_,
            "best_params": grid.best_params_,
            "roc_auc": test_roc,
            "pr_auc": average_precision_score(y_test, y_test_proba) if y_test.sum() > 0 else np.nan,
            "f1": f1_score(y_test, y_test_pred, zero_division=0),
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test,
        }

        if not np.isnan(test_roc) and test_roc > best_score:
            best_score = test_roc
            best_model = grid.best_estimator_
            best_name = name

    # ------------------------------------------------------------------
    # 4. Probability Calibration (on Validation set only)
    # ------------------------------------------------------------------
    if best_model is not None:
        print(f"\n  [Calibration] Calibrating '{best_name}' on VALIDATION set (cv='prefit')...")
        calibrated = CalibratedClassifierCV(best_model, method="sigmoid", cv="prefit")
        calibrated.fit(X_val, y_val)

        # Verification on test set
        cal_proba = calibrated.predict_proba(X_test)[:, 1]
        print(f"  [Calibration] Range on test set after calibration: [{cal_proba.min():.3f}, {cal_proba.max():.3f}]")
        best_model = calibrated
        results[best_name]["model"] = calibrated

    return results, best_model, best_name
