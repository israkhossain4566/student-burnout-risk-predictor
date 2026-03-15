"""SHAP explainability for burnout model: local values and bar chart figure."""
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

# Model type hints for explainer selection
TREE_MODELS = ("RandomForest", "XGBoost", "GradientBoosting")


def _get_pipe(pipeline):
    """Extract Pipeline from CalibratedClassifierCV if wrapped."""
    if hasattr(pipeline, "estimator"):
        return pipeline.estimator
    if hasattr(pipeline, "base_estimator"):
        return pipeline.base_estimator
    return pipeline


def _get_transformed(pipeline, X: pd.DataFrame) -> np.ndarray:
    """Run imputer then scaler on X."""
    pipe = _get_pipe(pipeline)
    imputer = pipe.named_steps["imputer"]
    scaler = pipe.named_steps["scaler"]
    return scaler.transform(imputer.transform(X))


def get_local_shap_values(
    pipeline,
    X_background: pd.DataFrame,
    x_single: pd.DataFrame,
    feature_names: list[str],
    model_name: str = "GradientBoosting",
) -> tuple[float, np.ndarray]:
    """
    Compute SHAP values for a single row. Uses TreeExplainer for tree models,
    LinearExplainer for LogReg, KernelExplainer otherwise.
    Returns (base_value, shap_values_1d).
    """
    pipe = _get_pipe(pipeline)
    model = pipe.named_steps["model"]
    X_bg_trans = _get_transformed(pipeline, X_background)
    x_trans = _get_transformed(pipeline, x_single)
    x_1d = x_trans[0]

    if model_name in TREE_MODELS:
        explainer = shap.TreeExplainer(
            model, data=X_bg_trans, feature_perturbation="interventional"
        )
        shap_vals = explainer.shap_values(x_trans, check_additivity=False)
    elif model_name == "LogReg":
        explainer = shap.LinearExplainer(model, X_bg_trans)
        shap_vals = explainer.shap_values(x_trans)
    else:
        def predict_proba_class1(x: np.ndarray) -> np.ndarray:
            return model.predict_proba(x)[:, 1]
        explainer = shap.KernelExplainer(predict_proba_class1, X_bg_trans)
        shap_vals = explainer.shap_values(x_trans)

    if isinstance(shap_vals, list):
        shap_1d = shap_vals[1][0] if len(shap_vals) > 1 else shap_vals[0][0]
    else:
        shap_1d = shap_vals[0]

    base_val = getattr(explainer, "expected_value", 0)
    if isinstance(base_val, (list, np.ndarray)):
        base_val = base_val[1] if len(base_val) > 1 else base_val[0]
    base_val = float(base_val)

    return base_val, np.asarray(shap_1d)


def _humanize_feature(name: str) -> str:
    """Convert statistical column names back to clear human-readable factors."""
    mapping = {
        "stress": "Stress Level",
        "sleep": "Sleep Duration",
        "workload": "Academic Workload",
        "recovery": "Physical Activity / Recovery",
        "social": "Social Interaction"
    }

    prefix = name.split("_")[0]
    base = mapping.get(prefix, name.replace("_", " ").title())
    
    if "roll4_mean" in name:
        return f"{base} (Recent Avg)"
    if "roll4_std" in name:
        return f"{base} (Recent Fluctuation)"
    if "diff" in name:
        return f"{base} (Recent Change)"
    if "z_mean" in name:
        return f"{base} (Relative to Peers)"
    if "missing" in name:
        return f"{base} (Missing Data)"
    if "std" in name[len(prefix):] and "roll" not in name:
        return f"{base} (Daily Fluctuation)"
    if "max" in name:
        return f"{base} (Maximum)"
        
    return base


def get_local_shap_bar_figure(
    feature_names: list[str],
    shap_1d: np.ndarray,
    top_k: int = 15,
    title: str = "Top contributing factors (SHAP)",
) -> "matplotlib.figure.Figure":
    """Build a horizontal bar chart of |SHAP| for top_k features. Returns matplotlib Figure."""
    import matplotlib.pyplot as plt

    order = np.argsort(np.abs(shap_1d))[::-1][:top_k]
    names = [_humanize_feature(feature_names[i]) for i in order]
    vals = shap_1d[order]
    colors = ["#e74c3c" if v > 0 else "#3498db" for v in vals]

    fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.35)))
    ax.barh(range(len(names)), vals, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("SHAP value (impact on burnout risk)")
    ax.set_title(title)
    ax.axvline(0, color="gray", linewidth=0.8)
    plt.tight_layout()
    return fig
