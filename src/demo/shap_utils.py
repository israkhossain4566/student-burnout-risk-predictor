import numpy as np
import pandas as pd
import shap

# Map technical feature names to simple, human-readable labels
FEATURE_LABEL_MAP = {
    "sleep_mean": "Average Sleep Duration",
    "stress_mean": "Perceived Stress Level",
    "workload_mean": "Academic Workload",
    "recovery_mean": "Physical Activity Level",
    "social_mean": "Social Interaction Level",
    "sleep_deficit": "Sleep Debt (Lack of Sleep)",
    "stress_score": "Chronic Stress Score",
    "workload_score": "Heavy Coursework Pressure",
    "social_deficit": "Lack of Social Connection",
    "stress_workload_int": "Workload & Stress Synergy",
    "activity_deficit": "Physical Inactivity",
    "isolation_flag": "Social Isolation Risk",
    "strain_index": "Overall Lifestyle Strain"
}

def get_shap_explanation(pipeline, input_row):
    """
    Calculate SHAP values for a given input row using a linear explainer logic 
    (special case for our ConstrainedLogged wrapper).
    """
    # 1. Access the inner model components
    model = pipeline
    # In our case, the 'pipeline' passed from streamlit_app.py IS the ConstrainedLogged instance
    # because load_demo_artifacts returns the pkl which is the model.
    # Actually, verify if it's a pipeline or the model.
    # In streamlit_app.py: pipeline = pickle.load(f)
    
    # We need to transform the raw row through imputer and scaler first
    # because ConstrainedLogged.predict_proba does this internally, but SHAP needs the scaled features.
    X_i = model.imputer.transform(input_row)
    X_s = model.scaler.transform(X_i)
    
    # Linear SHAP: each feature's contribution is (scaled_value - baseline) * weight
    # We can use LinearExplainer with the weights
    # Note: SHAP for logistic regression can be calculated on logit or probability.
    # Logit is usually better for "additive" explanations.
    
    # We'll use a simple background (zeros in scaled space, which is the mean)
    # Since we don't have the whole training set here easily, we assume the scaler 
    # already centered data (mean=0, scale=1). So baseline is 0.
    
    weights = model.w[1:]
    shap_values = (X_s[0] - 0) * weights
    
    # Create a Series with simplified names
    df_shap = pd.DataFrame({
        "Feature": [FEATURE_LABEL_MAP.get(col, col) for col in input_row.columns],
        "Impact": shap_values
    })
    
    # Sort by absolute impact
    df_shap["AbsImpact"] = df_shap["Impact"].abs()
    df_shap = df_shap.sort_values("AbsImpact", ascending=False).drop(columns=["AbsImpact"])
    
    return df_shap
