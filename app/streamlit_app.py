import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd

from config.paths import MODELS_PATH
from src.config import get_risk_label, get_risk_color
from src.demo.input_mapping import user_inputs_to_feature_row
from src.training.constrained_model import ConstrainedLogged

st.set_page_config(page_title="AI Student Burnout Risk Predictor", layout="wide")

@st.cache_resource
def load_demo_artifacts():
    model_path = MODELS_PATH / "best_demo_model.pkl"
    names_path = MODELS_PATH / "demo_feature_names.json"
    meta_path  = MODELS_PATH / "metadata.json"
    
    if not model_path.exists() or not names_path.exists():
        return None, None, "Unknown"
        
    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)
    with open(names_path) as f:
        feature_names = json.load(f)
        
    model_name = "LogisticRegression"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
            model_name = meta.get("best_demo_name", model_name)
            
    return pipeline, feature_names, model_name

def get_risk_drivers(inputs: dict) -> list:
    """Identify main risk drivers based on lifestyle thresholds."""
    drivers = []
    if inputs["sleep_hours"] < 6.0:
        drivers.append(f"Severe Sleep Deficit ({inputs['sleep_hours']}h/night)")
    if inputs["stress_level"] == "High":
        drivers.append("High Perceived Stress Levels")
    if inputs["academic_workload"] == "High":
        drivers.append("Heavy Academic Workload")
    if inputs["physical_activity_min"] < 20:
        drivers.append(f"Low Physical Recovery Activity ({inputs['physical_activity_min']} min)")
    if inputs["social_interaction"] == "Low":
        drivers.append("Social Isolation / Low Interaction")
    return drivers

def main():
    st.title("AI Student Burnout Risk Predictor")
    st.markdown(
        "Predict your burnout risk based on lifestyle patterns. "
        "This model is tuned to respond meaningfully to your current lifestyle choices."
    )

    pipeline, feature_names, model_name = load_demo_artifacts()
    
    if pipeline is None:
        st.error(f"Model artifacts not found. Please run training first.")
        return

    st.subheader("Interactive Lifestyle Inputs")
    st.markdown("Adjust the sliders and dropdowns to see how your habits affect your predicted risk.")
    
    col1, col2 = st.columns(2)
    with col1:
        sleep_hours = st.slider("Sleep duration (hours per night)", 3.0, 10.0, 7.0, step=0.5)
        stress_level = st.selectbox("Current Stress level", ["Low", "Medium", "High"], index=1)
        physical_activity_min = st.slider("Daily physical activity (minutes)", 0, 120, 30)
    with col2:
        academic_workload = st.selectbox("Academic workload", ["Low", "Medium", "High"], index=1)
        social_interaction = st.selectbox("Social interaction level", ["Low", "Medium", "High"], index=1)

    if st.button("Calculate Burnout Risk Score", type="primary"):
        with st.spinner("Analyzing lifestyle patterns..."):
            # 1. Map inputs
            row = user_inputs_to_feature_row(
                sleep_hours=sleep_hours,
                stress_level=stress_level,
                physical_activity_min=physical_activity_min,
                academic_workload=academic_workload,
                social_interaction=social_interaction,
                feature_names=feature_names
            )
            
            # 2. Prediction
            prob = float(pipeline.predict_proba(row)[0, 1])
            level = get_risk_label(prob)
            color = get_risk_color(prob)
            
            inputs_dict = {
                "sleep_hours": sleep_hours, "stress_level": stress_level, "physical_activity_min": physical_activity_min,
                "academic_workload": academic_workload, "social_interaction": social_interaction
            }
            drivers = get_risk_drivers(inputs_dict)

        from src.config import LOW_RISK_THRESHOLD, HIGH_RISK_THRESHOLD

        # 4. Display Result Card
        st.subheader("Risk Assessment Result")
        rcol1, rcol2 = st.columns([1, 2])
        with rcol1:
            st.markdown(
                f"""
                <div style="padding:1.5rem;border-radius:1rem;
                            background:linear-gradient(135deg,{color}22,{color}11);
                            border:2px solid {color};text-align:center;box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                  <p style="margin:0;font-size:0.9rem;color:#888;letter-spacing:0.15em;font-weight:bold;">
                    ANNUALIZED RISK PROXIMITY
                  </p>
                  <h1 style="margin:0.2rem 0;color:{color};font-size:3.2rem;font-weight:900;text-transform:uppercase;">
                    {level}
                  </h1>
                  <p style="margin:0;font-size:1.6rem;color:{color};font-weight:700;">
                    {(prob * 100):.1f}% Prob.
                  </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with rcol2:
            st.markdown(f"**Confidence Meter ({level} Risk Band)**")
            st.progress(prob)
            
            if prob >= HIGH_RISK_THRESHOLD:
                st.error("**High Strain Alert**: Significant burnout-risk patterns detected. Immediate lifestyle intervention is recommended.")
            elif prob >= LOW_RISK_THRESHOLD:
                st.warning("**Warning**: Moderate strain detected. Consider proactive adjustments to sleep or workload.")
            else:
                if drivers:
                    st.warning("**Mixed Signal**: While your overall statistical risk is currently 'Low', we have detected specific high-strain factors (see below) that could accumulate.")
                else:
                    st.success("**Stable Pattern**: Your current lifestyle metrics suggest high resilience and low burnout risk.")
            
            if drivers:
                st.markdown("### Key Risk Drivers")
                for d in drivers:
                    st.markdown(f"- **{d}**")

            # Final Demo-Day Information Sections (Appears only after prediction)
            st.divider()
            
            inf_col1, inf_col2 = st.columns(2)
            
            with inf_col1:
                st.subheader("Model Information")
                st.markdown(
                    f"""
                    **Model Type**:  
                    Logistic Regression
                    
                    *Note: This architecture was selected as it **outperformed** other candidates (Random Forest, SVM) in providing logically consistent and reliable predictions for this domain.*
        
                    **Dataset**:  
                    StudentLife behavioral dataset
        
                    **Primary Features Used**:
                    - Sleep deficit
                    - Perceived stress level
                    - Academic workload
                    - Physical activity deficit
                    - Social interaction deficit
                    """
                )
        
            with inf_col2:
                st.subheader("How the Model Works")
                st.info(
                    "The model estimates burnout risk by analyzing behavioral patterns "
                    "linked to academic strain. Worsening lifestyle factors (e.g., lower sleep, "
                    "higher stress, isolation) monotonically increase the predicted risk score.\n\n"
                    "By enforcing monotonicity, we ensure the system remains scientifically honest "
                    "and logically predictable for real-world scenarios."
                )

    st.divider()
    st.caption(
        "**Disclaimer**: This tool is an academic demonstration developed as part of a University project. "
        "It uses a proxy model calibrated on StudentLife behavioral data and does NOT provide medical or psychological diagnosis. "
        #"If you are feeling overwhelmed, please contact University student support services."
    )

if __name__ == "__main__":
    main()
