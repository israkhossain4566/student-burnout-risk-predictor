# Student Burnout Risk Predictor — Demo Day

Interactive demo that predicts student burnout risk from lifestyle inputs (sleep, stress, activity, workload, social) using a multimodal model trained on StudentLife-style data, with SHAP explainability.

## Setup

1. **Install dependencies** (from project root):

   ```bash
   pip install -r requirements.txt
   ```

2. **Paths** (optional): Copy `.env.example` to `.env` and set:
   - `STUDENTLIFE_PATH` — path to StudentLife data (`data/raw/studentlife` by default)
   - `MODELS_PATH` — where to save/load models (`models` by default)

## Training

From the project root, run:

```bash
python -m src.training.train
```

This will:

- Load and clean StudentLife Stress, Activity, and Sleep data from `data/raw/studentlife`
- Build weekly multimodal features and a proxy burnout target
- Run GridSearchCV for baseline (stress-only) and full multimodal models
- Save to `models/`:
  - `best_multimodal_model.pkl` — sklearn pipeline (imputer + scaler + classifier)
  - `multimodal_feature_names.json` — feature column names
  - `background_sample.csv` — sample for SHAP background
  - `metadata.json` — best model name
  - `best_models_info.txt` — summary

## Run the demo app

From the project root:

```bash
python -m streamlit run app/streamlit_app.py
```

If the `streamlit` command is on your PATH, you can instead run `streamlit run app/streamlit_app.py`. Then open the URL shown in the terminal (e.g. http://localhost:8501) in your browser.

Then open the URL shown in the terminal. Use the sliders and dropdowns to set lifestyle inputs and click **Predict Burnout Risk** to see:

- Burnout risk level (LOW / MEDIUM / HIGH) and probability
- A simple risk meter
- SHAP bar chart (top contributing factors)
- Short lifestyle recommendations

## Project layout

- `config/paths.py` — path configuration
- `src/data/` — load and clean StudentLife JSON
- `src/features/` — weekly aggregation, temporal features, target
- `src/training/` — pipelines, grid search, train script
- `src/explain/` — SHAP utilities
- `src/demo/` — input mapping (6 UI inputs → 77 features) and recommendations
- `app/streamlit_app.py` — Streamlit UI
- `notebooks/` — original research notebook
- `data/raw/studentlife/` — StudentLife Stress, Sleep, Activity folders
- `models/` — saved pipeline and artifacts (created by training)

## Note

The model uses **77 weekly/temporal features** derived from StudentLife. The demo maps **6 simple inputs** (sleep, stress, activity, workload, social; phone is informational only) to one synthetic row so the same trained pipeline can be used without retraining.
