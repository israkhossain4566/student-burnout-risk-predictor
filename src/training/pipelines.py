"""Model and GridSearch parameter definitions for burnout prediction."""
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def get_models_and_params() -> dict:
    """Return dict of model name -> {model, params} for GridSearchCV."""
    return {
        "LogReg": {
            "model": LogisticRegression(max_iter=2000, class_weight="balanced"),
            "params": {"model__C": [0.1, 1.0, 10.0]},
        },
        "RandomForest": {
            "model": RandomForestClassifier(random_state=42, class_weight="balanced"),
            "params": {
                "model__n_estimators": [200, 500],
                "model__max_depth": [5, 10, None],
                "model__min_samples_split": [2, 10],
            },
        },
        "GradientBoosting": {
            "model": GradientBoostingClassifier(random_state=42),
            "params": {
                "model__n_estimators": [100, 300],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [2, 3],
            },
        },
        "SVM": {
            "model": SVC(probability=True, class_weight="balanced"),
            "params": {"model__C": [0.5, 1.0, 2.0], "model__kernel": ["rbf"]},
        },
    }
