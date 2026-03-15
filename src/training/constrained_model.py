import numpy as np

class ConstrainedLogged:
    """
    A simple Logistic Regression wrapper that supports manual weights 
    (used to enforce monotonicity in the burnout demo).
    """
    def __init__(self, w, imputer, scaler):
        self.w = w
        self.imputer = imputer
        self.scaler = scaler
        self.classes_ = np.array([0, 1])
        # Sklearn-compatible attributes
        self.coef_ = np.array([w[1:]])
        self.intercept_ = np.array([w[0]])
        
    def predict_proba(self, X):
        X_i = self.imputer.transform(X)
        X_s = self.scaler.transform(X_i)
        z = self.w[0] + np.dot(X_s, self.w[1:])
        p1 = 1 / (1 + np.exp(-z))
        return np.column_stack([1-p1, p1])
