"""
model.py — Activity classifier.

Default: RandomForest on engineered features. Robust, fast, trains on a few
hundred labelled windows, runs real-time on a laptop CPU. This is the
pragmatic best for limited data.

For a deep-learning upgrade (CNN on Doppler spectrograms) see model_cnn.py —
but that needs far more labelled data to beat this.
"""
import os
import numpy as np

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    import joblib
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False


class ActivityClassifier:
    def __init__(self, classes):
        self.classes = list(classes)
        self.pipe = None

    def train(self, X: np.ndarray, y: np.ndarray, n_estimators: int = 300):
        if not SKLEARN_OK:
            raise RuntimeError("scikit-learn not installed. pip install scikit-learn joblib")
        self.pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=None,
                min_samples_leaf=2,
                class_weight="balanced",
                n_jobs=-1,
                random_state=42,
            )),
        ])
        self.pipe.fit(X, y)
        return self.pipe.score(X, y)

    def predict(self, x: np.ndarray):
        """Return (label, confidence)."""
        if self.pipe is None:
            return None, 0.0
        x = x.reshape(1, -1)
        proba = self.pipe.predict_proba(x)[0]
        idx = int(np.argmax(proba))
        label = self.pipe.classes_[idx]
        return label, float(proba[idx])

    def feature_importance(self):
        if self.pipe is None:
            return None
        return self.pipe.named_steps["rf"].feature_importances_

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump({"pipe": self.pipe, "classes": self.classes}, path)

    @classmethod
    def load(cls, path: str):
        if not SKLEARN_OK or not os.path.exists(path):
            return None
        blob = joblib.load(path)
        obj = cls(blob["classes"])
        obj.pipe = blob["pipe"]
        return obj
