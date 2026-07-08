"""
train.py — Train the activity classifier from collected data.

Usage:
    python train.py

Loads every dataset/<label>.npz, trains a RandomForest, reports
cross-validated accuracy and a confusion matrix, and saves the model.
"""
import os
import glob
import numpy as np

from config import CONFIG
from features import FEATURE_NAMES
from model import ActivityClassifier, SKLEARN_OK


def load_dataset():
    X, y = [], []
    files = sorted(glob.glob(os.path.join(CONFIG.data_dir, "*.npz")))
    if not files:
        raise FileNotFoundError(
            f"No data in {CONFIG.data_dir}. Run collect.py per class first."
        )
    for f in files:
        d = np.load(f, allow_pickle=True)
        Xi = d["X"]
        label = str(d["label"])
        X.append(Xi)
        y += [label] * Xi.shape[0]
        print(f"[data] {label:15s}: {Xi.shape[0]} windows")
    return np.vstack(X), np.array(y)


def main():
    if not SKLEARN_OK:
        print("Install scikit-learn: pip install scikit-learn joblib")
        return

    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.metrics import confusion_matrix

    X, y = load_dataset()
    classes = sorted(set(y))
    print(f"\n[train] {X.shape[0]} windows, {X.shape[1]} features, {len(classes)} classes")

    clf = ActivityClassifier(classes)

    # Cross-validated accuracy (honest estimate)
    if X.shape[0] >= 20:
        skf = StratifiedKFold(n_splits=min(5, np.min(np.bincount(
            [classes.index(v) for v in y]))), shuffle=True, random_state=42)
        # build a fresh pipeline for CV
        clf.train(X, y)  # fit once to get a pipeline template
        scores = cross_val_score(clf.pipe, X, y, cv=skf, n_jobs=-1)
        print(f"[train] CV accuracy: {scores.mean():.3f} +/- {scores.std():.3f}")

    train_acc = clf.train(X, y)
    print(f"[train] train accuracy: {train_acc:.3f}")

    # Confusion matrix on training data (indicative)
    preds = clf.pipe.predict(X)
    cm = confusion_matrix(y, preds, labels=classes)
    print("\nConfusion matrix (rows=true, cols=pred):")
    print("        " + "  ".join(f"{c[:6]:>6}" for c in classes))
    for i, c in enumerate(classes):
        print(f"{c[:7]:>7} " + "  ".join(f"{cm[i, j]:>6}" for j in range(len(classes))))

    # Feature importance
    imp = clf.feature_importance()
    if imp is not None:
        order = np.argsort(imp)[::-1]
        print("\nTop features:")
        for i in order[:6]:
            print(f"  {FEATURE_NAMES[i]:16s} {imp[i]:.3f}")

    clf.save(CONFIG.model_path)
    print(f"\n[train] model saved -> {CONFIG.model_path}")


if __name__ == "__main__":
    main()
