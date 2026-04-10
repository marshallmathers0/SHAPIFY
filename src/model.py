from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int,
) -> RandomForestClassifier:
    """Train a black-box RandomForest classifier."""
    model = RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, str]:
    """Compute model metrics for reporting."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    feature_importances = np.asarray(model.feature_importances_)

    return {
        "accuracy": f"{accuracy_score(y_test, y_pred):.4f}",
        "roc_auc": f"{roc_auc_score(y_test, y_proba):.4f}",
        "precision": f"{precision:.4f}",
        "recall": f"{recall:.4f}",
        "confusion_matrix": np.array2string(cm),
        "feature_importances": np.array2string(feature_importances, precision=6, separator=", "),
        "classification_report": classification_report(y_test, y_pred, digits=4),
    }
