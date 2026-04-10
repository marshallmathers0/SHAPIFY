from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
from sklearn.ensemble import RandomForestClassifier


def lime_explain(
    model: RandomForestClassifier,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    plots_dir: Path,
    instance_index: int,
    random_state: int,
) -> pd.DataFrame:
    """Generate LIME explanation for the same instance used by SHAP."""
    plots_dir.mkdir(parents=True, exist_ok=True)

    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns.tolist(),
        class_names=["good_credit", "bad_credit"],
        mode="classification",
        discretize_continuous=True,
        random_state=random_state,
    )

    def predict_fn_lime(x: np.ndarray) -> np.ndarray:
        x_df = pd.DataFrame(x, columns=X_train.columns)
        return model.predict_proba(x_df)

    explanation = explainer.explain_instance(
        data_row=X_test.iloc[instance_index].values,
        predict_fn=predict_fn_lime,
        num_features=15,
        labels=(1,),
    )

    label = 1 if 1 in explanation.local_exp else next(iter(explanation.local_exp.keys()))
    lime_df = pd.DataFrame(explanation.as_list(label=label), columns=["feature", "weight"])
    lime_df["abs_weight"] = lime_df["weight"].abs()
    lime_df = lime_df.sort_values("abs_weight", ascending=False).reset_index(drop=True)

    fig = explanation.as_pyplot_figure(label=label)
    fig.set_size_inches(10, 5)
    fig.tight_layout()
    fig.savefig(plots_dir / "lime_local_explanation.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    return lime_df
