from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class PreprocessArtifacts:
    X_train_transformed: pd.DataFrame
    X_test_transformed: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_names: List[str]


def preprocess_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    random_state: int,
) -> PreprocessArtifacts:
    """Impute, encode, scale, and split data for model training and explanations."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = X_train.select_dtypes(exclude=["object", "category"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        sparse_threshold=0,
    )

    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names_out().tolist()

    X_train_df = pd.DataFrame(X_train_t, columns=feature_names, index=X_train.index)
    X_test_df = pd.DataFrame(X_test_t, columns=feature_names, index=X_test.index)

    return PreprocessArtifacts(
        X_train_transformed=X_train_df,
        X_test_transformed=X_test_df,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
    )
