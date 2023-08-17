"""Grid for tuning model."""
from typing import Dict, List
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer

import xgboost as xgb
from lightgbm import LGBMClassifier
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.svm import SVC


def get_tuning_grid(grid_name: str,
                    numeric_features: List,
                    categorical_features: List) -> Dict:
    """Pega o pipeline e grid de hyperparametros de um modelo.

    Args:
        grid_name (str): Nome do modelo/grid

    Returns:
        Dict: pipeline e parametros
    """
    numeric_transformer = Pipeline(steps=[
        ('imputer', KNNImputer(n_neighbors=5)),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    tuning_grids = {
        "default": {
            "pipeline": Pipeline(
                [("preprocessor", preprocessor), ("classifier", xgb.XGBClassifier())]
            ),
            "param_grid": {},
        },
        "xgboost": {
            "pipeline": Pipeline(
                [("preprocessor", preprocessor), ("classifier", xgb.XGBClassifier())]
            ),
            "param_grid": {
                "classifier__n_estimators": [100, 150, 200, 250, 300],
                "classifier__learning_rate": [0.1, 0.05],
                "classifier__gamma": [0, 0.5, 1, 1.5, 2, 5],
                "classifier__reg_alpha": [0, 0.1, 0.3, 0.5, 0.8],
                "classifier__reg_lambda": [0.2, 0.4, 0.6, 0.8, 1],
                "classifier__min_child_weight": [1, 3, 5],
                "classifier__subsample": [0.6, 0.7, 0.8, 0.9, 1],
                "classifier__colsample_bytree": [0.6, 0.7, 0.8, 0.9],
            },
        },
        "random_forest": {
            "pipeline": Pipeline(
                [("preprocessor", preprocessor),
                 ("classifier", RandomForestClassifier())]
            ),
            "param_grid": {
                "classifier__n_estimators": [350, 500, 600],
                "classifier__max_features": randint(1, 16),
                "classifier__max_depth": [None] + list(range(5, 21, 5)),
                "classifier__min_samples_split": randint(2, 15),
                "classifier__min_samples_leaf": randint(1, 15),
            },
        },
        "lightgbm": {
            "pipeline": Pipeline(
                [
                    ("preprocessor", preprocessor),
                    ("classifier", LGBMClassifier()),
                ]
            ),
            "param_grid": {
                "classifier__boosting_type": ["gbdt", "dart"],
                "classifier__n_estimators": [20, 50, 100, 200, 350, 500],
                "classifier__learning_rate": [0.05, 0.075, 0.1],
                "classifier__max_depth": [-1, 7, 9, 10, 12],
                "classifier__min_child_samples": [30, 70, 90, 100],
                "classifier__num_leaves": [75, 90, 105],
                "classifier__reg_alpha": [0, 0.2, 0.35, 0.5, 0.65, 0.85],
                "classifier__reg_lambda": [0, 0.5, 1, 2, 4, 5.5],
                "classifier__subsample": [0.7, 0.8, 0.9, 1],
            },
        },
        "logistic_regression": {
            "pipeline": Pipeline(
                [
                    ("preprocessor", preprocessor),
                    ("poly", PolynomialFeatures()),
                    ("classifier", LogisticRegression()),
                ]
            ),
            "param_grid": {
                "poly__degree": [1, 2],
                "classifier__C": [0.01, 0.1, 1, 10, 100],
                "classifier__penalty": ["elasticnet"],
                "classifier__l1_ratio": [0, 0.2, 0.4, 0.6, 0.8, 1],
                "classifier__solver": ["saga"],
                "classifier__max_iter": [10000],
            },
        },
        "support_vector": {
            "pipeline": Pipeline(
                [
                    ("preprocessor", preprocessor),
                    ("poly", PolynomialFeatures()),
                    ("classifier", SVC(probability=True)),
                ]
            ),
            "param_grid": {
                "poly__degree": [1],
                "classifier__C": [0.1, 1, 10, 100],
                "classifier__gamma": [0.1, 1, 10, 100],
                "classifier__kernel": ["linear"],
            },
        },
    }

    return tuning_grids[grid_name]
