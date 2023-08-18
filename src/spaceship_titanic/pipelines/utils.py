import copy
from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def get_train_data(
    modeling_data: pd.DataFrame, parameters: Dict
) -> Tuple[pd.DataFrame, pd.Series]:
    """Cria dados de treino e teste para o modelo"""
    TARGET_COL = parameters["col_maps"]["TARGET_COL"]
    modeling_data[TARGET_COL] = np.where(modeling_data[TARGET_COL], 1, 0)
    features = modeling_data.columns.difference(
        [parameters["col_maps"]["ID_COL"], TARGET_COL]
    )

    X = modeling_data[features]
    y = modeling_data[TARGET_COL]

    return X, y


def compare_models(
    candidate_model_input: Pipeline,
    model_input: Pipeline,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    scoring: Callable,
) -> bool:
    """Compara o desempenho do modelo candidato com o modelo atual.

    Args:
        candidate_model (Pipeline): _description_
        model (Pipeline): _description_

    Returns:
        bool: _description_
    """
    candidate_model = copy.deepcopy(candidate_model_input)
    model = copy.deepcopy(model_input)

    candidate_model.fit(X_train, y_train)
    model.fit(X_train, y_train)

    candidate_model_score = scoring(y_test, candidate_model.predict(X_test))
    model_score = scoring(y_test, model.predict(X_test))

    if candidate_model_score > model_score:
        return True
    return False
