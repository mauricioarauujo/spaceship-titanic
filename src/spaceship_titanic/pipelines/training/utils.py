from typing import Dict, Tuple

import numpy as np
import pandas as pd


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
