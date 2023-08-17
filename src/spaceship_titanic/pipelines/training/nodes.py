import copy
import logging
from typing import Dict

import pandas as pd
from sklearn.pipeline import Pipeline

from spaceship_titanic.pipelines.utils import get_train_data

logger = logging.getLogger(__name__)


def gen_inference_model(
    candidate_model: Pipeline, modeling_data: pd.DataFrame, parameters: Dict
) -> Pipeline:
    """Treina o modelo com todos os dados para realizar inferência."""
    X, y = get_train_data(modeling_data, parameters)

    model = copy.deepcopy(candidate_model)

    model_input_features = model.feature_names_in_
    features_out_dataset = [f for f in model_input_features if f not in X.columns]
    if features_out_dataset:
        raise ValueError(f"Features {features_out_dataset} not in dataset")

    model.fit(X, y)

    return model
