"""This is a boilerplate pipeline 'inference' generated using Kedro 0.18.11."""
from typing import Dict

import numpy as np
import pandas as pd

from spaceship_titanic.pipelines.preprocessing.nodes import preprocess_data


def prepare_inference_data(
    data: pd.DataFrame, params: Dict, parameters: Dict
) -> pd.DataFrame:
    """Prepare the inference data.

    Args:
        data: The data to prepare.

    Returns:
        The prepared data.

    """
    inference_data = preprocess_data(data, params, parameters)
    return inference_data


def predict_inference(inference_model, inference_data) -> pd.DataFrame:
    """Predict the inference data.

    Args:
        inference_model: The model to use for prediction.
        inference_data: The data to predict.

    Returns:
        The predicted data.

    """
    features = inference_model.feature_names_in_

    X = inference_data[features]
    predictions = inference_model.predict(X)

    submission_data = pd.DataFrame(
        {"PassengerId": inference_data["PassengerId"], "Transported": predictions}
    )
    submission_data["Transported"] = np.where(
        submission_data["Transported"] == 1, True, False
    )
    return submission_data
