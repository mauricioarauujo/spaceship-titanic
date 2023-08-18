"""This is a boilerplate pipeline 'inference' generated using Kedro 0.18.11."""
from typing import Dict
from venv import logger

import mlflow
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


def predict_inference(inference_data: pd.DataFrame, parameters: Dict) -> pd.DataFrame:
    """Predict the inference data.

    Args:
        inference_data: The data to predict.
        parameters: Global parameters.

    Returns:
        The predicted data.

    """
    logger.info("Predicting submission data...")
    inference_model = mlflow.sklearn.load_model(
        model_uri=f"models:/{parameters['model_name']}/Production"
    )
    features = inference_model.feature_names_in_

    X = inference_data[features]
    predictions = inference_model.predict(X)

    submission_data = pd.DataFrame(
        {"PassengerId": inference_data["PassengerId"], "Transported": predictions}
    )
    submission_data["Transported"] = np.where(
        submission_data["Transported"] == 1, True, False
    )
    logger.info("Submission data predicted.")
    return submission_data
