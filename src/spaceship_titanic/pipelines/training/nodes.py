import copy
import logging
from typing import Dict

import mlflow
import pandas as pd
from mlflow import MlflowClient
from sklearn.metrics import accuracy_score

logger = logging.getLogger(__name__)


def re_train_model(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    params: Dict,
    parameters: Dict,
) -> None:
    """Treina o modelo com todos os dados para realizar inferência.

    Args:
        modeling_data: Dados de treino e teste.
        X_train: Dados de treino de features.
        y_train: Dados de treino de target.
        X_test: Dados de teste de features.
        y_test: Dados de teste de target.
        parameters: Parâmetros

    Returns:
        None
    """
    client = MlflowClient()

    production_model = mlflow.sklearn.load_model(
        model_uri=f"models:/{parameters['model_name']}/Production"
    )
    logger.info(f"Production model type: {type(production_model)}")

    model_input_features = production_model.feature_names_in_
    features_out_dataset = [f for f in model_input_features if f not in X_train.columns]
    if features_out_dataset:
        raise ValueError(f"Features {features_out_dataset} not in dataset")

    retrained_model = copy.deepcopy(production_model)
    retrained_model.fit(X_train, y_train)
    retrained_model_score = round(
        accuracy_score(y_test, retrained_model.predict(X_test)), 4
    )
    production_model_score = round(
        accuracy_score(y_test, production_model.predict(X_test)), 4
    )

    logger.info(f"Retrained model score: {retrained_model_score}")
    logger.info(f"Production model score: {production_model_score}")

    if retrained_model_score > production_model_score * params["min_accuracy_increase"]:
        improvement_percent = round(
            (retrained_model_score - production_model_score) / production_model_score, 3
        )
        logger.info(
            f"Retrained model is better than production by {improvement_percent}%"
        )
        mlflow.log_metric("retraining_pct_model_improvement", improvement_percent)

        X = pd.concat([X_train, X_test])
        y = pd.concat([y_train, y_test]).values.ravel()

        retrained_model.fit(X, y)
        mlflow.sklearn.log_model(
            artifact_path="Production",
            sk_model=retrained_model,
            registered_model_name=parameters["model_name"],
        )
        client.transition_model_version_stage(
            name=parameters["model_name"],
            version=client.get_latest_versions(parameters["model_name"])[0].version,
            stage="Production",
        )
    else:
        logger.info("Retrained model is not better than production")

    return
