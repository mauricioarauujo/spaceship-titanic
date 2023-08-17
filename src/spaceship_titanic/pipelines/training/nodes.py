import logging
from typing import Dict, Tuple
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import mlflow
# import shap
from .utils import get_train_data
from .tuning_configs import get_tuning_grid
import copy


logger = logging.getLogger(__name__)


def split_data(modeling_data: pd.DataFrame, params: Dict, parameters: Dict) -> Tuple:
    """Splits data into features and targets training and test sets.

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters/data_science.yml.
    Returns:
        Split data.
    """

    X, y = get_train_data(modeling_data, parameters)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params["test_size"], random_state=params["random_state"]
    )
    mlflow.log_param("test_size", params["test_size"])

    return X_train, X_test, y_train.to_frame(), y_test.to_frame()


def tune_candidate_models(X_train: pd.DataFrame,
                          y_train: pd.DataFrame,
                          params: Dict) -> None:
    """Tuna diferentes modelos e pega o melhor nos dados de teste.

    Args:
        X_train: Training data of independent features.
        y_train: Training data for price.

    Returns:
        None
    """

    # Definindo as etapas de pré-processamento do pipeline
    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()
    mlflow.log_param("numeric_features", numeric_features)
    mlflow.log_param("categorical_features", categorical_features)

    for grid_name in params["grid_names"]:
        logger.info(f"Tuning {grid_name} model")
        tuning_grid = get_tuning_grid(grid_name,
                                      numeric_features,
                                      categorical_features)

        param_grid = tuning_grid["param_grid"]
        pipeline = tuning_grid["pipeline"]

        # Realize o random search para encontrar os melhores hiperparâmetros
        tuning = RandomizedSearchCV(
            pipeline,
            param_distributions=param_grid,
            n_iter=params["n_iter"],
            cv=5,
            scoring=params["scoring"],
            verbose=0
        )

        tuning.fit(X_train, y_train.values.ravel())

        mlflow.log_params({grid_name: tuning.best_params_})

        # Salve o melhor modelo encontrado pelo grid search
        model = tuning.best_estimator_
        best_score = round(tuning.best_score_, 3)
        mlflow.sklearn.log_model(model, grid_name)
        mlflow.log_metric(f"{grid_name}_best_score", best_score)
        logger.info(f"{grid_name}_best_score: {best_score} \n")

    return pd.DataFrame()


def evaluate_candidate_models(
        dummy: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.Series, params: Dict
) -> Pipeline:
    """Evaluate different tuned models.

    Args:
        X_test: Testing data of independent features.
        y_test: Testing data for target.
    """
    run = mlflow.active_run()
    run_id = run.info.run_id

    model_resuts = []
    y_test = y_test.values.ravel()

    for grid_name in params["grid_names"]:

        model = mlflow.sklearn.load_model(f"runs:/{run_id}/{grid_name}")
        acc = np.round(accuracy_score(y_test, model.predict(X_test)), 3)
        auc = np.round(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]), 3)

        mlflow.log_metric(f"{grid_name}_accuracy", acc)
        mlflow.log_metric(f"{grid_name}_auc_score", auc)

        logger.info(f"{grid_name}_accuracy: {acc}")
        logger.info(f"{grid_name}_auc_score:{auc}")

        model_resuts.append(auc)
        try:
            feature_importances = pd.Series(
                model['classifier'].feature_importances_,
                model[:-1].get_feature_names_out()).sort_values(ascending=True)
        except AttributeError:
            feature_importances = pd.Series(
                model['classifier'].coef_[0],
                model[:-1].get_feature_names_out()).sort_values(ascending=True)

        # Crie um gráfico de barras das importâncias das features
        plt.figure(figsize=(10, 18))
        plt.barh(feature_importances.index, feature_importances.values)
        plt.xlabel('Importância')
        plt.ylabel('Feature')
        plt.title(f'Importância das Features para o Modelo {grid_name}')
        plt.tight_layout()

        # Salve o gráfico como uma imagem temporária
        plot_file = f"feature_importances_{grid_name}.png"
        plt.savefig(plot_file)
        plt.close()  # Certifique-se de fechar o gráfico para liberar memória
        mlflow.log_artifact(plot_file)
        os.remove(plot_file)
    # explainer = shap.Explainer(model[-1])
    # shap_values = explainer(model[:-1].transform(X_test))
    # fig = shap.summary_plot(shap_values, model[:-1].transform(X_test))
    # mlflow.log_figure(fig, "shap_summary_plot.png")

    best_model_index = np.argmax(model_resuts)
    best_model_name = params["grid_names"][best_model_index]
    best_candidate_model = mlflow.sklearn.load_model(
        f"runs:/{run_id}/{best_model_name}")
    acc = np.round(accuracy_score(y_test, best_candidate_model.predict(X_test)), 3)
    auc = np.round(roc_auc_score(
        y_test, best_candidate_model.predict_proba(X_test)[:, 1]), 3)

    mlflow.log_param("best_model", best_model_name)
    mlflow.log_metric("best_model_accuracy", acc)
    mlflow.log_metric("best_model_auc_score", auc)

    logger.info(f"Best model: {best_model_name}")
    logger.info(f"Best model score: {model_resuts[best_model_index]}")

    return best_candidate_model


def gen_inference_model(
        candidate_model: Pipeline,
        modeling_data: pd.DataFrame,
        parameters: Dict) -> Pipeline:
    """Treina o modelo com todos os dados para realizar inferência."""
    model = copy.deepcopy(candidate_model)
    features = model.feature_names_in_

    X = modeling_data[features]
    y = modeling_data[parameters["col_maps"]["TARGET_COL"]]

    model.fit(X, y)

    return model
