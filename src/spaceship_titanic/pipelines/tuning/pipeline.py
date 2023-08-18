from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    compare_candidate_model,
    evaluate_candidate_models,
    split_data,
    tune_candidate_models,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_data,
                inputs=["modeling_data", "params:tuning", "parameters"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data_node",
            ),
            node(
                func=tune_candidate_models,
                inputs=["X_train", "y_train", "params:tuning"],
                outputs="dummy",  # forçando a ordem de rodagem
                name="train_model_node",
            ),
            node(
                func=evaluate_candidate_models,
                inputs=["dummy", "X_test", "y_test", "params:tuning"],
                outputs="candidate_model",  # forçando a ordem de rodagem
                name="evaluate_model_node",
            ),
            node(
                func=compare_candidate_model,
                inputs=[
                    "candidate_model",
                    "X_train",
                    "y_train",
                    "X_test",
                    "y_test",
                    "params:tuning",
                    "parameters",
                ],
                outputs=None,
                name="compare_candidate_model_node",
            ),
        ]
    )
