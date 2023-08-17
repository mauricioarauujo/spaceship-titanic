from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    evaluate_candidate_models,
    split_data,
    tune_candidate_models,
    gen_inference_model
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_data,
                inputs=["modeling_data", "params:model_options", "parameters"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data_node",
            ),
            node(
                func=tune_candidate_models,
                inputs=["X_train", "y_train", "params:model_options"],
                outputs="dummy",
                name="train_model_node",
            ),
            node(
                func=evaluate_candidate_models,
                inputs=["dummy", "X_test", "y_test", "params:model_options"],
                outputs="candidate_model",
                name="evaluate_model_node",
            ),
            node(
                func=gen_inference_model,
                inputs=["candidate_model", "modeling_data", "parameters"],
                outputs="model",
                name="gen_inference_model_node",
            ),
        ]
    )
