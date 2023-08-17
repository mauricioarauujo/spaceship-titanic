from kedro.pipeline import Pipeline, node, pipeline

from .nodes import prepare_inference_data, predict_inference


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=prepare_inference_data,
                inputs=["test", "parameters"],
                outputs="inference_data",
                name="prepare_inference_data_node",
            ),
            node(
                func=predict_inference,
                inputs=["model", "inference_data"],
                outputs="submission_data",
                name="predict_inference_node",
            ),
        ]
    )
