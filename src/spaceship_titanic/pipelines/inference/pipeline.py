from kedro.pipeline import Pipeline, node, pipeline

from .nodes import predict_inference, prepare_inference_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=prepare_inference_data,
                inputs=["test", "params:preprocessing", "parameters"],
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
