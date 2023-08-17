from kedro.pipeline import Pipeline, node, pipeline

from .nodes import gen_inference_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=gen_inference_model,
                inputs=["candidate_model", "modeling_data", "parameters"],
                outputs="model",
                name="gen_inference_model_node",
            ),
        ]
    )
