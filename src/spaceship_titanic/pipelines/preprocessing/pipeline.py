from kedro.pipeline import Pipeline, node, pipeline

from .nodes import preprocess_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_data,
                inputs=["train", "parameters"],
                outputs="modeling_data",
                name="create_modeling_data",
            ),
        ]
    )
