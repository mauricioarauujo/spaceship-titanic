from kedro.pipeline import Pipeline, node, pipeline

from .nodes import re_train_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=re_train_model,
                inputs=[
                    "X_train",
                    "y_train",
                    "X_test",
                    "y_test",
                    "params:training",
                    "parameters",
                ],
                outputs=None,
                name="re_train_model_node",
            ),
        ]
    )
