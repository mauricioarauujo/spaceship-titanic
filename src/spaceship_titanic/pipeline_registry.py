"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline

import spaceship_titanic.pipelines.inference as inference
import spaceship_titanic.pipelines.preprocessing as preprocessing
import spaceship_titanic.pipelines.training as training


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """

    return {
        "__default__": preprocessing.create_pipeline() + training.create_pipeline()
        # + inference.create_pipeline()
        ,
        "pp": preprocessing.create_pipeline(),
        "tp": training.create_pipeline(),
        "ip": inference.create_pipeline(),
    }
