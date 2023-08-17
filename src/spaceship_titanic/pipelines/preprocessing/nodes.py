from typing import Dict

import numpy as np
import pandas as pd

from .utils import (
    create_age_cat_features,
    create_cabin_region,
    create_counts_from_cat_features,
)


def preprocess_data(input_data: pd.DataFrame, params: Dict, parameters: Dict):
    processed_data = input_data.copy()
    processed_data[["Cabin_Deck", "Cabin_Num", "Cabin_Side"]] = processed_data[
        "Cabin"
    ].str.split("/", expand=True)
    processed_data = processed_data.astype({"Cabin_Num": "float64"})
    processed_data = create_cabin_region(processed_data)

    processed_data["Nickname"] = processed_data["Name"].str.split(" ").str[1]
    processed_data["Group"] = (
        processed_data["PassengerId"].apply(lambda x: x.split("_")[0]).astype(int)
    )
    processed_data = processed_data.drop(columns=["Cabin", "Name"])

    processed_data = create_counts_from_cat_features(processed_data)
    processed_data = create_age_cat_features(processed_data)

    processed_data["Solo"] = (processed_data["Group_Size"] == 1).astype(int)

    processed_data = processed_data.drop(columns=params["drop_features"])

    for feature in params["filter_cat_features"].keys():
        processed_data[feature] = np.where(
            processed_data[feature].isin(params["filter_cat_features"][feature]),
            "Other",
            processed_data[feature],
        )

    return processed_data
