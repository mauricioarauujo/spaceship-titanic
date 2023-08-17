import pandas as pd


def gen_counts_per_cat_col(
    df: pd.DataFrame, cat_col: str, feature_name: str
) -> pd.DataFrame:
    """Gera contagem por coluna categorica.

    Args:
        df (pd.DataFrame): dataset a ser modificado.
        cat_col (str): coluna categorial alvo.
        feature_name (str): nome final da feature.

    Returns:
        pd.DataFrame: dataset com a nova feature
    """

    df_count = df.groupby(cat_col).size().reset_index(name=feature_name)

    df = df.merge(df_count, on=[cat_col], how="left")

    return df


def create_counts_from_cat_features(input_df: pd.DataFrame) -> pd.DataFrame:
    """Cria features numericas a partir de categoricas."""
    cat_counts_params = {
        "Cabin_Num": {"Feature_Name": "People_in_Cabin_Num", "remove_col": True},
        "Cabin_Deck": {"Feature_Name": "People_in_Cabin_Deck", "remove_col": False},
        "Nickname": {"Feature_Name": "Family_Size", "remove_col": True},
        "Group": {"Feature_Name": "Group_Size", "remove_col": True},
    }

    df = input_df.copy()
    for col in list(cat_counts_params.keys()):

        df = gen_counts_per_cat_col(df, col, cat_counts_params[col]["Feature_Name"])
        if cat_counts_params[col]["remove_col"]:
            df = df.drop(columns=[col])

    return df


def create_age_cat_features(df_input: pd.DataFrame) -> pd.DataFrame:
    """Cria colunas categoricas a partir da idade."""
    df = df_input.copy()
    df["Age_Cat"] = pd.cut(
        df["Age"],
        bins=[0, 12, 18, 25, 50, 200],
        labels=["Child", "Teenager", "Pre_Adult", "Adult", "Elder"],
    )
    df = df.drop(columns=["Age"])

    return df


def create_expenditure_features(df_input: pd.DataFrame) -> pd.DataFrame:
    """Cria features de gasto.

    Args:
        df_input (pd.DataFrame): dataframe de entrada

    Returns:
        pd.DataFrame: dataframe com as novas features
    """
    df = df_input.copy()
    exp_feats = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    df["Expenditure"] = df[exp_feats].sum(axis=1)
    df["No_spending"] = (df["Expenditure"] == 0).astype(int)

    return df


def create_cabin_region(df_input: pd.DataFrame) -> pd.DataFrame:
    """Cria coluna de região do navio baseado na cabine.

    Args:
        df_input (pd.DataFrame): dataframe de entrada

    Returns:
        pd.DataFrame: dataframe com a nova feature
    """

    def _return_cabin_region(cabin_num: int) -> str:
        """Retorna a região da cabine."""
        if cabin_num < 300:
            return "A"
        elif cabin_num < 600:
            return "B"
        elif cabin_num < 900:
            return "C"
        elif cabin_num < 1200:
            return "D"
        elif cabin_num < 1500:
            return "E"
        elif cabin_num < 1800:
            return "F"
        else:
            return "G"

    df = df_input.copy()

    df["Cabin_Region"] = df["Cabin_Num"].apply(_return_cabin_region)

    return df
