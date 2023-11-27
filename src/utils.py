from typing import Any
from typing import List

import numpy as np
import pandas as pd


def find_cat_columns(data: pd.DataFrame) -> List[str]:
    cat_columns = [col_name for col_name in data.columns if data[col_name].dtypes == "category"]
    return cat_columns


def find_num_columns(data: pd.DataFrame) -> List[str]:
    num_columns = [col_name for col_name in data.columns if data[col_name].dtypes in [int, float]]
    return num_columns


def binarize_column(data: pd.DataFrame, num_feature: str, thr: float, scaler: Any):
    data = data.copy()

    thr = np.round(thr, 3)

    data[num_feature] = scaler.inverse_transform(data[[num_feature]])

    data[num_feature] = data[num_feature] >= thr
    data.rename(columns={num_feature: f"{num_feature}>={thr}"}, inplace=True)

    return data


def one_hot_encode(data: pd.DataFrame, cat_feature: str, encoder: Any, return_bool: bool = False) -> pd.DataFrame:
    data = data.copy()

    data[cat_feature] = encoder.inverse_transform(data[cat_feature])

    dummies = pd.get_dummies(data[cat_feature], drop_first=True)
    dummies.columns = [f"{cat_feature}_{dummy_name}" for dummy_name in dummies.columns]

    if return_bool:
        for col in dummies:
            dummies[col] = dummies[col].astype(bool)

    data = pd.concat([data, dummies], axis=1)
    data.drop(columns=cat_feature, inplace=True)

    return data


def label_encode(data: pd.DataFrame, cat_feature: str, encoder: Any, is_inference: bool) -> pd.DataFrame:
    data = data.copy()

    if not is_inference:
        data[cat_feature] = encoder.fit_transform(data[cat_feature])
        data[cat_feature] = data[cat_feature].astype("category")
    else:
        data[cat_feature] = encoder.transform(data[cat_feature])
        data[cat_feature] = data[cat_feature].astype("category")

    return data


def normalize_numeric_values(data: pd.DataFrame, num_feature: str, scaler: Any, is_inference: bool) -> pd.DataFrame:
    data = data.copy()

    if not is_inference:
        data[num_feature] = scaler.fit_transform(data[[num_feature]])
    else:
        data[num_feature] = scaler.transform(data[[num_feature]])

    return data
