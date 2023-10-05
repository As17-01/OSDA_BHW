from enum import Enum

from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator
from xgboost import XGBClassifier


class BaseModelType(Enum):
    """Enum for models."""

    cat = CatBoostClassifier
    xgb = XGBClassifier
    sklearn = BaseEstimator

    @classmethod
    def _missing_(cls, value):
        raise NotImplementedError(
            f"{value} is not a valid {cls.__name__}. Only {', '.join([repr(m.value) for m in cls])} types are allowed"
        )
