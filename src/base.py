from enum import Enum

from catboost import CatBoostClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


class BaseModelType(Enum):
    """Enum for models."""

    cat = CatBoostClassifier
    xgb = XGBClassifier
    dummy = DummyClassifier
    forest = RandomForestClassifier
    nb = GaussianNB
    knn = KNeighborsClassifier
    regression = LogisticRegression
    tree = DecisionTreeClassifier

    @classmethod
    def _missing_(cls, value):
        raise NotImplementedError(
            f"{value} is not a valid {cls.__name__}. Only {', '.join([repr(m.value) for m in cls])} types are allowed"
        )
