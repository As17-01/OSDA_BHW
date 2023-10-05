from typing import Any
from typing import Callable
from typing import Dict
from typing import List

import pandas as pd


class ModelPipeline:
    def __init__(self, base_model: Any, metrics: List[Callable]):
        self.base_model = base_model
        self.metrics = metrics

    @staticmethod
    def _prepare_features(fold: pd.DataFrame) -> pd.DataFrame:
        feature_names = [i for i in fold.columns if i != "target"]
        features = fold[feature_names]
        return features

    def fit(self, fold: pd.DataFrame) -> None:
        features = self._prepare_features(fold)
        target = fold["target"]
        self.base_model.fit(features, target)

    def predict(self, fold: pd.DataFrame) -> pd.Series:
        features = self._prepare_features(fold)
        result = pd.Series(self.base_model.predict_proba(features)[:, 1])
        return result

    def calculate_metrics(self, fold: pd.DataFrame, predictions: pd.Series) -> Dict[str, Any]:
        result: Dict[str, Any] = {}

        for metric in self.metrics:
            result[repr(metric)] = metric(y_true=fold["target"], y_score=predictions)

        return result
