from typing import Any
from typing import Callable
from typing import Tuple
from typing import Dict
from typing import List

import pandas as pd

from src.base import BaseModelType
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler



class ModelPipeline:
    def __init__(self, base_model: Any, features: List[str], metrics: List[Tuple[str, Callable]]):
        self.base_model = base_model
        self.features = features
        self.metrics = metrics

        self._label_encoders = []
        self._scalers = []

    def one_hot_encode(self, fold: pd.DataFrame, cat_features: List[str]) -> Tuple[pd.DataFrame, List[str]]:
        fold = fold.copy()
        reg_features = [feature for feature in self.features if feature not in cat_features]

        for cat_column in cat_features:
            dummies = pd.get_dummies(fold[cat_column], drop_first=True)
            new_col_names = [f"{cat_column}_{dummy_name}" for dummy_name in dummies.columns]
            dummies.columns = new_col_names

            fold = pd.concat([fold, dummies], axis=1)
            fold.drop(columns=cat_column, inplace=True)

            reg_features.extend(new_col_names)

        return fold, reg_features
    
    def label_encode(self, fold: pd.DataFrame, cat_features: List[str], inference: bool) -> pd.DataFrame:
        fold = fold.copy()

        if not inference:
            self._label_encoders = []
            for cat_column in cat_features:
                l_encoder = LabelEncoder()
                fold[cat_column] = l_encoder.fit_transform(fold[cat_column])
                fold[cat_column] = fold[cat_column].astype("category")
                self._label_encoders.append(l_encoder)
        else:
            for i, cat_column in enumerate(cat_features):
                l_encoder = self._label_encoders[i]
                fold[cat_column] = l_encoder.transform(fold[cat_column])
                fold[cat_column] = fold[cat_column].astype("category")

        return fold
    
    def normalize_numeric_values(self, fold: pd.DataFrame, inference: bool) -> pd.DataFrame:
        fold = fold.copy()
        
        numeric_features = [col_name for col_name in fold.columns if fold[col_name].dtypes in [int, float] and col_name != "target"]
        if not inference:
            self._scalers = []
            for num_column in numeric_features:
                s_scaler = StandardScaler()
                fold[num_column] = s_scaler.fit_transform(fold[[num_column]])
                self._scalers.append(s_scaler)
        else:
            for i, num_column in enumerate(numeric_features):
                s_scaler = self._scalers[i]
                fold[num_column] = s_scaler.transform(fold[[num_column]])

        return fold

    def fit(self, fold: pd.DataFrame) -> None:
        fold = fold.copy()

        cat_features = [col_name for col_name in self.features if fold[col_name].dtypes == "category"]
        fold = self.label_encode(fold, cat_features, inference=False)

        if isinstance(self.base_model, BaseModelType.cat.value):
            params = {**self.base_model.get_params()}
            params["cat_features"] = cat_features

            self.base_model = CatBoostClassifier(**params)
            self.base_model.fit(fold[self.features], fold["target"])

        elif isinstance(self.base_model, BaseModelType.xgb.value):
            self.base_model.fit(fold[self.features], fold["target"])

        elif isinstance(self.base_model, BaseModelType.dummy.value):
            self.base_model.fit(fold[self.features], fold["target"])

        elif isinstance(self.base_model, BaseModelType.forest.value):
            self.base_model.fit(fold[self.features], fold["target"])

        elif isinstance(self.base_model, BaseModelType.nb.value):
            fold = self.normalize_numeric_values(fold, inference=False)
            encoded_fold, encoded_features = self.one_hot_encode(fold, cat_features)
            self.base_model.fit(encoded_fold[encoded_features], fold["target"])

        elif isinstance(self.base_model, BaseModelType.knn.value):
            fold = self.normalize_numeric_values(fold, inference=False)
            encoded_fold, encoded_features = self.one_hot_encode(fold, cat_features)
            self.base_model.fit(encoded_fold[encoded_features], fold["target"])
            
        elif isinstance(self.base_model, BaseModelType.regression.value):
            fold = self.normalize_numeric_values(fold, inference=False)
            encoded_fold, encoded_features = self.one_hot_encode(fold, cat_features)
            self.base_model.fit(encoded_fold[encoded_features], fold["target"])

        else:
            raise ValueError("Not valid model!")

    def predict(self, fold: pd.DataFrame) -> pd.Series:
        fold = fold.copy()

        cat_features = [col_name for col_name in self.features if fold[col_name].dtypes == "category"]
        fold = self.label_encode(fold, cat_features, inference=True)

        if isinstance(self.base_model, BaseModelType.nb.value):
            fold = self.normalize_numeric_values(fold, inference=True)
            encoded_fold, encoded_features = self.one_hot_encode(fold, cat_features)
            result = pd.Series(self.base_model.predict_proba(encoded_fold[encoded_features])[:, 1])

        elif isinstance(self.base_model, BaseModelType.knn.value):
            fold = self.normalize_numeric_values(fold, inference=True)
            encoded_fold, encoded_features = self.one_hot_encode(fold, cat_features)
            result = pd.Series(self.base_model.predict_proba(encoded_fold[encoded_features])[:, 1])

        elif isinstance(self.base_model, BaseModelType.regression.value):
            fold = self.normalize_numeric_values(fold, inference=True)
            encoded_fold, encoded_features = self.one_hot_encode(fold, cat_features)
            result = pd.Series(self.base_model.predict_proba(encoded_fold[encoded_features])[:, 1])

        else:
            result = pd.Series(self.base_model.predict_proba(fold[self.features])[:, 1])

        return result

    def calculate_metrics(self, fold: pd.DataFrame, predictions: pd.Series) -> Dict[str, Any]:
        predictions_binary = (predictions > 0.5).astype("int32")
        
        result: Dict[str, Any] = {}
        for metric_type, metric in self.metrics:
            if metric_type == "binary":
                result[repr(metric)] = metric(y_true=fold["target"], y_pred=predictions_binary)
            elif metric_type == "score":
                result[repr(metric)] = metric(y_true=fold["target"], y_score=predictions)
            else:
                raise ValueError("Not valid metric type")

        return result
