import pathlib
import pickle
import tempfile
import zipfile
from typing import Any
from typing import Dict
from typing import List

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from src.base import BaseModelType
from src.utils import binarize_column
from src.utils import find_cat_columns
from src.utils import find_num_columns
from src.utils import label_encode
from src.utils import normalize_numeric_values
from src.utils import one_hot_encode


class Pipeline:
    def __init__(self, base_model: Any):
        self.base_model = base_model
        self.encoders = []
        self.scalers = []

    def fit_model(self, data: pd.DataFrame, target: pd.Series, cat_columns: List[str], num_columns: List[str]):
        if isinstance(self.base_model, BaseModelType.cat.value):
            params = {**self.base_model.get_params()}
            params["cat_features"] = cat_columns

            self.base_model = CatBoostClassifier(**params)
            self.base_model.fit(data, target)

        elif isinstance(self.base_model, BaseModelType.xgb.value):
            self.base_model.fit(data, target)

        elif isinstance(self.base_model, BaseModelType.dummy.value):
            self.base_model.fit(data, target)

        elif isinstance(self.base_model, BaseModelType.forest.value):
            self.base_model.fit(data, target)

        elif isinstance(self.base_model, BaseModelType.nb.value):
            for i, col_name in enumerate(cat_columns):
                data = one_hot_encode(data, col_name, encoder=self.encoders[i])
            self.base_model.fit(data, target)

        elif isinstance(self.base_model, BaseModelType.knn.value):
            for i, col_name in enumerate(cat_columns):
                data = one_hot_encode(data, col_name, encoder=self.encoders[i])
            self.base_model.fit(data, target)

        elif isinstance(self.base_model, BaseModelType.regression.value):
            for i, col_name in enumerate(cat_columns):
                data = one_hot_encode(data, col_name, encoder=self.encoders[i])
            self.base_model.fit(data, target)

        elif isinstance(self.base_model, BaseModelType.tree.value):
            self.base_model.fit(data, target)

        elif isinstance(self.base_model, BaseModelType.formal_concept.value):
            # I know that it is a horrible practice, but whatever
            data = data[self.base_model.features].copy()

            for i, col_name in enumerate(num_columns):
                if col_name in self.base_model.features:
                    data = binarize_column(
                        data=data, num_feature=col_name, thr=self.base_model.thr_dict[col_name], scaler=self.scalers[i]
                    )
            for i, col_name in enumerate(cat_columns):
                if col_name in self.base_model.features:
                    data = one_hot_encode(data, col_name, encoder=self.encoders[i], return_bool=True)

            data.reset_index(inplace=True)
            data["index"] = data["index"].astype("str")
            data.set_index("index", inplace=True)

            self.base_model.fit(data, target)

        else:
            raise ValueError("Not valid model!")

    def predict_proba_model(self, data: pd.DataFrame, cat_columns: List[str], num_columns: List[str]) -> np.ndarray:
        if isinstance(self.base_model, BaseModelType.cat.value):
            predictions = self.base_model.predict_proba(data)

        elif isinstance(self.base_model, BaseModelType.xgb.value):
            predictions = self.base_model.predict_proba(data)

        elif isinstance(self.base_model, BaseModelType.dummy.value):
            predictions = self.base_model.predict_proba(data)

        elif isinstance(self.base_model, BaseModelType.forest.value):
            predictions = self.base_model.predict_proba(data)

        elif isinstance(self.base_model, BaseModelType.nb.value):
            for i, col_name in enumerate(cat_columns):
                data = one_hot_encode(data, col_name, encoder=self.encoders[i])
            predictions = self.base_model.predict_proba(data)

        elif isinstance(self.base_model, BaseModelType.knn.value):
            for i, col_name in enumerate(cat_columns):
                data = one_hot_encode(data, col_name, encoder=self.encoders[i])
            predictions = self.base_model.predict_proba(data)

        elif isinstance(self.base_model, BaseModelType.regression.value):
            for i, col_name in enumerate(cat_columns):
                data = one_hot_encode(data, col_name, encoder=self.encoders[i])
            predictions = self.base_model.predict_proba(data)

        elif isinstance(self.base_model, BaseModelType.tree.value):
            predictions = self.base_model.predict_proba(data)

        elif isinstance(self.base_model, BaseModelType.formal_concept.value):
            data = data[self.base_model.features].copy()
            for i, col_name in enumerate(num_columns):
                if col_name in self.base_model.features:
                    data = binarize_column(
                        data=data, num_feature=col_name, thr=self.base_model.thr_dict[col_name], scaler=self.scalers[i]
                    )
            for i, col_name in enumerate(cat_columns):
                if col_name in self.base_model.features:
                    data = one_hot_encode(data, col_name, encoder=self.encoders[i], return_bool=True, drop_first=False)
            predictions = self.base_model.predict_proba(data)

        else:
            raise ValueError("Not valid model!")

        return predictions

    def fit(self, data: pd.DataFrame, target: pd.Series):
        cat_columns = find_cat_columns(data)
        num_columns = find_num_columns(data)

        for col_name in cat_columns:
            le = LabelEncoder()
            data = label_encode(data=data, cat_feature=col_name, encoder=le, is_inference=False)
            self.encoders.append(le)

        for col_name in num_columns:
            ss = StandardScaler()
            data = normalize_numeric_values(data=data, num_feature=col_name, scaler=ss, is_inference=False)
            self.scalers.append(ss)

        self.fit_model(data, np.ravel(target), cat_columns, num_columns)

    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        cat_columns = find_cat_columns(data)
        num_columns = find_num_columns(data)

        for i, col_name in enumerate(cat_columns):
            data = label_encode(data=data, cat_feature=col_name, encoder=self.encoders[i], is_inference=True)

        for i, col_name in enumerate(num_columns):
            data = normalize_numeric_values(data=data, num_feature=col_name, scaler=self.scalers[i], is_inference=True)

        predictions = self.predict_proba_model(data, cat_columns, num_columns)
        return predictions

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        probs = self.predict_proba(data)[:, 1]
        predictions = np.where(probs > 0.5, 1, 0)
        return predictions

    def save(self, path: pathlib.Path):
        with tempfile.TemporaryDirectory() as _output_path:
            output_path = pathlib.Path(_output_path)

            with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED) as archieve:
                par: Dict[str, Any] = {}

                pickle.dump(par, open(output_path / f"par.pkl", "wb"))
                pickle.dump(self.base_model, open(output_path / f"base_model.pkl", "wb"))
                pickle.dump(self.encoders, open(output_path / f"encoders.pkl", "wb"))
                pickle.dump(self.scalers, open(output_path / f"scalers.pkl", "wb"))

                archieve.write(output_path / f"par.pkl", pathlib.Path(f"par.pkl"))
                archieve.write(output_path / f"base_model.pkl", pathlib.Path(f"base_model.pkl"))
                archieve.write(output_path / f"encoders.pkl", pathlib.Path(f"encoders.pkl"))
                archieve.write(output_path / f"scalers.pkl", pathlib.Path(f"scalers.pkl"))

    @classmethod
    def load(cls, path: pathlib.Path) -> "Pipeline":
        with tempfile.TemporaryDirectory() as _output_path:
            output_path = pathlib.Path(_output_path)

            with zipfile.ZipFile(path, mode="r") as archieve:
                archieve.extractall(output_path)

            par = pickle.load(open(output_path / f"par.pkl", "rb"))
            base_model = pickle.load(open(output_path / f"base_model.pkl", "rb"))
            encoders = pickle.load(open(output_path / f"encoders.pkl", "rb"))
            scalers = pickle.load(open(output_path / f"scalers.pkl", "rb"))

        loaded_instance = cls(base_model=base_model, **par)
        loaded_instance.encoders = encoders
        loaded_instance.scalers = scalers
        return loaded_instance
