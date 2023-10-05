import pathlib
from enum import Enum
from enum import auto
from typing import List

import numpy as np
import pandas as pd


class DatasetEnum(Enum):
    """Enum for datasets."""

    mushrooms = auto()
    water = auto()
    heart = auto()
    fruits = auto()
    weather = auto()

    @classmethod
    def _missing_(cls, value):
        raise NotImplementedError(
            f"{value} is not a valid {cls.__name__}. Only {', '.join([repr(m.value) for m in cls])} types are allowed"
        )


class DatasetLoader:
    def __init__(self, path: pathlib.Path):
        self.path = path

    @staticmethod
    def _convert_to_categorical(df: pd.DataFrame, exclude: List[str] = []) -> None:
        for col_name in df.columns:
            if df[col_name].dtypes == "object" and col_name not in exclude:
                df[col_name] = df[col_name].astype("category")

    def _load_mushrooms(self) -> pd.DataFrame:
        column_names = [
            "class",
            "cap-shape",
            "cap-surface",
            "cap-color",
            "bruises",
            "odor",
            "gill-attachment",
            "gill-spacing",
            "gill-size",
            "gill-color",
            "stalk-shape",
            "stalk-root",
            "stalk-surface-above-ring",
            "stalk-surface-below-ring",
            "stalk-color-above-ring",
            "stalk-color-below-ring",
            "veil-type",
            "veil-color",
            "ring-number",
            "ring-type",
            "spore-print-color",
            "population",
            "habitat",
        ]

        mushrooms = pd.read_csv(self.path / "mushroom-uci-dataset/agaricus-lepiota.data", names=column_names)

        mushrooms["class"] = mushrooms["class"].map({"p": 1, "e": 0})
        mushrooms.rename(columns={"class": "target"}, inplace=True)

        self._convert_to_categorical(mushrooms)
        return mushrooms

    def _load_water(self) -> pd.DataFrame:
        water = pd.read_csv(self.path / "water/waterQuality.csv")

        water.rename(columns={"is_safe": "target"}, inplace=True)
        water.replace("#NUM!", np.nan, regex=False, inplace=True)

        water = water[~water["target"].isna()]

        ammonia_mean = np.mean(water[~water["ammonia"].isna()].astype("float64").values)
        water["ammonia"] = water["ammonia"].fillna(ammonia_mean).astype("float64")
        water["target"] = water["target"].astype("int32")

        self._convert_to_categorical(water)
        return water

    def _load_heart(self) -> pd.DataFrame:
        heart = pd.read_csv(self.path / "heart-failure-dataset/heart.csv")
        heart.rename(columns={"HeartDisease": "target"}, inplace=True)

        self._convert_to_categorical(heart)
        return heart
    
    def _load_fruits(self) -> pd.DataFrame:
        fruits = pd.read_excel(self.path / "date_fruit_datasets/Date_Fruit_Datasets.xlsx")

        self._convert_to_categorical(fruits)
        return fruits
    
    def _load_weather(self) -> pd.DataFrame:
        weather = pd.read_csv(self.path / "weather/weatherAUS.csv")

        self._convert_to_categorical(weather)
        return weather

    def load_dataset(self, dataset: DatasetEnum) -> pd.DataFrame:
        if dataset == DatasetEnum.mushrooms:
            df = self._load_mushrooms()
        elif dataset == DatasetEnum.water:
            df = self._load_water()
        elif dataset == DatasetEnum.heart:
            df = self._load_heart()
        elif dataset == DatasetEnum.fruits:
            df = self._load_fruits()
        elif dataset == DatasetEnum.weather:
            df = self._load_weather()
        else:
            raise ValueError("Not valid dataset!")

        return df
