import pathlib
from enum import Enum
from enum import auto
from typing import List

import numpy as np
import pandas as pd


class Dataset(Enum):
    """Enum for datasets."""

    mushrooms = auto()
    water = auto()
    heart = auto()
    fruits = auto()
    weather = auto()
    titanic = auto()
    breast = auto()

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

        weather["RainTomorrow"] = weather["RainTomorrow"].map({"Yes": 1, "No": 0})
        weather.rename(columns={"RainTomorrow": "target"}, inplace=True)

        weather = weather[~weather["target"].isna()]
        weather["target"] = weather["target"].astype("int32")

        exclude = ["Date"]
        self._convert_to_categorical(weather, exclude=exclude)

        np.random.seed(100)

        weather = weather.iloc[np.random.choice(len(weather), 2000, replace=False)]

        weather.drop(columns=["Date", "Location", "RainToday"], inplace=True)

        for col_name in weather.drop(columns="target").columns:
            if weather[col_name].dtypes == "float" and np.all(
                weather[~weather[col_name].isna()][col_name] % 1.0 == 0.0
            ):
                weather[col_name] = weather[col_name].fillna(np.round(np.mean(weather[col_name]))).astype("int32")
            elif weather[col_name].dtypes == "float":
                weather[col_name] = weather[col_name].fillna(np.mean(weather[col_name])).astype("float64")
            elif weather[col_name].dtypes == "category":
                weather[col_name] = weather[col_name].astype("str").fillna("unk").astype("category")
        
        weather = weather[
            [
                "Evaporation",
                "Sunshine",
                "Rainfall",
                "WindSpeed9am",
                "WindSpeed3pm",
                "Pressure9am",
                "Pressure3pm",
                "target",
            ]
        ]
        return weather

    def _load_titanic(self) -> pd.DataFrame:
        titanic = pd.read_csv(self.path / "titanic/train.csv")

        titanic.rename(columns={"Survived": "target"}, inplace=True)

        exclude = ["Name", "Ticket", "Cabin"]
        self._convert_to_categorical(titanic, exclude=exclude)

        return titanic

    def _load_breast(self) -> pd.DataFrame:
        breast = pd.read_csv(self.path / "breast/breast-cancer.csv")

        breast["diagnosis"] = breast["diagnosis"].map({"M": 1, "B": 0})
        breast.rename(columns={"diagnosis": "target"}, inplace=True)

        return breast

    def load_dataset(self, dataset: Dataset) -> pd.DataFrame:
        if dataset == Dataset.mushrooms:
            df = self._load_mushrooms()
        elif dataset == Dataset.water:
            df = self._load_water()
        elif dataset == Dataset.heart:
            df = self._load_heart()
        elif dataset == Dataset.fruits:
            df = self._load_fruits()
        elif dataset == Dataset.weather:
            df = self._load_weather()
        elif dataset == Dataset.titanic:
            df = self._load_titanic()
        elif dataset == Dataset.breast:
            df = self._load_breast()
        else:
            raise ValueError("Not valid dataset!")

        return df
