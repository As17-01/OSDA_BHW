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
        
        weather = weather.iloc[np.random.choice(len(weather), 10000, replace=False)]

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
        elif dataset == DatasetEnum.titanic:
            df = self._load_titanic()
        elif dataset == DatasetEnum.breast:
            df = self._load_breast()
        else:
            raise ValueError("Not valid dataset!")

        return df
