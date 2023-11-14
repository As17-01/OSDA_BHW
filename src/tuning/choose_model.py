from typing import Any
from typing import Dict

import optuna
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def _choose_catboost(trial: optuna.Trial) -> Dict[str, Any]:
    model_config = {
        "silent": True,
    }

    model_config["iterations"] = trial.suggest_int("iterations", 50, 500, 25)
    model_config["depth"] = trial.suggest_int("depth", 1, 9)
    model_config["learning_rate"] = trial.suggest_float("learning_rate", 0.001, 1, log=True)
    model_config["l2_leaf_reg"] = trial.suggest_float("l2_leaf_reg", 1e-3, 20, log=True)
    model_config["colsample_bylevel"] = trial.suggest_float("colsample_bylevel", 0.01, 1)

    bootstrap_type = trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"])
    model_config["bootstrap_type"] = bootstrap_type
    if bootstrap_type == "Bayesian":
        model_config["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
    elif bootstrap_type == "Bernoulli":
        model_config["subsample"] = trial.suggest_float("subsample", 0.1, 1)

    return model_config


def _choose_xgboost(trial: optuna.Trial) -> Dict[str, Any]:
    model_config = {
        'max_depth': trial.suggest_int('max_depth', 1, 9),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 1.0, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 50, 500, 25),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.01, 1.0, log=True),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.01, 1.0, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
        "enable_categorical": True,
    }

    return model_config


def _choose_knn(trial: optuna.Trial) -> Dict[str, Any]:
    model_config = {
        "n_neighbors": trial.suggest_int("n_neighbors", 2, 10, log=True),
    }
    model_config["algorithm"] = trial.suggest_categorical('algorithm', ['auto','ball_tree','kd_tree','brute'])

    return model_config


def _choose_forest(trial: optuna.Trial) -> Dict[str, Any]:
    model_config = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
        'max_depth': trial.suggest_int('max_depth', 4, 50),
        'min_samples_split': trial.suggest_int('min_samples_split', 1, 150),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 60),
    }

    return model_config


def _choose_logreg(trial: optuna.Trial) -> Dict[str, Any]:
    model_config = {
        "max_iter": 10000,
        'tol' : trial.suggest_float('tol' , 1e-6 , 1e-3),
        'C' : trial.suggest_float("C", 1e-2, 1, log=True),
    }

    return model_config


def _choose_tree(trial: optuna.Trial) -> Dict[str, Any]:
    model_config = {
        'max_depth': trial.suggest_int('max_depth', 1, 9),
    }

    return model_config


def choose_model(trial: optuna.Trial, model_name: str) -> Any:
    if model_name == "CatBoost":
        model_config = _choose_catboost(trial)
        model = CatBoostClassifier(**model_config)
    elif model_name == "XGBoost":
        model_config = _choose_xgboost(trial)
        model = XGBClassifier(**model_config)
    elif model_name == "KNN":
        model_config = _choose_knn(trial)
        model = KNeighborsClassifier(**model_config)
    elif model_name == "NaiveBayes":
        model_config = _choose_bayes(trial)
        model = GaussianNB(**model_config)
    elif model_name == "RandomForest":
        model_config = _choose_forest(trial)
        model = RandomForestClassifier(**model_config)
    elif model_name == "LogRegression":
        model_config = _choose_logreg(trial)
        model = LogisticRegression(**model_config)
    elif model_name == "DecisionTree":
        model_config = _choose_tree(trial)
        model = DecisionTreeClassifier(**model_config)
    else:
        raise ValueError("Not valid model name!")
    return model
