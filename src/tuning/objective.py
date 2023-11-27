import optuna
import pandas as pd
from loguru import logger
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from src.pipeline import Pipeline
from src.tuning import choose_model


def objective(
    trial: optuna.Trial,
    data: pd.DataFrame,
    target: pd.Series,
    model_name: str,
) -> float:
    """Optuna objective."""
    # logger.info("Forming config for cross-validation")
    model = choose_model(trial, model_name, data)
    pipeline = Pipeline(base_model=model)
    logger.info(f"Selected config: {trial.params}")

    n_splits = 3
    fold_generator = KFold(n_splits=n_splits, shuffle=True, random_state=101)

    folds = fold_generator.split(data, y=target)
    avg_metric_value = 0
    for i, (train_fold_idx, test_fold_idx) in enumerate(folds):
        train_data, train_target = data.iloc[train_fold_idx], target.iloc[train_fold_idx]
        test_data, test_target = data.iloc[test_fold_idx], target.iloc[test_fold_idx]

        pipeline.fit(train_data, train_target)
        predictions = pipeline.predict(test_data)

        logger.info(f"Fold {i}: {accuracy_score(y_true=test_target, y_pred=predictions)}")
        avg_metric_value += accuracy_score(y_true=test_target, y_pred=predictions) / n_splits

    return avg_metric_value
