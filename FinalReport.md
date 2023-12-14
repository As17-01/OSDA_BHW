# Final Report

## Datasets

I used the following datasets in the project:

### Heart failure prediction

[Source](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)

This dataset contains 918 samples of medical conditions, which were used to predict heart failure. This dataset had a good combination of both categorical and numerical features, an appropriate number of samples.

Metrics:
![Alt text](heart_metrics.jpg?raw=true "Heart metrics")

### Water quality prediction

[Source](https://www.kaggle.com/datasets/mssmartypants/water-quality)

This dataset contains water compound elements. Theoretically, some combinations of elements in water could lead to the water being safe or not to drink by a human. The whole dataset contains only numerical features.

Metrics:
![Alt text](water_metrics.jpg?raw=true "Water metrics")

### Weather prediction

[Source](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package)

The dataset I used is a small part of a big dataset, which could be found at the link. It contains different weather conditions, which could be used to predict the weather. 

Metrics:
![Alt text](weather_metrics.jpg?raw=true "Weather metrics")

## Preprocessing

The preprocessing was pretty minor:
* I fixed datatypes in the datasets: formatted numerical to `int` and `float` datatypes; converted categorical features to `categorical` datatype.
* Fixed obvious errors in data, like using different typing of the same value of a categorical feature.

## Special processing for specific models:

Some of the models require special processing of features:

* `Sklearn` models require categorical features to be encoded. I used one-hot encoding for the purpose.
* `Logistic regression`, `KNN` require normalized data. I used `Standard scaler` to achieve that.
* `NFCA` requires logical type values. I separately used binarization of numerical features for the model and used a modified one-hot encoder, which does not drop linear dependent category. Also, this model is extremely slow, and I needed to cut the number of features in datasets, otherwise it was running for eternity.

## Experiments design

For cross-validation I used `K-Fold` validator with `k=5` and data shuffling for testing and `k=3` for fine-tuning.

For hyperparameters optimization I used `optuna` with different number of trials for each dataset. The best parameters are saved in `checkpoints` directory. I optimized every major parameter of each model.
