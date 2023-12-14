# Final Report

## Datasets

I used the following datasets in the project:

### Heart failure prediction

[Source](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)

This dataset contains 918 samples of medical conditions, which were used to predict heart failure. This dataset had a good combination of both categorical and numerical features, an appropriate number of samples.

Accuracy score:
![Alt text](heart_metrics.jpg?raw=true "Heart metrics")

### Water quality prediction

[Source](https://www.kaggle.com/datasets/mssmartypants/water-quality)

This dataset contains water compound elements. Theoretically, some combinations of elements in water could lead to the water being safe or not to drink by a human. The whole dataset contains only numerical features.

Accuracy score:
![Alt text](water_metrics.jpg?raw=true "Water metrics")

### Weather prediction

[Source](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package)

The dataset I used is a small part of a big dataset, which could be found at the link. It contains different weather conditions, which could be used to predict the weather. 

Accuracy score:
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

## NFCA parameters optimization

I optimized the following parameters of NFCA:

* Binarization of numerical features - I used only binary splitting of features because there `32` numerical features across the `3` datasets, and I have been choosing the thresholds for the binarizations using optuna on cross-validation.
* Used different features to use as attributes - because the algorithm runs in a reasonable time on approximately 15-16 features, I had to choose features from the original dataset to run NFCA. I also tried to select the features on a cross-validation.
* Tuned the number of epochs - I tried selecting different number of epochs for the NFCA.
* Tuned the number of best concepts - I tried adjusting the number of best concepts also on cross-validation.

## Results

NFCA does not show good results on all `3` datasets I used. It is much slower, comparing to the other algorithms. It requires a lot of manual work on features, which could be quite complicated, while other algorithms could also benefit from this, if not more, than NFCA. This makes an application of this algorithm quite complicated.

In the meantime, configuration: binarization, the number of best concepts and etc do not hold up, if the model gets trained on a different train dataset.  