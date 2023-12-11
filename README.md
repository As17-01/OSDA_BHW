# OSDA Big Homework

## Installation 

Install the project using poetry:

`poetry install`

Some packages to run Neural FCA need to be installed separately because they cannot be downloaded via poetry. See them in `./neural_fca_example/NeuralFCA_Big_hw.ipynb`. Also I could not install them inside a virtual environment. The workaround was to preinstall them outside venv, then create one, and then install all the remaining dependencies.

Alternatively, you can create a VM, which should solve the problem.

## Structure

* `checkpoints` - various checkpoints, such as best validation parameters and etc.
* `datasets` - a small collection of datasets to try the algorithms.
* `images` - folder with neural network pictures.
* `metrics` - metric results.
* `neural_fca_example` - see `https://github.com/MariiaZueva/neuralFCA/blob/main/NeuralFCA_Big_hw.ipynb`.
* `src` - source code.

Check the following notebooks for the results:
* `experiments_heart.ipynb` - model performance on a dataset with heart diseases.
* `experiments_water.ipynb` - model performance on a dataset with safe water detection.
* `experiments_weather.ipynb` - model performance on a dataset with weather predictions.

## Decisions

To deal with numeric values I used binarization of features. 

# Results

Accuracy metric values are available either inside `*.ipynb` files, or as `*.csv` files in `/metrics/`.

## Conclusions

Neural FCA does not show good results on the following datasets.
