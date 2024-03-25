# TensorFlow Deep Neural Network

The TensorFlow deep neural network is trained and evaluated on the Did it Rain in Seattle tabular dataset.

Dataset link:

https://www.kaggle.com/datasets/rtatman/did-it-rain-in-seattle-19482017

Python files in the repository:

## Jupyter Notebooks:

Seattle_Weather_dataset_binary_classification.ipynb     Creates the Did it Rain in Seattle tabular dataset train and test splits (80% training, 20% testing)
FNN-tensorflow-seattle-weather-train.ipynb              TensorFlow Feed Forward Neural Network (FNN) training script.
tensorflow-test.ipynb                                   TensorFlow Feed Forward Neural Network (FNN) training script.

## Python (.py) Script Files:

csv_dataset.py                                          TensorFlow CSV dataset object.
fnn.py                                                  TensorFlow Feed Forward neural network (FNN) object.
FNN-tensorflow-seattle-weather-train.py                 TensorFlow Feed Forward Neural Network (FNN) training script.
tensorflow-test.ipynb                                   TensorFlow Feed Forward Neural Network (FNN) test script.

## Folders:

dataset                                                 Input Did it Rain in Seattle CSV data
tensorflow-model                                        FNN model weights file from the Jupyter Notebook - FNN-tensorflow-seattle-weather-train.ipynb
tensorflow-model-python                                 FNN model weights file from the Python script file - FNN-tensorflow-seattle-weather-train.py
TFplots                                                 Output plots from the Python script files - FNN-tensorflow-seattle-weather-train.py and tensorflow-test.py

For ease of use, the Jupyter Notebooks and the Python script files have been configured to be trained on a CPU device rather than a GPU.