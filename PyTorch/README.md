# PyTorch Deep Neural Network

The PyTorch deep neural network is trained and evaluated on the Did it Rain in Seattle tabular dataset.

Dataset link:

https://www.kaggle.com/datasets/rtatman/did-it-rain-in-seattle-19482017

Python files in the repository:

## Jupyter Notebooks:

Seattle_Weather_dataset_binary_classification.ipynb     Creates the Did it Rain in Seattle tabular dataset train and test splits (80% training, 20% testing)
pytorch-prepare-data.ipynb                              Aditional data pre-processing for tabular datasets in PyTorch. The additional steps allow the Did it Rain in Seattle CSV data to be converted to PyTorch tensors more easily.
FNN-pytorch-seattle-weather-train.ipynb                 PyTorch Feed Forward Neural Network (FNN) training script.
pytorch-test.ipynb                                      PyTorch Feed Forward Neural Network (FNN) training script.

## Python (.py) Script Files

csv_dataset.py                                          PyTorch CSV dataset object.
fnn.py                                                  PyTorch Feed Forward neural network (FNN) object.
FNN-pytorch-seattle-weather-train.py                    PyTorch Feed Forward Neural Network (FNN) training script.
pytorch-test.ipynb                                      PyTorch Feed Forward Neural Network (FNN) test script.

## Folders:

output_csv_data                                         Input Did it Rain in Seattle CSV data
pytorch-model                                           FNN model weights file from the Jupyter Notebook - FNN-pytorch-seattle-weather-train.ipynb
pytorch-model-python                                    FNN model weights file from the Python script file - FNN-pytorch-seattle-weather-train.py
PTplots                                                 Output plots from the Python script files - FNN-pytorch-seattle-weather-train.py and pytorch-test.py

For ease of use, the Jupyter Notebooks and the Python script files have been configured to be trained on a CPU device rather than a GPU.