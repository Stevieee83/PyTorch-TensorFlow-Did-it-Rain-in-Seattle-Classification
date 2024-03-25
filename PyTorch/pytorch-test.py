# # PyTorch Feed-Forward Neural Network Tests
# ## Dataset: Did it Rain in Seattle? (1948-2017)
# 
# Kaggle Dataset Web Link:
# 
# https://www.kaggle.com/datasets/rtatman/did-it-rain-in-seattle-19482017
# 
# 
# A separate test Jupyter Notebook displays how a deep aural network model is loaded to a runtime, and predictions are made from the model on test data. The Jupyter Notebook simulates how a deep neural network operates when deployed to a web application online without the deployment steps.

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchmetrics
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

import pandas as pd
import os
import pathlib
import matplotlib.pyplot as plt

from fnn import FNN
from csv_dataset import CSVDataset

# Model paramters from the training Jupyter Notebook
input_layer = 4        # Sets the number of parameters in the input layer
hiddel_layer_1 = 7     # Sets the number of parameters in hidden layer 1
hiddel_layer_2 = 7     # Sets the number of parameters in hidden layer 2
output_layer = 1       # Sets the number of parameters in the output layer
batch_size = 32        # Sets the batch size
# -----------------

# Sets the device to the CPU
device = 'cpu'

# Prints out the available processing device to the screen
print("Device: ", device)

# Defines the FNN PyTorch model
model = FNN(input_layer=input_layer,
            hiddel_layer_1=hiddel_layer_1,
            hiddel_layer_2=hiddel_layer_2,
            output_layer=output_layer)

# Pritns out the FNN model architecture to the screen
print("Model: ", model)

# Loads the test data to the Jupyter Notebook
X_test = pd.read_csv("./output_csv_data/X_test", header=None).values
y_test = pd.read_csv("./output_csv_data/y_test", header=None).values

# Reshapes the target classes y
y_test.reshape(-1, 1)


# Stores the test CSVDataset in the dataset variable
testset = CSVDataset(X_test, y_test)

# Creates the test PyTorch DataLoader
test_loader = DataLoader(testset, shuffle=False, batch_size=batch_size)

# Checkpoint file path directory
path = './pytorch-model-python/'

# Gets the name of the saved model weight file State Dictionary
model_file = os.listdir(path)

# load the daves model to the Jupyter Notebook
model_load = torch.load(path + model_file[0])

# Pritns out the saved model weights file State Dictionary to the screen
print("Model Loaded: ", model_load)

# Defines the torchmetrics evaluation test metrics for the FNN model
accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=2)
recall = torchmetrics.Recall(task="multiclass", average="macro", num_classes=2)
precision = torchmetrics.Precision(task="multiclass", average="macro", num_classes=2)
f1_score = torchmetrics.F1Score(task="multiclass", num_classes=2)

# Stores the test accuracies in a Python list
test_accuracies = []

# Stores all the predictions and labels from the test dataset
y_preds = []
labels = []

# Freezes the model gradients for validation
with torch.no_grad():
    
    # Stores the test accuracy
    test_acc, test_rec, test_pre, test_f1 = 0, 0, 0, 0
    
    # Loads the test dataset batches to the model
    for k, data in enumerate(test_loader):
        
        # Unpacks the features and target classes from the data variable
        X_batch, y_batch = data
        
        # Sends the features and the target classes to the device
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        # Stores the test predictions from the model
        y_pred = model_load(X_batch)
        
        # Calculates the test accuracy, recall, precision and F1 score for the batch
        acc = accuracy(y_pred.round(), y_batch)
        rec = recall(y_pred.round(), y_batch)
        pre = precision(y_pred.round(), y_batch)
        f1 = f1_score(y_pred.round(), y_batch)
        
        # Accumulates the test accuracy, recall, precision and F1 score
        test_acc += acc
        test_rec += rec
        test_pre += pre
        test_f1 += f1
        
        # Appends the test accuracies to the test_accuracies list
        test_accuracies.append(acc)
        y_preds.append(y_pred.round())
        labels.append(y_batch)
        
    # Calculates the overall test accuracy, recall, precision and F1 Score
    test_acc /= len(test_loader)
    test_rec /= len(test_loader)
    test_pre /= len(test_loader)
    test_f1 /= len(test_loader)
    
    # Creates a PyTorch tensor for the labels for the test dataset
    y_pred_tensor = torch.cat(y_preds)
    labels=torch.cat(labels).int()
    
    # Prints out the overall test accuracy and the number of test batches to the screen
    print("Number of Test Batches:", len(test_accuracies))


# Prints out the overall test accuracy and the number of test batches to the screen
print(f"Test Accuracy {test_acc*100:.2f}%")

# Prints out the overall test Recall and the number of test batches to the screen
print(f"Test Recall {test_rec*100:.2f}%")

# Prints out the overall test precision and the number of test batches to the screen
print(f"Test Precision {test_pre*100:.2f}%")

# Prints out the overall test F1 score and the number of test batches to the screen
print(f"Test F1 Score {test_f1*100:.2f}%")

# Class names from the Seattle Weather Dataset
class_names = ['Did not Rain', 'Raied']

# Setup confusion matrix instance and compare predictions to targets
confmat = ConfusionMatrix(num_classes=len(class_names), task='multiclass')
confmat_tensor = confmat(preds=y_pred_tensor,
                         target=labels)

# Plot the confusion matrix
fig, ax = plot_confusion_matrix(
    conf_mat=confmat_tensor.numpy(), # matplotlib likes working with NumPy 
    class_names=class_names, # turn the row and column labels into class names
    figsize=(10, 7),
    show_normed=True,
    colorbar=True
);

# Adds a title to the Confusion Matrix plot
plt.title('Seattle Weather Confusion Matrix')

# Saves the plot to the plots folder
plt.savefig('./PTplots/confusion.png')

# Displays the plot to the screen
plt.show()