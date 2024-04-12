# # PyTorch Feed-Forward Neural Network
# 
# ## Dataset: Did it Rain in Seattle? (1948-2017)
# 
# Kaggle Dataset Web Link:
# 
# https://www.kaggle.com/datasets/rtatman/did-it-rain-in-seattle-19482017

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, default_collate
import torchmetrics
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

import pandas as pd
import os
import pathlib
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report
from sklearn import metrics

from fnn import FNN
from csv_dataset import CSVDataset

# Hyperparameters
input_layer = 4        # Sets the number of parameters in the input layer
hidden_layer_1 = 7     # Sets the number of parameters in hidden layer 1
hidden_layer_2 = 7     # Sets the number of parameters in hidden layer 2
hidden_layer_3 = 7     # Sets the number of parameters in hidden layer 3
hidden_layer_4 = 7     # Sets the number of parameters in hidden layer 4
hidden_layer_5 = 7     # Sets the number of parameters in hidden layer 5
hidden_layer_6 = 7     # Sets the number of parameters in hidden layer 6
output_layer = 1       # Sets the number of parameters in the output layer
lr = 1e-3              # Sets the leanring rate
epochs = 50            # Sets the number of training epochs
batch_size = 32        # Sets the batch size
# -----------------

# Sets the random seed to repeat any randomness
torch.random.manual_seed(1234)

# Sets the device to CPU
device = 'cpu'

# Prints out the available processing device to the screen
print("Device: ", device)

# Read in the data as Pandas DataFrames and convert to NumPy arrays
X = pd.read_csv("./output_csv_data/X_train", header=None).values
y = pd.read_csv("./output_csv_data/y_train", header=None).values

# Reshapes the target classes y
y.reshape(-1, 1)

# Stores the CSVDataset object in the dataset Python variable
dataset = CSVDataset(X, y)

# Randomly slits the training and validation dataset
trainset, valset = random_split(dataset, [0.8, 0.2])

# Creates the training and validation PyTorch DataLoaders
train_loader = DataLoader(trainset, shuffle=True, batch_size=batch_size)
val_loader = DataLoader(valset, shuffle=False, batch_size=batch_size)

# Defines the FNN PyTorch model
model = FNN(input_layer=input_layer,
            hidden_layer_1=hidden_layer_1,
            hidden_layer_2=hidden_layer_2,
            hidden_layer_3=hidden_layer_3,
            hidden_layer_4=hidden_layer_4,
            hidden_layer_5=hidden_layer_5,
            hidden_layer_6=hidden_layer_6,
            output_layer=output_layer).to(device)

# Pritns out the FNN model architecture to the screen
print(model)

# Defines the Binary Cross Entropy loss functions and AdamW optimisation algorithm
loss_fn = nn.BCELoss()
optimizer = optim.AdamW(model.parameters(), lr=lr)

# Python method to check for and create a model file path directory
def makedir(path):
    
    """Checks for a file path directory and makes one if required."""

    # File path directory to check for
    file = pathlib.Path(path)

    # Conditional if statement to check for the model checkpoint filep ath directory
    if file.exists():
        # Prints out the file path directory exists if it does and passes the conditional
        print('File path exists: ', path)
        pass
    else:
        # Outputs the folder contents to checkpoint saved models to
        os.makedirs(path)

# Checkpoint file path directory
path = './pytorch-model-python/'

# Calls the makedir Python method
makedir(path)

# Python method to calculate trianing and validation accuracy
def accuracy(y_pred, y_batch):
    
    """Calcuates the trianing and validaiton accuracy."""
    
    # Retuns the calculated accuracy value
    return (y_pred.round() == y_batch).float().mean()

# Sets the FNN model to training mode and to activate backpropagation
model.train()

# Python lists to store the training and validation losses and accuracies for plotting
train_losses = []
train_accuracy = []
val_losses = []
val_accuracy = []

# Variables for model checkpointing
low_loss = 1000
file_name = None

# Training loop to control the number of training epochs
for epoch in range(epochs):
    
    # Prints out the epoch number to the screen during training
    print(f"Epoch {epoch+1}")
    print("----------------")
    
    # Stores the accumulative training loss and accuracy
    train_loss, train_acc = 0, 0
    
    # Loads the training data batches to the model
    for i, train_data in enumerate(train_loader):
        
        # Unpack the training data features and target classes
        X_batch, y_batch = train_data
        
        # Sends the training data features and target classes to the device
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        # Stores the predictions from the FNN model
        y_pred = model(X_batch)
        
        # Calculates the training loss for the batch during training
        loss = loss_fn(y_pred, y_batch)
        
        # Activates backpropagation after calculating the training loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Calcualtes the training accuracy for the batch during training
        acc = accuracy(y_pred, y_batch)
        
        # Accumulates the training loss and accuracy
        train_loss += loss
        train_acc += acc
        
    
    # Calculates the training loss and accuracy for the epoch
    train_loss /= len(train_loader)
    train_acc /= len(train_loader)
    
    # Appends the training loss and accuracy to their respective list for plotting
    train_losses.append(train_loss.item())
    train_accuracy.append(train_acc.item())
    
    # Prints out the training loss and accuracy to the screen
    print(f"Taining Loss {train_loss}% Training Accuracy {train_acc*100:.3f}%")
    
    # Freezes the model gradients for validation
    model.eval()
    
    # Stores the validation loss and accuracy
    val_loss, val_acc = 0, 0
    
    # Loads the validation batches to the model
    for j, val_data in enumerate(val_loader):
        
        # Unpacks the validation features and target classes
        X_batch, y_batch = val_data
        
        # Sends the validation features and target classes to the device
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        # Stores the validation predictions from the model
        y_pred = model(X_batch)
        
        # Calculates the validation loss for the batch
        loss = loss_fn(y_pred, y_batch)
        
        # Calculates the validation accuracy for the batch
        acc = accuracy(y_pred, y_batch)
        
        # Accumulates the validation loss and accuracy
        val_loss += loss
        val_acc += acc
        
    # Accumulates the validation loss and accuracy
    val_loss /= len(val_loader)
    val_acc /= len(val_loader)
    
    # Appends the validation loss and accuracy to their respective list for plotting
    val_losses.append(val_loss.item())
    val_accuracy.append(val_acc.item())
    
    # Prints out the validation loss and accuracy to the screen
    print(f"Validation Loss {val_loss}% Validation Accuracy {val_acc*100:.3f}%")
    print("\n")
    
    # Save the FNN model with the lowest validation loss
    if val_loss < low_loss:
        
        # Update the low_loss variable with the lowest validation loss
        low_loss = val_loss
        
        # Conditional if statement to and remove a model with a higher loss if required
        if file_name == None:
            # Passes if the file name is equal to none
            pass
        else:
            # Removes the file name of the model with the higher validation loss
            os.remove(file_name)
            
        # Model checkpoint file path directory
        model_path = path + f'epoch-{epoch+1}.pth'
        
        # Saves the model with PyTorch
        torch.save(model, model_path)
        
        # Sets the file_name variable to the model path
        file_name = model_path

# Creates a path for the plots to be output to
plot_dir = './PTplots/'

# Calls the makedir Python method
makedir(plot_dir)

# Plots the training and validation loss curves
plt.title('Training and Valivdation Loss Plot')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.savefig(plot_dir + 'loss.png')
plt.show()

# Plots the training and validation accuracy curves
plt.title('Training and Valivdation Accuracy Plot')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.plot(train_accuracy, label='Train Accuracy')
plt.plot(val_accuracy, label='Val Accuracy')
plt.savefig(plot_dir + 'accurccy.png')
plt.show()