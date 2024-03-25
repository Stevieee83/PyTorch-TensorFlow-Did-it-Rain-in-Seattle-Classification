# # TensorFlow Feed-Forward Neural Network
# ## Dataset: Did it Rain in Seattle? (1948-2017)
# 
# Kaggle Dataset Web Link:
# 
# https://www.kaggle.com/datasets/rtatman/did-it-rain-in-seattle-19482017
# 

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn import metrics

import pandas as pd
import matplotlib.pyplot as plt
import os
import pathlib

# Hyperparameters
input_shape = (4,)       # Sets the number of input parameters in the input layer
input_units = 7          # Sets the number of output parameters in the input layer
hidden_units_1 = 7       # Sets the number of input parameters in hidden layer 1
hidden_units_2 = 7       # Sets the number of input parameters in hidden layer 2
output_units = 1         # Sets the number of parameters at the output layer
lr = 1e-3                # Sets the leanring rate
epochs = 60              # Sets the number of training epochs
batch_size = 32          # Sets the batch size
# -----------------

# Read in the CSV data to the script with the Pandas library
X_train_in = pd.read_csv('./dataset/X_train.csv')
y_train_in = pd.read_csv('./dataset/y_train.csv')
X_test = pd.read_csv('./dataset/X_test.csv')
y_test = pd.read_csv('./dataset/y_test.csv')

# Split the training and validation data with ScikitLearn
X_train, X_val, y_train, y_val = train_test_split(X_train_in, y_train_in, test_size=0.2, stratify=y_train_in, random_state=42)

# Build the Feed Forward Neural Network (FNN) model using the Keras sequential API
model = Sequential([
    
    # Input layer
    Dense(units=input_units, input_shape=input_shape, activation='relu'),
    
    # Hidden layer 1
    Dense(units=hidden_units_1, activation='relu'),
    
    # Hidden layer 2
    Dense(units=hidden_units_2, activation='relu'),
    
    # Output layer
    Dense(units=output_units, activation='sigmoid')
  ])

# Prints out the summary of the neural network architecture
print(model.summary())

# Compiles the TensorFlow Keras Neural Network classifier with the AdamW optimiser
model.compile(optimizer=AdamW(learning_rate=lr,
                                beta_1=0.9,
                                beta_2=0.999,
                                epsilon=1e-07),
                                loss ="binary_crossentropy", 
                                metrics=["accuracy"])

# Sets the random seed to repeat any randomness
tf.random.set_seed(1234)

# Initialise early stopping to the validation loss
callback = EarlyStopping(monitor='val_loss', patience=3)

# Fits the data to the TensorFlow Keras Neural Network classifier model
history = model.fit(X_train, y_train, 
                     epochs=epochs, 
                     batch_size=batch_size,
                     validation_data=(X_val, y_val),
                     callbacks=[callback]
)


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

# Creates a path for the plots to be output to
plot_dir = './TFplots/'

# Calls the makedir Python method
makedir(plot_dir)

# Stores the training and validation loss from training
loss = history.history["loss"]
val_loss = history.history["val_loss"]

# Conts the number of training epochs run during training
epochs = range(len(history.history["loss"]))

# Plots the training and validation loss
plt.plot(epochs, loss, label="Train Loss")
plt.plot(epochs, val_loss, label="Val Loss")
plt.title("Training and Valivdation Loss Plot")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig(plot_dir + 'loss.png')
plt.show()

# Stores the training and validation accuracy from training
accuracy = history.history["accuracy"]
val_accuracy = history.history["val_accuracy"]

# Plots the training and validation accuracy
plt.plot(epochs, accuracy, label="Train Accuracy")
plt.plot(epochs, val_accuracy, label="Val Accuracy")
plt.title("Training and Valivdation Accuracy Plot")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(plot_dir + 'accurccy.png')
plt.show()

# Evaluate the test dataset on the TensorFlow Keras Neural Network classifier
loss, accuracy = model.evaluate(X_test, y_test)

# Test Accuracy and loss
print("Accuracy", accuracy)
print("Loss", loss)

# Checkpoint file path directory
path = './tensorflow-model-python/'

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

# Saves the model weights as classifier.h5
model.save(path + 'classifier.h5')