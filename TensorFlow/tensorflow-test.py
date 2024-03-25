# # TensorFlow Feed-Forward Neural Network
# ## Dataset: Did it Rain in Seattle? (1948-2017)
# 
# Kaggle Dataset Web Link:
# 
# https://www.kaggle.com/datasets/rtatman/did-it-rain-in-seattle-19482017
# 
# A separate test Jupyter Notebook displays how a deep aural network model is loaded to a runtime, and predictions are made from the model on test data. The Jupyter Notebook simulates how a deep neural network operates when deployed to a web application online without the deployment steps.

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense

from sklearn import metrics

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import classification_report

# Model paramters from the training Jupyter Notebook
input_shape = (4,)       # Sets the number of input parameters in the input layer
input_units = 7          # Sets the number of output parameters in the input layer
hidden_units_1 = 7       # Sets the number of input parameters in hidden layer 1
hidden_units_2 = 7       # Sets the number of input parameters in hidden layer 2
output_units = 1         # Sets the number of parameters at the output layer
batch_size = 32          # Sets the batch size
# -----------------

# Read in the CSV data to the script with the Pandas library
X_test = pd.read_csv('./dataset/X_test.csv')
y_test = pd.read_csv('./dataset/y_test.csv')

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

# Checkpoint file path directory
path = './tensorflow-model-python/'

# Reloads the classifier.h5 model weights
reloaded_model = tf.keras.models.load_model(path + 'classifier.h5')

# Makes the predictions from the model
predictions = reloaded_model.predict(X_test)

# Prints out a prediction to the screen from the test dataset
print("As evaluated by our model, there was a %.1f probability it rained at "
      "Seattle Airport at the time the input data was recorded." % (100 * predictions[0][0],)
)

# Displays the classification report from the deep neural network to the screen
ac=accuracy_score(y_test, predictions.round())
print('Accuracy of the model: ',ac)

# Displays the recall score from the deep neural network to the screen
rs=recall_score(y_test, predictions.round())
print('Recall score for the model: ',rs)

# Displays the recall score from the deep neural network to the screen
ps=precision_score(y_test, predictions.round())
print('Precision score for the model: ',ps)

# Displays the F1 score from the deep neural network to the screen
f1 = f1_score(y_test, predictions.round())
print('F1 score for the model: ',f1)

# Class names from the Seattle Weather Dataset
class_names = ['Did not Rain', 'Raied']

# Setup confusion matrix instance and compare predictions to targets
cm = confusion_matrix(y_test, predictions.round())

# Plot the confusion matrix
fig, ax = plot_confusion_matrix(
    conf_mat=cm,
    class_names=class_names,
    figsize=(10, 7),
    show_normed=True,
    colorbar=True
);

# Adds a title to the Confusion Matrix plot
plt.title('Seattle Weather Confusion Matrix')

# Saves the plot to the plots folder
plt.savefig('./TFplots/confusion.png')

# Displays the plot to the screen
plt.show()

# Displays the classification report for the deep neural network to the screen
cr = classification_report(y_test, predictions.round(), target_names=['Heart Disease (1)', 'Disease Free (0)', ])
print(cr)

# Stores the false positives, true positives and the thresholds for AUROC plotting
fpr, tpr, thresholds = metrics.roc_curve(y_test, predictions.round())

# Stores the AUROC curve plot values in the roc_auc variable
roc_auc = metrics.auc(fpr, tpr)

# Plots the Receiver Operating Characteristic AUROC curve
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig('./TFplots/auroc.png')
plt.show()