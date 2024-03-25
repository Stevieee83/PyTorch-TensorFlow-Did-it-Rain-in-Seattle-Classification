import torch
import torch.nn as nn

# FNN PyTorch Neural Netwrk Model
class FNN(nn.Module):
    
    """Feed Forward Neural Network (FNN) object.
       Args:
       input_layer (tensor): Sets the number of neurons in the input layer.
       hiddel_layer_1 (tensor): Sets the number of neurons in the 1st hidden layer.
       hiddel_layer_2 (tensor): Sets the number of neurons in the 2nd hidden layer.
       output_layer (tensor): Sets the number of neurons in the output layer.
    """
    
    # FNN class constructor
    def __init__(self, input_layer, hiddel_layer_1, hiddel_layer_2, output_layer):
        super().__init__()
        
        # FNN class attributes
        self.fc1 = nn.Linear(input_layer, hiddel_layer_1)
        self.fc2 = nn.Linear(hiddel_layer_1, hiddel_layer_2)
        self.output = nn.Linear(hiddel_layer_2, output_layer)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    # Python forward method to control the forward pass of the FNN object
    def forward(self, x):
        
        # PyTorch FNN neural network layers and activation functions
        x = self.fc1(x)      # Input layer
        x = self.relu(x)
        x = self.fc2(x)      # Hidden Layer 1
        x = self.relu(x)
        x = self.fc2(x)      # Hidden Layer 2
        x = self.relu(x)
        x = self.output(x)   # Output layer
        x = self.sigmoid(x)
        
        # Returns the output results of the FNN object
        return x