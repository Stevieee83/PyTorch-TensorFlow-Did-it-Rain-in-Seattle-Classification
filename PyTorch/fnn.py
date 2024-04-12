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
    def __init__(self, input_layer,
                 hidden_layer_1,
                 hidden_layer_2,
                 hidden_layer_3,
                 hidden_layer_4,
                 hidden_layer_5,
                 hidden_layer_6,
                 output_layer):
        super().__init__()

        # FNN class attributes
        self.fnn = nn.Sequential(nn.Linear(input_layer, hidden_layer_1),  # Input layer
                                 nn.ReLU(),
                                 nn.Linear(hidden_layer_2, hidden_layer_3),  # Hidden Layer 1
                                 nn.ReLU(),
                                 nn.Linear(hidden_layer_4, hidden_layer_5),  # Hidden Layer 3
                                 nn.ReLU(),
                                 nn.Linear(hidden_layer_6, output_layer),  # Output layer
                                 nn.Sigmoid())

    # Python forward method to control the forward pass of the FNN object
    def forward(self, x):
        # Returns the output results of the FNN object
        return self.fnn(x)