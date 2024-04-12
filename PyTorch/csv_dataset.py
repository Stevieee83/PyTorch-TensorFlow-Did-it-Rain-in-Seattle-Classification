import torch
from torch.utils.data import Dataset
import numpy as np

# CSVDataset Python class object
class CSVDataset(Dataset):
    
    """CSVDataset PyTorch dataset.
       Args:
       X (NumPy Array): Features of the CSV dataset
       y (NumPy Array): Targets of the CSV dataset
    """
    
    # CSVDataset class constructor
    def __init__(self, X, y):
        # Converts the features and targets into PyTorch tensors
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    # Python method to measure the lenght of the dataset
    def __len__(self):
        return len(self.X)

    # Python method to get an item from the dataset
    def __getitem__(self, idx):
        
        # Gets one sample from the dataset
        features = self.X[idx]
        target = self.y[idx]
        
        # Returns the features and the targets of the CSV dataset
        return features, target