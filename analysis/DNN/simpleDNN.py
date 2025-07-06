import torch.nn as nn
import torch

class SimpleDNN(nn.Module):
    def __init__(self, input_size, hidden_layers):
        super(SimpleDNN, self).__init__()
        # self.weights = torch.tensor(weights,dtype=torch.float32)
        layers = []
        in_features = input_size
        
        # Add hidden layers dynamically
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(0.15))
            in_features = hidden_size
        
        # Output layer
        layers.append(nn.Linear(in_features, 2))  
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        # x = x*self.weights # Apply the weights
        return self.network(x)





