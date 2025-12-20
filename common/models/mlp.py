import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiLayerPerceptronSkips(nn.Module):
    def __init__(self, dim_in, dim_hidden, n_layers):
        super().__init__()
        
        layers = []
        layers.append(nn.Linear(dim_in, dim_hidden))
        layers.append(nn.ReLU())
        
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(dim_hidden, dim_hidden))
            layers.append(nn.ReLU())
            
        layers.append(nn.Linear(dim_hidden, 1))
        
        self.net = nn.Sequential(*layers)
        
        # metadata (保持和你原来一致，方便实验管理)
        self.id = "MLP"
        self.meta = [dim_in, dim_hidden, n_layers]
        
    def forward(self, x):
        return self.net(x)
    
    
def MultiLayerPerceptron(dim_in, dim_hidden, n_layers):
    return DenseMLP(dim_in, dim_hidden, n_layers)
