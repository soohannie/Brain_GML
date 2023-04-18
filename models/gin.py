import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.models import GIN as gin_model

class GIN(nn.Module):
    def __init__(self, input_dim=200, num_class=2, hidden_channels=32, dropout=0.5):
        super(GIN, self).__init__()
        self.gin_layers = gin_model(in_channels=input_dim, hidden_channels=hidden_channels, num_layers=3, dropout=dropout)
        self.linear = Linear(hidden_channels, num_class)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.gin_layers(x=x, edge_index=edge_index, edge_attr=edge_attr)
        x = global_mean_pool(x, batch)  
        x = self.linear(x)
        return F.log_softmax(x, dim=1)