import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, global_mean_pool
from torch_geometric.nn.models import GCN as gcn_model

# class GCN(nn.Module):
#     def __init__(self, input_dim=200, num_class=2, hidden_channels=32, dropout=0.5):
#         super(GCN, self).__init__()
#         self.gcn_layers = gcn_model(in_channels=input_dim, hidden_channels=hidden_channels, num_layers=3, out_channels=None)
#         self.linear = Linear(hidden_channels, num_class)
#         self.dropout = dropout

#     def forward(self, x, edge_index, edge_attr, batch):
#         x = self.gcn_layers(x=x, edge_index=edge_index) #, edge_attr=data.edge_attr)
#         x = global_mean_pool(x, batch)  
#         x = F.dropout(x, p=self.dropout, training=self.training)
#         x = self.linear(x)
#         return F.log_softmax(x, dim=1)

class GCN(nn.Module):
    def __init__(self, input_dim=200, num_class=2, hidden_channels=32, dropout=0.5):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(input_dim, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.linear = Linear(hidden_channels, num_class)
        self.dropout = dropout
    
    def forward(self,  x, edge_index, edge_attr, batch):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index)
        x = F.elu(x)
        x = global_mean_pool(x, batch)  
        x = self.linear(x)
        return F.log_softmax(x, dim=1)