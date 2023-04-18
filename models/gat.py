import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class GAT(nn.Module):
    def __init__(self, input_dim=200, num_class=2, hidden_channels=32, n_heads=5, dropout=0.5):
        super(GAT, self).__init__()
        self.gc1 = GATConv(input_dim, hidden_channels,
                           heads=n_heads, dropout=dropout)
        self.gc2 = GATConv(hidden_channels*n_heads, hidden_channels,
                           heads=n_heads, dropout=dropout)
        self.gc3 = GATConv(hidden_channels*n_heads, hidden_channels,
                           heads=n_heads, dropout=dropout)
        self.linear = Linear(hidden_channels*n_heads, num_class)
        self.dropout = dropout

    def reset_parameters(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()

    def forward(self,  x, edge_index, edge_attr, batch):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gc1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gc2(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gc3(x, edge_index)
        x = F.elu(x)
        x = global_mean_pool(x, batch)
        x = self.linear(x)
        return F.log_softmax(x, dim=1)