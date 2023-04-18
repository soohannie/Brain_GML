import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool

class GATV2(nn.Module):
    def __init__(self, input_dim=200, num_class=2, hidden_channels=32, n_heads=5, dropout=0.5):
        super(GATV2, self).__init__()
        self.gc1 = GATv2Conv(input_dim, hidden_channels,
                           heads=n_heads, dropout=dropout)
        self.gc2 = GATv2Conv(hidden_channels*n_heads, hidden_channels,
                           heads=n_heads, dropout=dropout)
        self.gc3 = GATv2Conv(hidden_channels*n_heads, hidden_channels,
                           heads=n_heads, dropout=dropout)
        self.linear = Linear(hidden_channels*n_heads, num_class)
        self.dropout = dropout

    def reset_parameters(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()

    def forward(self,  x, edge_index, edge_attr, batch):

        # No ReLU / dropout layers as these functionn are already included in torch_geometric.nn.GATv2Conv
        x = self.gc1(x, edge_index)
        x = self.gc2(x, edge_index)
        x = self.gc3(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.linear(x)
        return F.log_softmax(x, dim=1)