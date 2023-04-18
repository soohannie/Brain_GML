import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool

class GraphSAGE(nn.Module):
    def __init__(self, input_dim=200, num_class=2, hidden_channels=32, dropout=0.5):
        super(GraphSAGE, self).__init__()
        self.sc1 = SAGEConv(input_dim, hidden_channels)
        self.sc2 = SAGEConv(hidden_channels, hidden_channels)
        self.sc3 = SAGEConv(hidden_channels, hidden_channels)
        self.linear = Linear(hidden_channels, num_class)
        self.dropout = dropout

    def forward(self,  x, edge_index, edge_attr, batch):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.sc1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.sc2(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.sc3(x, edge_index)
        x = F.elu(x)
        x = global_mean_pool(x, batch)
        x = self.linear(x)
        return F.log_softmax(x, dim=1)