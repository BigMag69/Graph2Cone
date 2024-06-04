import torch
from torch import nn
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.data import Data, DataLoader, Batch

class GINSubgraphModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GINSubgraphModel, self).__init__()
        # Define the GIN layers
        self.gin1 = GINConv(nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))
        self.gin2 = GINConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))
        # Readout layer
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)  # Multiply by 2 for concatenated features of two graphs

    def forward(self, data1, data2):
        # Embed both graphs
        x1, edge_index1, batch1 = data1.x, data1.edge_index, data1.batch
        x2, edge_index2, batch2 = data2.x, data2.edge_index, data2.batch

        # Pass through GIN layers
        x1 = self.gin1(x1, edge_index1)
        x1 = self.gin2(x1, edge_index1)
        x2 = self.gin1(x2, edge_index2)
        x2 = self.gin2(x2, edge_index2)

        # Global pooling (summing node embeddings)
        x1 = global_add_pool(x1, batch1)
        x2 = global_add_pool(x2, batch2)

        # Concatenate and classify
        x = torch.cat([x1, x2], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))

        return x
