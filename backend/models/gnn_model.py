
import torch
import torch.nn as nn
import torch.nn.functional as F
# !!! ВИПРАВЛЕНИЙ ІМПОРТ !!!
from torch_geometric.nn import GATv2Conv, LayerNorm

class GNNModel(nn.Module):
    def __init__(self, hidden_channels=128, num_layers=8, dropout=0.1):
        super(GNNModel, self).__init__()
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.embed = nn.Embedding(10, hidden_channels)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            self.convs.append(GATv2Conv(hidden_channels, hidden_channels // 4, heads=4, concat=True))
            self.norms.append(LayerNorm(hidden_channels))

        self.classifier = nn.Linear(hidden_channels, 9)
        self.dropout = dropout
        self.register_buffer('single_edge_index', self._create_sudoku_edges())

    def _create_sudoku_edges(self):
        edges = []
        def get_idx(r, c): return r * 9 + c
        for r in range(9):
            for c in range(9):
                src = get_idx(r, c)
                neighbors = set()
                for k in range(9):
                    if k != c: neighbors.add(get_idx(r, k))
                    if k != r: neighbors.add(get_idx(k, c))
                br, bc = r // 3, c // 3
                for i in range(br*3, (br+1)*3):
                    for j in range(bc*3, (bc+1)*3):
                        if i != r or j != c: neighbors.add(get_idx(i, j))
                for dst in neighbors:
                    edges.append([src, dst])
        return torch.tensor(edges, dtype=torch.long).t().contiguous()

    def forward(self, x):
        batch_size = x.size(0)
        x_flat = x.view(-1)
        h = self.embed(x_flat)
        edge_index = self.single_edge_index.repeat(1, batch_size)
        shift = torch.arange(batch_size, device=x.device).repeat_interleave(self.single_edge_index.size(1)) * 81
        edge_index = edge_index + shift

        for conv, norm in zip(self.convs, self.norms):
            h_in = h
            h = conv(h, edge_index)
            h = norm(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = h + h_in

        logits = self.classifier(h)
        return logits.view(batch_size, 9, 9, 9)
