"""
Model C: Graph Neural Network (GNN) for Sudoku Solving
Uses PyTorch Geometric to treat Sudoku as a graph problem
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data
import numpy as np


class GNNModel(nn.Module):
    """
    Graph Neural Network Model for Sudoku
    
    Architecture:
    - Treat each Sudoku cell as a graph node (81 nodes)
    - Edges connect cells in same row, column, or 3x3 box
    - Node features: One-hot encoding of digit (10 classes)
    - Multiple GCN/GAT layers
    - Output: Predict digit class (0-8) for each node
    """
    
    def __init__(self, 
                 hidden_channels: int = 128, 
                 num_layers: int = 6,
                 use_gat: bool = False,
                 dropout: float = 0.1):
        super(GNNModel, self).__init__()
        
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.use_gat = use_gat
        self.dropout = dropout
        
        # Input projection: 10 -> hidden_channels
        self.input_proj = nn.Linear(10, hidden_channels)
        
        # GNN layers
        self.gnn_layers = nn.ModuleList()
        
        for i in range(num_layers):
            if use_gat:
                # Use Graph Attention Network
                self.gnn_layers.append(
                    GATConv(hidden_channels, hidden_channels // 8, heads=8, dropout=dropout)
                )
            else:
                # Use Graph Convolutional Network
                self.gnn_layers.append(
                    GCNConv(hidden_channels, hidden_channels)
                )
        
        # Output layer: Project to 9 classes
        self.output = nn.Linear(hidden_channels, 9)
        
        # Precompute edge indices for Sudoku graph
        self.register_buffer('edge_index', self._create_sudoku_edges())
    
    def _create_sudoku_edges(self) -> torch.Tensor:
        """
        Create edge indices for the Sudoku graph
        Connects cells in the same row, column, or 3x3 box
        
        Returns:
            Edge index tensor of shape (2, num_edges)
        """
        edges = []
        
        # Helper function to get cell index
        def cell_idx(row, col):
            return row * 9 + col
        
        # Helper function to get 3x3 box index
        def box_idx(row, col):
            return (row // 3, col // 3)
        
        for i in range(9):
            for j in range(9):
                current = cell_idx(i, j)
                
                # Connect to cells in same row
                for k in range(9):
                    if k != j:
                        edges.append([current, cell_idx(i, k)])
                
                # Connect to cells in same column
                for k in range(9):
                    if k != i:
                        edges.append([current, cell_idx(k, j)])
                
                # Connect to cells in same 3x3 box
                box_r, box_c = box_idx(i, j)
                for r in range(box_r * 3, (box_r + 1) * 3):
                    for c in range(box_c * 3, (box_c + 1) * 3):
                        if r != i or c != j:
                            edges.append([current, cell_idx(r, c)])
        
        # Remove duplicates and convert to tensor
        edges = list(set([tuple(e) for e in edges]))
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        return edge_index
    
    def one_hot_encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert input tensor to one-hot encoding
        
        Args:
            x: Tensor of shape (Batch, 9, 9) with values 0-9
            
        Returns:
            One-hot tensor of shape (Batch, 81, 10)
        """
        batch_size = x.size(0)
        
        # Flatten to (Batch, 81)
        x_flat = x.view(batch_size, 81)
        
        # Create one-hot encoding
        one_hot = torch.zeros(batch_size, 81, 10, device=x.device)
        one_hot.scatter_(2, x_flat.unsqueeze(2), 1.0)
        
        return one_hot
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (Batch, 9, 9) with values 0-9
            
        Returns:
            Output tensor of shape (Batch, 9, 9, 9) with logits
        """
        batch_size = x.size(0)
        
        # One-hot encode: (Batch, 9, 9) -> (Batch, 81, 10)
        x = self.one_hot_encode(x)
        
        # Project to hidden dimensions: (Batch, 81, 10) -> (Batch, 81, hidden_channels)
        x = self.input_proj(x)
        x = F.relu(x)
        
        # Process each graph in the batch
        outputs = []
        
        for b in range(batch_size):
            # Get node features for this graph
            node_features = x[b]  # (81, hidden_channels)
            
            # Apply GNN layers
            for i, gnn_layer in enumerate(self.gnn_layers):
                node_features = gnn_layer(node_features, self.edge_index)
                node_features = F.relu(node_features)
                node_features = F.dropout(node_features, p=self.dropout, training=self.training)
            
            outputs.append(node_features)
        
        # Stack batch: (Batch, 81, hidden_channels)
        x = torch.stack(outputs, dim=0)
        
        # Output projection: (Batch, 81, hidden_channels) -> (Batch, 81, 9)
        x = self.output(x)
        
        # Reshape to (Batch, 9, 9, 9)
        x = x.view(batch_size, 9, 9, 9)
        
        return x


if __name__ == "__main__":
    # Test the model
    print("Testing GNNModel...")
    
    model = GNNModel(hidden_channels=128, num_layers=6, use_gat=False)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Number of edges in Sudoku graph: {model.edge_index.size(1)}")
    
    # Create a dummy input
    batch_size = 4
    dummy_input = torch.randint(0, 10, (batch_size, 9, 9))
    
    # Forward pass
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output should be: (Batch={batch_size}, 9, 9, 9)")
    
    # Test prediction
    predictions = torch.argmax(output, dim=-1)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Predictions range: {predictions.min()}-{predictions.max()}")

