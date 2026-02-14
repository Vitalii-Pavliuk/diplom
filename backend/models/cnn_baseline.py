"""
Model A: Baseline CNN for Sudoku Solving
A simple CNN that maintains spatial dimensions (9x9) throughout
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNBaseline(nn.Module):
    """
    Baseline CNN Model for Sudoku
    
    Architecture:
    - Input: (Batch, 9, 9) with values 0-9
    - One-Hot encode to 10 channels
    - 5 Convolutional layers with BatchNorm and ReLU
    - Output: (Batch, 9, 9, 9) logits for each cell
    """
    
    def __init__(self, hidden_channels: int = 64, dropout: float = 0.1):
        super(CNNBaseline, self).__init__()
        
        self.hidden_channels = hidden_channels
        self.dropout = dropout
        
        # Convolutional layers (maintaining 9x9 spatial dimensions)
        self.conv1 = nn.Conv2d(10, hidden_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.dropout1 = nn.Dropout2d(dropout)
        
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.dropout2 = nn.Dropout2d(dropout)
        
        self.conv3 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(hidden_channels)
        self.dropout3 = nn.Dropout2d(dropout)
        
        self.conv4 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(hidden_channels)
        self.dropout4 = nn.Dropout2d(dropout)
        
        self.conv5 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(hidden_channels)
        self.dropout5 = nn.Dropout2d(dropout)
        
        # Output layer: Project to 9 classes per cell
        self.output = nn.Conv2d(hidden_channels, 9, kernel_size=1)
    
    def one_hot_encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert input tensor to one-hot encoding
        
        Args:
            x: Tensor of shape (Batch, 9, 9) with values 0-9
            
        Returns:
            One-hot tensor of shape (Batch, 10, 9, 9)
        """
        # Create one-hot encoding for 10 classes (0-9)
        batch_size = x.size(0)
        one_hot = torch.zeros(batch_size, 10, 9, 9, device=x.device)
        
        # Scatter the one-hot values
        x_expanded = x.unsqueeze(1)  # (Batch, 1, 9, 9)
        one_hot.scatter_(1, x_expanded, 1.0)
        
        return one_hot
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (Batch, 9, 9) with values 0-9
            
        Returns:
            Output tensor of shape (Batch, 9, 9, 9) with logits
        """
        # One-hot encode the input
        x = self.one_hot_encode(x)  # (Batch, 10, 9, 9)
        
        # Convolutional layers with BatchNorm, ReLU, and Dropout2d
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout3(x)
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.dropout4(x)
        
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.dropout5(x)
        
        # Output layer
        x = self.output(x)  # (Batch, 9, 9, 9)
        
        # Rearrange to (Batch, 9, 9, 9)
        x = x.permute(0, 2, 3, 1)  # (Batch, H, W, Classes)
        
        return x


if __name__ == "__main__":
    # Test the model
    print("Testing CNNBaseline model...")
    
    model = CNNBaseline(hidden_channels=64)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
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

