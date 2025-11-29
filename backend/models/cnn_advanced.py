"""
Model B: Advanced CNN (ResNet-style) for Sudoku Solving
Deep residual network with skip connections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Residual Block with skip connection
    """
    
    def __init__(self, channels: int):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with skip connection
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor with skip connection added
        """
        identity = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Add skip connection
        out += identity
        out = F.relu(out)
        
        return out


class CNNAdvanced(nn.Module):
    """
    Advanced CNN Model (ResNet-style) for Sudoku
    
    Architecture:
    - Input: (Batch, 9, 9) with values 0-9
    - One-Hot encode to 10 channels
    - Initial convolution to expand channels
    - 15-20 Residual blocks
    - Output: (Batch, 9, 9, 9) logits for each cell
    """
    
    def __init__(self, hidden_channels: int = 128, num_residual_blocks: int = 20):
        super(CNNAdvanced, self).__init__()
        
        self.hidden_channels = hidden_channels
        self.num_residual_blocks = num_residual_blocks
        
        # Initial convolution to expand from 10 to hidden_channels
        self.initial_conv = nn.Conv2d(10, hidden_channels, kernel_size=3, padding=1)
        self.initial_bn = nn.BatchNorm2d(hidden_channels)
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_channels) for _ in range(num_residual_blocks)
        ])
        
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
        
        # Initial convolution
        x = F.relu(self.initial_bn(self.initial_conv(x)))
        
        # Pass through residual blocks
        for residual_block in self.residual_blocks:
            x = residual_block(x)
        
        # Output layer
        x = self.output(x)  # (Batch, 9, 9, 9)
        
        # Rearrange to (Batch, 9, 9, 9)
        x = x.permute(0, 2, 3, 1)  # (Batch, H, W, Classes)
        
        return x


if __name__ == "__main__":
    # Test the model
    print("Testing CNNAdvanced model...")
    
    model = CNNAdvanced(hidden_channels=128, num_residual_blocks=20)
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

