"""
Model C: RNN (LSTM) for Sudoku Solving
Uses LSTM to process Sudoku as a sequence
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SudokuRNN(nn.Module):
    """
    RNN Model (LSTM) for Sudoku
    
    Architecture:
    - Input: (Batch, 81) flattened sequence with values 0-9
    - Embedding layer: 10 (digits 0-9) -> 64
    - LSTM layer: 64 -> 128, 2 layers, bidirectional
    - Fully Connected layer: 256 (128*2 bidirectional) -> 9 classes
    - Output: (Batch, 9, 9, 9) logits for each cell
    """
    
    def __init__(self, embedding_dim: int = 64, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.1):
        super(SudokuRNN, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Embedding layer: 10 classes (0-9) -> embedding_dim
        self.embedding = nn.Embedding(10, embedding_dim)
        
        # LSTM layer: bidirectional, so output will be hidden_size * 2
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully Connected layer: (hidden_size * 2) -> 9 classes
        # Bidirectional LSTM outputs hidden_size * 2
        self.fc = nn.Linear(hidden_size * 2, 9)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (Batch, 81) with values 0-9
            
        Returns:
            Output tensor of shape (Batch, 9, 9, 9) with logits
        """
        # x shape: (Batch, 81)
        
        # Embedding: (Batch, 81) -> (Batch, 81, embedding_dim)
        x = self.embedding(x)
        
        # LSTM: (Batch, 81, embedding_dim) -> (Batch, 81, hidden_size * 2)
        lstm_out, _ = self.lstm(x)
        
        # Apply dropout
        lstm_out = self.dropout(lstm_out)
        
        # Fully Connected: (Batch, 81, hidden_size * 2) -> (Batch, 81, 9)
        output = self.fc(lstm_out)
        
        # Reshape to (Batch, 9, 9, 9)
        output = output.view(-1, 9, 9, 9)
        
        return output


if __name__ == "__main__":
    # Test the model
    print("Testing SudokuRNN model...")
    
    model = SudokuRNN(embedding_dim=64, hidden_size=128, num_layers=2)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create a dummy input (flattened 9x9 board)
    batch_size = 4
    dummy_input = torch.randint(0, 10, (batch_size, 81))
    
    # Forward pass
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output should be: (Batch={batch_size}, 9, 9, 9)")
    
    # Test prediction
    predictions = torch.argmax(output, dim=-1)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Predictions range: {predictions.min()}-{predictions.max()}")
