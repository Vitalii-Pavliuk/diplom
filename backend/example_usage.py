"""
Example usage of the Sudoku models
Demonstrates how to use the models programmatically
"""

import torch
from models import CNNBaseline, CNNAdvanced, GNNModel
import numpy as np


def example_puzzle():
    """
    Return an example Sudoku puzzle
    """
    # Easy puzzle (as string)
    puzzle_str = "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
    
    # Convert to 9x9 array
    puzzle = np.array([int(c) for c in puzzle_str]).reshape(9, 9)
    return puzzle


def solve_with_model(model, puzzle):
    """
    Solve a puzzle using a model
    
    Args:
        model: PyTorch model
        puzzle: 9x9 numpy array with values 0-9
        
    Returns:
        solution: 9x9 numpy array with predicted values
        confidence: Average prediction confidence
    """
    # Convert to tensor
    puzzle_tensor = torch.from_numpy(puzzle).unsqueeze(0).long()
    
    # Inference
    model.eval()
    with torch.no_grad():
        output = model(puzzle_tensor)  # (1, 9, 9, 9)
        
        # Get predictions and confidence
        probabilities = torch.softmax(output, dim=-1)
        confidence, predictions = torch.max(probabilities, dim=-1)
        
        # Convert from 0-8 to 1-9
        predictions = predictions + 1
    
    # Convert to numpy
    solution = predictions[0].numpy()
    avg_confidence = confidence.mean().item()
    
    return solution, avg_confidence


def print_board(board, title="Board"):
    """
    Pretty print a Sudoku board
    """
    print(f"\n{title}:")
    print("─" * 25)
    for i, row in enumerate(board):
        if i % 3 == 0 and i != 0:
            print("─" * 25)
        row_str = ""
        for j, cell in enumerate(row):
            if j % 3 == 0 and j != 0:
                row_str += "│ "
            row_str += str(cell) + " "
        print(row_str)
    print("─" * 25)


def main():
    print("=" * 50)
    print("Sudoku Solver - Example Usage")
    print("=" * 50)
    
    # Get example puzzle
    puzzle = example_puzzle()
    print_board(puzzle, "Input Puzzle")
    
    # Try each model
    models = {
        'Baseline CNN': CNNBaseline(hidden_channels=64),
        'Advanced CNN': CNNAdvanced(hidden_channels=128, num_residual_blocks=20),
        'GNN': GNNModel(hidden_channels=128, num_layers=6),
    }
    
    for model_name, model in models.items():
        print(f"\n{'=' * 50}")
        print(f"Solving with {model_name}")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"{'=' * 50}")
        
        # Solve
        solution, confidence = solve_with_model(model, puzzle)
        print_board(solution, f"Solution ({model_name})")
        print(f"Confidence: {confidence:.2%}")
        
        # Check if valid (basic check: all digits 1-9)
        valid = np.all((solution >= 1) & (solution <= 9))
        print(f"Valid solution format: {valid}")
    
    print("\n" + "=" * 50)
    print("Note: These are untrained models, so solutions are random!")
    print("Train the models first using train.py")
    print("=" * 50)


if __name__ == "__main__":
    main()

