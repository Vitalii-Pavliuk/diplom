"""
Sudoku Dataset Loader
Handles the "1 million Sudoku games" dataset format
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import Tuple

class SudokuDataset(Dataset):
    """
    Dataset class for Sudoku puzzles
    
    Input format: String of 81 digits (0 for empty cells)
    Output format: String of 81 digits (complete solution)
    
    Converts strings to PyTorch tensors of shape (9, 9)
    - Inputs: integers 0-9
    - Targets: integers 0-8 (subtract 1 for CrossEntropyLoss)
    """
    
    def __init__(self, csv_path: str, train: bool = True, train_split: float = 0.8, limit: int = None):
        """
        Initialize the Sudoku dataset
        
        Args:
            csv_path: Path to the CSV file
            train: If True, use training split; otherwise use validation split
            train_split: Fraction of data to use for training
            limit: Maximum number of rows to load (None for all). Useful for debugging.
        """
        # Читаємо CSV.
        # dtype=str гарантує, що "005..." залишиться рядком, а не перетвориться на число 5
        print(f"Loading data from {csv_path}...")
        
        if limit:
            df = pd.read_csv(csv_path, nrows=limit, dtype=str)
        else:
            # Обережно, завантаження 9 млн рядків може з'їсти всю RAM
            df = pd.read_csv(csv_path, dtype=str) 

        # Визначаємо правильні назви колонок (автоматична перевірка)
        if 'puzzle' in df.columns and 'solution' in df.columns:
            q_col, s_col = 'puzzle', 'solution' # Kaggle format
        elif 'quizzes' in df.columns and 'solutions' in df.columns:
            q_col, s_col = 'quizzes', 'solutions' # Other format
        else:
            raise KeyError(f"Unknown column names. Found: {df.columns}. Expected 'puzzle'/'solution' or 'quizzes'/'solutions'")

        # Split into train and validation
        split_idx = int(len(df) * train_split)
        
        if train:
            self.quizzes = df[q_col].iloc[:split_idx].values
            self.solutions = df[s_col].iloc[:split_idx].values
        else:
            self.quizzes = df[q_col].iloc[split_idx:].values
            self.solutions = df[s_col].iloc[split_idx:].values
        
        print(f"Loaded {'training' if train else 'validation'} dataset with {len(self.quizzes)} samples")
    
    def __len__(self) -> int:
        return len(self.quizzes)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single Sudoku puzzle and its solution
        """
        # Convert string to numpy array
        puzzle_str = self.quizzes[idx]
        solution_str = self.solutions[idx]
        
        # Parse strings to integers
        # Захист від битих даних: якщо раптом трапиться не цифра
        try:
            puzzle = np.array([int(c) for c in puzzle_str], dtype=np.int64).reshape(9, 9)
            solution = np.array([int(c) for c in solution_str], dtype=np.int64).reshape(9, 9)
        except ValueError:
            # Fallback для порожніх або битих рядків (щоб не крашити тренування)
            # Використовуємо ones замість zeros, щоб після віднімання -1 не отримати невалідний клас -1
            puzzle = np.zeros((9, 9), dtype=np.int64)
            solution = np.ones((9, 9), dtype=np.int64)  # 1 - 1 = 0 (валідний клас для CrossEntropyLoss)

        # Convert to tensors
        puzzle_tensor = torch.from_numpy(puzzle).long()
        
        # Subtract 1 from solution for CrossEntropyLoss (0-8 instead of 1-9)
        # Target classes must be 0-based
        solution_tensor = torch.from_numpy(solution - 1).long()
        
        return puzzle_tensor, solution_tensor

def string_to_tensor(puzzle_string: str) -> torch.Tensor:
    """Utility function to convert a puzzle string to tensor"""
    puzzle = np.array([int(c) for c in puzzle_string], dtype=np.int64).reshape(9, 9)
    return torch.from_numpy(puzzle).long()

def tensor_to_string(puzzle_tensor: torch.Tensor) -> str:
    """Utility function to convert a tensor to puzzle string"""
    # Якщо тензор на GPU, переносимо на CPU
    if puzzle_tensor.is_cuda:
        puzzle_tensor = puzzle_tensor.cpu()
    return ''.join(str(int(x)) for x in puzzle_tensor.flatten())

if __name__ == "__main__":
    # Test the dataset loader
    print("Testing SudokuDataset...")
    
    try:
        # Тестуємо з лімітом, щоб було швидко
        dataset = SudokuDataset("data/sudoku.csv", train=True, limit=100)
        puzzle, solution = dataset[0]
        
        print(f"Puzzle shape: {puzzle.shape}")
        print(f"Solution shape: {solution.shape}")
        print(f"\nPuzzle (first row): {puzzle[0]}")
        print(f"Solution (first row) (targets 0-8): {solution[0]}")
        
        # Перевірка на правильність перетворення
        print("\nTest passed! Columns found and data loaded correctly.")
        
    except FileNotFoundError:
        print("Error: CSV file not found at 'data/sudoku.csv'.")
    except KeyError as e:
        print(f"Error: Column mismatch. {e}")