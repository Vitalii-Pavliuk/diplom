"""
Neural Network Models for Sudoku Solving

ВАЖЛИВО: Цей файл використовує lazy imports, щоб уникнути завантаження
torch_geometric при імпорті CNN/RNN моделей.

Використовуйте прямі імпорти:
    from models.cnn_baseline import CNNBaseline
    from models.cnn_advanced import CNNAdvanced
    from models.gnn_model import GNNModel  # Потребує torch_geometric!
    from models.rnn_model import SudokuRNN
"""

# Lazy imports - завантажуються тільки при використанні
def __getattr__(name):
    """Lazy import моделей"""
    if name == 'CNNBaseline':
        from .cnn_baseline import CNNBaseline
        return CNNBaseline
    elif name == 'CNNAdvanced':
        from .cnn_advanced import CNNAdvanced
        return CNNAdvanced
    elif name == 'GNNModel':
        from .gnn_model import GNNModel
        return GNNModel
    elif name == 'SudokuRNN':
        from .rnn_model import SudokuRNN
        return SudokuRNN
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ['CNNBaseline', 'CNNAdvanced', 'GNNModel', 'SudokuRNN']

