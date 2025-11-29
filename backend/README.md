# Sudoku Solver - Backend

Python backend using PyTorch and FastAPI for neural network-based Sudoku solving.

## Quick Start

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Download dataset**:
   - Place the "1 million Sudoku games" CSV in `data/sudoku.csv`
   - Format: CSV with `quizzes` and `solutions` columns

3. **Train a model**:
```bash
python train.py --model baseline --epochs 20
```

4. **Start API server**:
```bash
python main.py
```

API will be available at `http://localhost:8000`

## Model Architectures

### Baseline CNN (`models/cnn_baseline.py`)
- Simple 5-layer CNN
- Maintains 9×9 spatial dimensions
- ~60K parameters

### Advanced CNN (`models/cnn_advanced.py`)
- Deep ResNet with 20 residual blocks
- Skip connections prevent vanishing gradients
- ~500K parameters

### Graph Neural Network (`models/gnn_model.py`)
- Treats Sudoku as a graph problem
- 81 nodes with constraint-based edges
- 6 GCN/GAT layers
- ~300K parameters

## Training Commands

```bash
# Train baseline model
python train.py --model baseline --batch-size 64 --epochs 20

# Train advanced model
python train.py --model advanced --hidden-channels 128 --num-residual-blocks 20 --epochs 30

# Train GNN with Graph Attention
python train.py --model gnn --num-gnn-layers 6 --use-gat --epochs 25
```

## API Endpoints

- `POST /solve` - Solve a Sudoku puzzle
- `GET /model` - Get current model info
- `POST /model/switch` - Switch models
- `GET /health` - Health check

## Testing Models

Test individual models:
```bash
python -m models.cnn_baseline
python -m models.cnn_advanced
python -m models.gnn_model
```

Test dataset loader:
```bash
python dataset.py
```

