# 🧠 Sudoku Solver - Backend

Python backend using PyTorch and FastAPI for neural network-based Sudoku solving.

## 🚀 Quick Start

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Download dataset**:
   - Place the "1 million Sudoku games" CSV in `data/sudoku.csv`
   - Format: CSV with `puzzle`/`solution` OR `quizzes`/`solutions` columns
   - Each row: 81 digits (0 = empty cell)

3. **Train a model**:
```bash
python train.py --model baseline --epochs 20
```

4. **Start API server**:
```bash
python main.py
```

API will be available at `http://localhost:8000`

---

## 🧠 Model Architectures

### 🔷 Baseline CNN (`models/cnn_baseline.py`)
- **Type**: Simple Convolutional Neural Network
- **Layers**: 5 Conv2D layers with BatchNorm + ReLU
- **Features**: Maintains 9×9 spatial dimensions throughout
- **Parameters**: ~60,000
- **Speed**: ⚡⚡⚡ Very Fast
- **Best for**: Quick prototyping and baseline comparison

### 🔷 Advanced CNN (`models/cnn_advanced.py`)
- **Type**: Deep Residual Network (ResNet-style)
- **Layers**: 20 Residual Blocks with skip connections
- **Features**: Skip connections prevent vanishing gradients
- **Parameters**: ~500,000
- **Speed**: ⚡⚡ Fast
- **Best for**: High accuracy with reasonable speed

### 🔷 Graph Neural Network (`models/gnn_model.py`)
- **Type**: Graph Attention Network (GAT)
- **Structure**: 81 nodes (cells) with constraint-based edges
- **Layers**: 8 GAT layers with 4 attention heads each
- **Features**: Message passing between related cells
- **Parameters**: ~300,000
- **Speed**: ⚡ Slower (graph processing overhead)
- **Best for**: Theoretical best fit for Sudoku structure

### 🔷 RNN (LSTM) (`models/rnn_model.py`)
- **Type**: Bidirectional LSTM
- **Input**: Flattened 81-position sequence
- **Layers**: 2 LSTM layers (forward + backward)
- **Features**: Captures sequential dependencies
- **Parameters**: ~200,000
- **Speed**: ⚡⚡ Fast
- **Best for**: Experimental comparison (may lose 2D structure)

## 🎓 Training Commands

### Basic Training

```bash
# Train baseline model (fastest, ~10 min on CPU)
python train.py --model baseline --batch-size 64 --epochs 20

# Train advanced model (~30 min on CPU)
python train.py --model advanced --hidden-channels 128 --num-residual-blocks 20 --epochs 30

# Train GNN with Graph Attention (~60 min on CPU)
python train.py --model gnn --num-gnn-layers 8 --epochs 25 --batch-size 32

# Train RNN (LSTM) (~15 min on CPU)
python train.py --model rnn --epochs 20 --batch-size 64
```

### Advanced Options

```bash
# Train on GPU with larger batch
python train.py --model gnn --device cuda --batch-size 128 --epochs 30

# Debug training (small dataset)
python train.py --model baseline --limit 1000 --epochs 5

# Resume from checkpoint
python train.py --model gnn --resume weights/gnn_last.pth --epochs 50

# Custom hyperparameters
python train.py \
    --model advanced \
    --hidden-channels 256 \
    --num-residual-blocks 30 \
    --lr 0.0005 \
    --weight-decay 1e-4 \
    --batch-size 32
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | baseline | Model architecture (baseline/advanced/gnn/rnn) |
| `--data` | data/sudoku.csv | Path to dataset |
| `--batch-size` | 64 | Batch size for training |
| `--epochs` | 20 | Number of training epochs |
| `--lr` | 0.001 | Learning rate |
| `--device` | auto | Device (cuda/cpu) |
| `--hidden-channels` | 128 | Hidden channels for CNN/GNN |
| `--num-residual-blocks` | 20 | Residual blocks (advanced CNN) |
| `--num-gnn-layers` | 8 | GNN layers |
| `--resume` | None | Path to checkpoint to resume from |
| `--limit` | None | Limit dataset size (for debugging) |

## 🌐 API Endpoints

### `POST /solve` - Solve Sudoku

**Request:**
```json
{
  "board": [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    ...
  ]
}
```

**Response:**
```json
{
  "solution": [[5, 3, 4, ...], ...],
  "model_used": "baseline",
  "confidence": 0.9523
}
```

### `GET /model` - Model Info

**Response:**
```json
{
  "current_model": "baseline",
  "available_models": ["baseline", "advanced", "gnn", "rnn"],
  "model_loaded": true
}
```

### `POST /model/switch?model_name=gnn` - Switch Model

**Response:**
```json
{
  "message": "Switched to gnn model",
  "success": true
}
```

### `GET /health` - Health Check

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda"
}
```

### `POST /solve/batch` - Batch Solving

Solve multiple puzzles at once for better throughput.

## 🧪 Testing & Development

### Test Individual Models

```bash
# Test model architecture and forward pass
python -m models.cnn_baseline
python -m models.cnn_advanced
python -m models.gnn_model
python -m models.rnn_model

# Test dataset loader
python dataset.py

# Test inference with example puzzles
python example_usage.py
```

### Model Architecture Details

#### Input/Output Format

All models accept and return the same format:

```python
# Input format (all models)
# CNN/GNN: (Batch, 9, 9) with values 0-9
# RNN:     (Batch, 81) with values 0-9 (flattened)

input = torch.randint(0, 10, (batch, 9, 9))  # CNN/GNN
input = torch.randint(0, 10, (batch, 81))    # RNN

# Output format (all models)
output = model(input)  # (Batch, 9, 9, 9)
# Dimensions: [Batch, Height, Width, Classes]
# Classes: 9 logits for digits 1-9

# Get predictions
predictions = torch.argmax(output, dim=-1)  # (Batch, 9, 9) with values 0-8
predictions = predictions + 1  # Convert to 1-9
```

#### Important: Target Encoding

```python
# For training with CrossEntropyLoss
targets = solution - 1  # Convert 1-9 to 0-8 (class indices)
```

CrossEntropyLoss expects:
- **Outputs**: (N, Classes) with any real numbers (logits)
- **Targets**: (N,) with class indices 0 to Classes-1

### Google Colab

See `colab_example.ipynb` for:
- Cloud-based training with free GPU
- Step-by-step instructions
- Visualization of results
- All 4 models working examples

**Note for RNN:**
- RNN requires flattened input `(batch, 81)` instead of `(batch, 9, 9)`
- RNN doesn't need PyTorch Geometric (unlike GNN)
- All other aspects are identical to CNN/GNN

---

## 📊 Expected Performance

After training on 1M+ puzzles:

| Metric | Baseline | Advanced | GNN | RNN |
|--------|----------|----------|-----|-----|
| Cell Accuracy | 85-90% | 92-95% | 93-96% | 80-88% |
| Board Accuracy | 30-40% | 60-75% | 65-80% | 25-35% |
| Training Time | 10 min | 30 min | 60 min | 15 min |
| Inference Speed | 5ms | 8ms | 25ms | 7ms |

*(on modern CPU, times are approximate)*

---

## 🔧 Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
python train.py --model gnn --batch-size 16
```

### PyTorch Geometric Installation Issues
```bash
# Install specific version for your CUDA
pip install torch-geometric -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

### Model Not Loading
```bash
# Check if weights exist
ls weights/

# Train from scratch if needed
python train.py --model baseline --epochs 20
```

---

## 📚 Additional Resources

- [Main README](../README.md) - Full project documentation
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

---

**For detailed explanation of model architectures and training process, see [main README](../README.md)**

