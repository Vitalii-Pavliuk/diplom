# Thesis Project: Effectiveness Analysis of Various Neural Network Architectures for Sudoku Solving

A comprehensive full-stack application that implements and compares three different neural network architectures for solving Sudoku puzzles.

## 🎯 Project Overview

This thesis project implements and analyzes the effectiveness of three neural network architectures:

1. **CNN Baseline** - Simple Convolutional Neural Network
2. **CNN Advanced** - Deep Residual Network (ResNet-style)
3. **Graph Neural Network (GNN)** - Graph-based approach using PyTorch Geometric

## 🏗️ Architecture

### Backend (Python)
- **Framework**: FastAPI
- **Deep Learning**: PyTorch, PyTorch Geometric
- **Data Processing**: Pandas, NumPy
- **API**: RESTful API with CORS support

### Frontend (Next.js)
- **Framework**: Next.js 14 (App Router)
- **Language**: TypeScript
- **Styling**: TailwindCSS
- **UI**: Modern, academic-style interface

## 📁 Project Structure

```
sudoku-thesis/
├── backend/
│   ├── data/                   # Dataset directory
│   ├── models/                 # Neural network architectures
│   │   ├── __init__.py
│   │   ├── cnn_baseline.py    # Model A: Baseline CNN
│   │   ├── cnn_advanced.py    # Model B: Advanced CNN (ResNet)
│   │   └── gnn_model.py       # Model C: Graph Neural Network
│   ├── dataset.py             # Data loading logic
│   ├── train.py               # Training script
│   ├── main.py                # FastAPI backend
│   └── requirements.txt       # Python dependencies
├── frontend/
│   ├── app/
│   │   ├── layout.tsx
│   │   ├── page.tsx           # Main application page
│   │   └── globals.css
│   ├── components/
│   │   ├── SudokuBoard.tsx    # Interactive Sudoku board
│   │   └── Controls.tsx       # Control panel
│   ├── lib/
│   │   └── api.ts             # API client
│   ├── package.json
│   └── tsconfig.json
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Node.js 18+
- CUDA-capable GPU (optional, for faster training)

### Backend Setup

1. **Create a virtual environment**:
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Download the dataset**:
   - Download the "1 million Sudoku games" dataset
   - Place the CSV file in `backend/data/sudoku.csv`
   - Expected format: CSV with columns `quizzes` and `solutions`
   - Each entry is a string of 81 digits (0 for empty cells)

4. **Train a model** (optional):
```bash
# Train baseline CNN
python train.py --model baseline --epochs 20 --batch-size 64

# Train advanced CNN
python train.py --model advanced --epochs 20 --batch-size 64

# Train GNN
python train.py --model gnn --epochs 20 --batch-size 32
```

5. **Start the API server**:
```bash
python main.py
```

The API will be available at `http://localhost:8000`

### Frontend Setup

1. **Install dependencies**:
```bash
cd frontend
npm install
```

2. **Create environment file**:
```bash
cp .env.local.example .env.local
```

Edit `.env.local` if your backend is running on a different URL.

3. **Start the development server**:
```bash
npm run dev
```

The frontend will be available at `http://localhost:3000`

## 🔬 Model Architectures

### 1. CNN Baseline (`cnn_baseline.py`)

A simple convolutional neural network that maintains spatial dimensions throughout:

- **Input**: (Batch, 9, 9) with values 0-9
- **Architecture**:
  - One-hot encoding (10 channels)
  - 5 convolutional layers with BatchNorm and ReLU
  - Maintains 9×9 spatial dimensions
  - Output layer projects to 9 classes per cell
- **Output**: (Batch, 9, 9, 9) logits
- **Parameters**: ~60K

### 2. CNN Advanced (`cnn_advanced.py`)

Deep residual network with skip connections:

- **Input**: (Batch, 9, 9) with values 0-9
- **Architecture**:
  - One-hot encoding (10 channels)
  - Initial convolution to expand channels
  - 20 residual blocks with skip connections
  - Prevents vanishing gradients
  - Output layer projects to 9 classes per cell
- **Output**: (Batch, 9, 9, 9) logits
- **Parameters**: ~500K

### 3. Graph Neural Network (`gnn_model.py`)

Graph-based approach treating each Sudoku cell as a node:

- **Graph Structure**:
  - 81 nodes (one per cell)
  - Edges connect cells in the same row, column, or 3×3 box
- **Architecture**:
  - Node features: One-hot encoding of digit
  - 6 GCN/GAT layers
  - Message passing between related cells
  - Output: 9-class prediction per node
- **Output**: (Batch, 9, 9, 9) logits
- **Parameters**: ~300K

## 📊 Training

### Training Command Examples

```bash
# Baseline CNN with default parameters
python train.py --model baseline --data data/sudoku.csv --epochs 20

# Advanced CNN with more parameters
python train.py --model advanced --hidden-channels 128 --num-residual-blocks 20 --epochs 30

# GNN with Graph Attention
python train.py --model gnn --num-gnn-layers 6 --use-gat --epochs 25

# Training on GPU with larger batch size
python train.py --model baseline --device cuda --batch-size 128 --lr 0.001
```

### Training Parameters

- `--model`: Model architecture (`baseline`, `advanced`, `gnn`)
- `--data`: Path to CSV dataset
- `--batch-size`: Batch size (default: 64)
- `--epochs`: Number of training epochs (default: 20)
- `--lr`: Learning rate (default: 0.001)
- `--hidden-channels`: Number of hidden channels (default: 128)
- `--device`: Device to use (`cuda` or `cpu`)

### Monitoring Training

The training script provides:
- Real-time progress bars with tqdm
- Per-epoch metrics (loss, cell accuracy, board accuracy)
- Automatic model checkpointing
- Learning rate scheduling
- Training history saved to JSON

## 🌐 API Endpoints

### `POST /solve`
Solve a Sudoku puzzle.

**Request**:
```json
{
  "board": [[0,0,0,2,6,0,7,0,1], ...]
}
```

**Response**:
```json
{
  "solution": [[4,3,5,2,6,9,7,8,1], ...],
  "model_used": "baseline",
  "confidence": 0.95
}
```

### `GET /model`
Get current model information.

### `POST /model/switch?model_name=advanced`
Switch to a different model.

### `GET /health`
Health check endpoint.

## 🎨 Frontend Features

- **Interactive Sudoku Board**: Click and type to enter numbers
- **Visual Distinction**: User inputs (black) vs AI solutions (blue)
- **Model Selection**: Switch between three neural network models
- **Confidence Display**: See the model's confidence in its solution
- **Import/Export**: Save and load boards as JSON
- **Random Puzzles**: Generate random sample puzzles
- **Responsive Design**: Clean, academic-style interface
- **Real-time API Status**: Connection indicator

## 📈 Performance Metrics

The training script tracks:
- **Cell Accuracy**: Percentage of correctly predicted cells
- **Empty Cell Accuracy**: Accuracy on initially empty cells only
- **Board Accuracy**: Percentage of completely solved puzzles
- **Confidence**: Average prediction confidence

## 🔧 Development

### Running Tests

Backend models can be tested individually:
```bash
python -m models.cnn_baseline
python -m models.cnn_advanced
python -m models.gnn_model
```

### Dataset Testing
```bash
python dataset.py
```

### API Testing
```bash
# Start the server
python main.py

# In another terminal
curl -X POST "http://localhost:8000/solve" \
  -H "Content-Type: application/json" \
  -d '{"board": [[0,0,0,2,6,0,7,0,1], ...]}'
```

## 📝 Dataset Format

The project uses the "1 million Sudoku games" dataset:

- **Format**: CSV with columns `quizzes` and `solutions`
- **Quiz**: String of 81 digits (0 = empty cell)
- **Solution**: String of 81 digits (complete solution)
- **Example**:
  ```
  Quiz: "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
  Solution: "534678912672195348198342567859761423426853791713924856961537284287419635345286179"
  ```

You can download this dataset from Kaggle or similar sources.

## 🎓 Thesis Components

This project is suitable for analyzing:

1. **Model Comparison**: Compare three different architectures
2. **Performance Analysis**: Accuracy, speed, model size
3. **Architecture Benefits**: Why GNN might outperform CNN
4. **Visualization**: Training curves, confusion matrices
5. **Ablation Studies**: Effect of hyperparameters

## 🛠️ Technologies Used

### Backend
- PyTorch 2.0+
- PyTorch Geometric
- FastAPI
- Uvicorn
- Pandas & NumPy

### Frontend
- Next.js 14
- TypeScript
- TailwindCSS
- Axios
- Lucide Icons

## 📄 License

This is a thesis project for educational purposes.

## 🤝 Contributing

This is a thesis project, but suggestions and improvements are welcome!

## 📧 Contact

For questions about this thesis project, please contact the repository owner.

---

**Note**: Make sure to download the Sudoku dataset and place it in `backend/data/sudoku.csv` before training models.

