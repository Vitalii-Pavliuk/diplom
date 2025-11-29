# Quick Start Guide

Get the Sudoku Solver up and running in minutes!

## Prerequisites

- Python 3.9 or higher
- Node.js 18 or higher
- pip and npm installed

## Step-by-Step Setup

### 1. Download the Dataset

Before training any models, you need the Sudoku dataset:

1. Download the "1 million Sudoku games" dataset from Kaggle or similar source
2. The file should be a CSV with columns: `quizzes` and `solutions`
3. Place it at: `backend/data/sudoku.csv`

Example format:
```csv
quizzes,solutions
004300209005009001070060043006002087190007400050083000600000105003508690042910300,864371259325849761971265843436192587198657432257483916689734125713528694542916378
...
```

### 2. Backend Setup (Windows)

```batch
# Run the automated setup script
start_backend.bat
```

**Or manually**:
```batch
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

### 2. Backend Setup (Mac/Linux)

```bash
# Run the automated setup script
chmod +x start_backend.sh
./start_backend.sh
```

**Or manually**:
```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

The backend will start at: `http://localhost:8000`

### 3. Frontend Setup (Windows)

**Open a new terminal**, then:

```batch
# Run the automated setup script
start_frontend.bat
```

**Or manually**:
```batch
cd frontend
npm install
copy .env.local.example .env.local
npm run dev
```

### 3. Frontend Setup (Mac/Linux)

**Open a new terminal**, then:

```bash
# Run the automated setup script
chmod +x start_frontend.sh
./start_frontend.sh
```

**Or manually**:
```bash
cd frontend
npm install
cp .env.local.example .env.local
npm run dev
```

The frontend will start at: `http://localhost:3000`

### 4. Open in Browser

Navigate to: **http://localhost:3000**

You should see the Sudoku Solver interface!

## Training Models (Optional)

The backend can run with untrained models, but for better results, train them first:

### Train Baseline CNN (fastest)
```bash
cd backend
python train.py --model baseline --epochs 20 --batch-size 64
```

### Train Advanced CNN (best accuracy)
```bash
python train.py --model advanced --epochs 30 --batch-size 64
```

### Train GNN (graph-based approach)
```bash
python train.py --model gnn --epochs 25 --batch-size 32
```

Training times (on GPU):
- Baseline: ~30 minutes for 20 epochs
- Advanced: ~2 hours for 30 epochs
- GNN: ~1.5 hours for 25 epochs

## Using the Application

1. **Enter a puzzle**: Click cells and type numbers (1-9)
2. **Or load a sample**: Click "Generate Random"
3. **Select a model**: Choose from the dropdown (Baseline/Advanced/GNN)
4. **Solve**: Click "Solve with AI"
5. **View results**: AI solutions appear in blue, your inputs in black

## Troubleshooting

### Backend won't start
- Make sure Python 3.9+ is installed: `python --version`
- Check if port 8000 is available
- Verify PyTorch installation: `pip show torch`

### Frontend won't start
- Make sure Node.js 18+ is installed: `node --version`
- Clear npm cache: `npm cache clean --force`
- Delete `node_modules` and reinstall: `rm -rf node_modules && npm install`

### API Connection Error
- Ensure backend is running at `http://localhost:8000`
- Check `.env.local` has correct API URL
- Check browser console for CORS errors

### Training Errors
- Verify dataset is at `backend/data/sudoku.csv`
- Check CSV format (must have `quizzes` and `solutions` columns)
- Ensure enough disk space for model checkpoints

### CUDA/GPU Issues
If you don't have a GPU, training will use CPU (slower but works):
```bash
python train.py --model baseline --device cpu
```

## Next Steps

- Train all three models to compare performance
- Experiment with different hyperparameters
- Modify the frontend to add new features
- Analyze model performance for your thesis

## Support

For issues or questions:
1. Check the main README.md
2. Review backend/README.md and frontend/README.md
3. Check the code comments in each file
4. Review the example_usage.py for programmatic usage

Happy Sudoku Solving! 🎓

