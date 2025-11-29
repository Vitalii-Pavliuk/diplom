# Project Structure Overview

Complete file structure of the Sudoku Solver Thesis Project.

```
sudoku-thesis/
│
├── README.md                           # Main project documentation
├── QUICKSTART.md                       # Quick start guide
├── PROJECT_STRUCTURE.md                # This file
│
├── start_backend.sh                    # Backend startup script (Unix)
├── start_backend.bat                   # Backend startup script (Windows)
├── start_frontend.sh                   # Frontend startup script (Unix)
├── start_frontend.bat                  # Frontend startup script (Windows)
│
├── backend/                            # Python Backend
│   ├── README.md                       # Backend documentation
│   ├── requirements.txt                # Python dependencies
│   ├── .gitignore                      # Git ignore rules
│   │
│   ├── data/                           # Dataset directory
│   │   └── sudoku.csv                  # [Download required] 1M Sudoku dataset
│   │
│   ├── weights/                        # [Created during training] Model checkpoints
│   │   ├── baseline_best.pth          # Best baseline model
│   │   ├── advanced_best.pth          # Best advanced model
│   │   ├── gnn_best.pth               # Best GNN model
│   │   └── *_history.json             # Training history
│   │
│   ├── models/                         # Neural Network Models
│   │   ├── __init__.py                # Package initialization
│   │   ├── cnn_baseline.py            # Model A: Baseline CNN (~60K params)
│   │   ├── cnn_advanced.py            # Model B: ResNet CNN (~500K params)
│   │   └── gnn_model.py               # Model C: Graph NN (~300K params)
│   │
│   ├── dataset.py                      # Dataset loader & utilities
│   ├── train.py                        # Training script
│   ├── main.py                         # FastAPI backend server
│   └── example_usage.py                # Example usage demonstration
│
└── frontend/                           # Next.js Frontend
    ├── README.md                       # Frontend documentation
    ├── package.json                    # Node.js dependencies
    ├── tsconfig.json                   # TypeScript configuration
    ├── next.config.js                  # Next.js configuration
    ├── tailwind.config.js              # TailwindCSS configuration
    ├── postcss.config.js               # PostCSS configuration
    ├── .gitignore                      # Git ignore rules
    ├── .env.local.example              # Environment variables template
    ├── .env.local                      # [Create from example] Local config
    │
    ├── app/                            # Next.js App Router
    │   ├── layout.tsx                  # Root layout
    │   ├── page.tsx                    # Main page with logic
    │   └── globals.css                 # Global styles
    │
    ├── components/                     # React Components
    │   ├── SudokuBoard.tsx             # Interactive Sudoku grid
    │   └── Controls.tsx                # Control panel UI
    │
    └── lib/                            # Utilities
        └── api.ts                      # API client functions
```

## Key Files Explained

### Backend Files

#### `models/cnn_baseline.py`
- Simple 5-layer CNN
- Maintains 9×9 spatial structure
- One-hot encoding input
- Output: 9 classes per cell

#### `models/cnn_advanced.py`
- Deep ResNet with 20 residual blocks
- Skip connections prevent vanishing gradients
- More parameters = better capacity

#### `models/gnn_model.py`
- Graph-based approach
- 81 nodes (one per cell)
- Edges connect cells in same row/column/box
- Uses PyTorch Geometric

#### `dataset.py`
- Loads CSV dataset
- Converts strings to tensors
- Train/validation split
- Data utilities

#### `train.py`
- Complete training script
- Supports all 3 models
- Progress bars with tqdm
- Automatic checkpointing
- Learning rate scheduling

#### `main.py`
- FastAPI server
- REST API endpoints
- Model loading & switching
- CORS enabled for frontend

### Frontend Files

#### `app/page.tsx`
- Main application logic
- State management
- API integration
- Event handlers

#### `components/SudokuBoard.tsx`
- Interactive 9×9 grid
- Cell input handling
- Visual styling (user vs AI)
- Thick borders for 3×3 boxes

#### `components/Controls.tsx`
- Control buttons
- Model selection dropdown
- Import/Export functionality
- Instructions panel

#### `lib/api.ts`
- Axios-based API client
- Type-safe requests
- Error handling
- All backend endpoints

## File Relationships

```
Training Flow:
  data/sudoku.csv → dataset.py → train.py → weights/*.pth

API Flow:
  weights/*.pth → main.py → FastAPI → HTTP

Frontend Flow:
  page.tsx → api.ts → Backend → SudokuBoard.tsx
                             → Controls.tsx

Model Architecture:
  models/__init__.py exports:
    ├── CNNBaseline (cnn_baseline.py)
    ├── CNNAdvanced (cnn_advanced.py)
    └── GNNModel (gnn_model.py)
```

## Generated/Downloaded Files

These files are NOT in the repository and need to be created:

### Must Download:
- `backend/data/sudoku.csv` - Download the 1M Sudoku dataset

### Auto-Generated (Backend):
- `backend/venv/` - Python virtual environment
- `backend/__pycache__/` - Python cache
- `backend/weights/` - Model checkpoints (after training)

### Auto-Generated (Frontend):
- `frontend/node_modules/` - Node.js dependencies
- `frontend/.next/` - Next.js build cache
- `frontend/.env.local` - Local environment config

## Configuration Files

### Backend Configuration:
- `requirements.txt` - Python package versions
- `.gitignore` - Files to ignore in git

### Frontend Configuration:
- `package.json` - Node.js packages and scripts
- `tsconfig.json` - TypeScript compiler options
- `tailwind.config.js` - Tailwind CSS theme
- `next.config.js` - Next.js settings
- `.env.local.example` - Environment template

## Development Workflow

1. **Setup**: Run `start_backend` and `start_frontend` scripts
2. **Download**: Get Sudoku dataset → `backend/data/sudoku.csv`
3. **Train**: Run `train.py` for each model
4. **Develop**: Edit files, auto-reload works for both backend and frontend
5. **Test**: Use `example_usage.py` to test models programmatically

## Import Structure

### Backend:
```python
from models import CNNBaseline, CNNAdvanced, GNNModel
from dataset import SudokuDataset
```

### Frontend:
```typescript
import SudokuBoard from '@/components/SudokuBoard'
import Controls from '@/components/Controls'
import { solveSudoku } from '@/lib/api'
```

## Port Usage

- Backend: `http://localhost:8000`
- Frontend: `http://localhost:3000`

Make sure these ports are available before starting!

## Next Steps

1. Download the Sudoku dataset
2. Train at least one model
3. Start both servers
4. Open browser to `http://localhost:3000`
5. Test the application!

For detailed instructions, see QUICKSTART.md

