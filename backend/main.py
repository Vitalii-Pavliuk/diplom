"""
FastAPI Backend for Sudoku Solver
Provides REST API endpoint for solving Sudoku puzzles using trained neural networks
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
import torch
import os
from typing import List, Optional
import numpy as np

from models import CNNBaseline, CNNAdvanced, GNNModel


# Initialize FastAPI app
app = FastAPI(
    title="Sudoku Solver API",
    description="Neural Network-based Sudoku Solver",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request and Response models
class SudokuBoard(BaseModel):
    """
    Sudoku board representation
    9x9 grid with values 0-9 (0 for empty cells)
    """
    board: List[List[int]]
    
    @validator('board')
    def validate_board(cls, v):
        # Check shape
        if len(v) != 9:
            raise ValueError("Board must have 9 rows")
        
        for row in v:
            if len(row) != 9:
                raise ValueError("Each row must have 9 columns")
            
            for cell in row:
                if not (0 <= cell <= 9):
                    raise ValueError("Cell values must be between 0 and 9")
        
        return v


class SudokuSolution(BaseModel):
    """
    Sudoku solution response
    """
    solution: List[List[int]]
    model_used: str
    confidence: Optional[float] = None


class ModelInfo(BaseModel):
    """
    Model information
    """
    current_model: str
    available_models: List[str]
    model_loaded: bool


# Global model state
class ModelState:
    def __init__(self):
        self.model = None
        self.model_name = "baseline"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_loaded = False


state = ModelState()


def load_model(model_name: str = "baseline", weights_path: Optional[str] = None):
    """
    Load a trained model
    
    Args:
        model_name: Name of the model architecture ('baseline', 'advanced', 'gnn')
        weights_path: Path to the weights file (optional)
    """
    print(f"Loading model: {model_name}")
    
    # Initialize model
    if model_name == "baseline":
        model = CNNBaseline(hidden_channels=128)
    elif model_name == "advanced":
        model = CNNAdvanced(hidden_channels=128, num_residual_blocks=20)
    elif model_name == "gnn":
        model = GNNModel(hidden_channels=128, num_layers=6)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Load weights if available
    if weights_path is None:
        weights_path = f"weights/{model_name}_best.pth"
    
    if os.path.exists(weights_path):
        print(f"Loading weights from: {weights_path}")
        checkpoint = torch.load(weights_path, map_location=state.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded weights (Epoch {checkpoint.get('epoch', 'unknown')})")
    else:
        print(f"⚠ Warning: No weights found at {weights_path}. Using untrained model.")
    
    model.to(state.device)
    model.eval()
    
    state.model = model
    state.model_name = model_name
    state.model_loaded = True
    
    print(f"✓ Model ready on {state.device}")


@app.on_event("startup")
async def startup_event():
    """
    Initialize the model on startup
    """
    print("=" * 60)
    print("Sudoku Solver API - Starting up")
    print("=" * 60)
    
    try:
        load_model("baseline")
    except Exception as e:
        print(f"⚠ Warning: Could not load model: {e}")
        print("API will start without a trained model.")
        state.model_loaded = False


@app.get("/")
async def root():
    """
    Root endpoint
    """
    return {
        "message": "Sudoku Solver API",
        "version": "1.0.0",
        "endpoints": {
            "solve": "POST /solve",
            "model_info": "GET /model",
            "switch_model": "POST /model/switch"
        }
    }


@app.get("/model", response_model=ModelInfo)
async def get_model_info():
    """
    Get information about the current model
    """
    return ModelInfo(
        current_model=state.model_name,
        available_models=["baseline", "advanced", "gnn"],
        model_loaded=state.model_loaded
    )


@app.post("/model/switch")
async def switch_model(model_name: str):
    """
    Switch to a different model
    
    Args:
        model_name: Name of the model to switch to
    """
    if model_name not in ["baseline", "advanced", "gnn"]:
        raise HTTPException(status_code=400, detail="Invalid model name")
    
    try:
        load_model(model_name)
        return {"message": f"Switched to {model_name} model", "success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


@app.post("/solve", response_model=SudokuSolution)
async def solve_sudoku(board: SudokuBoard):
    """
    Solve a Sudoku puzzle
    
    Args:
        board: 9x9 Sudoku board with 0s for empty cells
        
    Returns:
        Solved Sudoku board
    """
    if not state.model_loaded or state.model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please check server logs."
        )
    
    try:
        # Convert board to tensor
        board_array = np.array(board.board, dtype=np.int64)
        board_tensor = torch.from_numpy(board_array).unsqueeze(0).to(state.device)
        
        # Run inference
        with torch.no_grad():
            output = state.model(board_tensor)  # (1, 9, 9, 9)
            
            # Get predictions and confidence
            probabilities = torch.softmax(output, dim=-1)
            max_probs, predictions = torch.max(probabilities, dim=-1)
            
            # Convert predictions from 0-8 to 1-9
            predictions = predictions + 1
            
            # Calculate average confidence
            confidence = max_probs.mean().item()
        
        # Convert to list
        solution = predictions[0].cpu().numpy().tolist()
        
        return SudokuSolution(
            solution=solution,
            model_used=state.model_name,
            confidence=float(confidence)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error solving Sudoku: {str(e)}")


@app.post("/solve/batch")
async def solve_sudoku_batch(boards: List[SudokuBoard]):
    """
    Solve multiple Sudoku puzzles in batch
    
    Args:
        boards: List of 9x9 Sudoku boards
        
    Returns:
        List of solved Sudoku boards
    """
    if not state.model_loaded or state.model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please check server logs."
        )
    
    try:
        # Convert boards to tensor
        board_arrays = [np.array(b.board, dtype=np.int64) for b in boards]
        board_tensor = torch.from_numpy(np.stack(board_arrays)).to(state.device)
        
        # Run inference
        with torch.no_grad():
            output = state.model(board_tensor)  # (Batch, 9, 9, 9)
            
            # Get predictions
            probabilities = torch.softmax(output, dim=-1)
            max_probs, predictions = torch.max(probabilities, dim=-1)
            
            # Convert predictions from 0-8 to 1-9
            predictions = predictions + 1
        
        # Convert to list
        solutions = []
        for i in range(len(boards)):
            solution = predictions[i].cpu().numpy().tolist()
            confidence = max_probs[i].mean().item()
            
            solutions.append(SudokuSolution(
                solution=solution,
                model_used=state.model_name,
                confidence=float(confidence)
            ))
        
        return solutions
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error solving Sudoku batch: {str(e)}")


@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "model_loaded": state.model_loaded,
        "device": state.device
    }


if __name__ == "__main__":
    import uvicorn
    
    print("Starting Sudoku Solver API...")
    print(f"Device: {state.device}")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

