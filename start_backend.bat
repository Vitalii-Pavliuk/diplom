@echo off
REM Start the backend API server (Windows)

echo Starting Sudoku Solver Backend...
echo =================================

cd backend

REM Check if virtual environment exists
if not exist "venv\" (
    echo Virtual environment not found. Creating one...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies if needed
if not exist "venv\installed" (
    echo Installing dependencies...
    pip install -r requirements.txt
    type nul > venv\installed
)

REM Start the server
echo Starting FastAPI server...
python main.py

pause

