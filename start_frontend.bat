@echo off
REM Start the frontend development server (Windows)

echo Starting Sudoku Solver Frontend...
echo ===================================

cd frontend

REM Install dependencies if needed
if not exist "node_modules\" (
    echo Installing dependencies...
    call npm install
)

REM Create .env.local if it doesn't exist
if not exist ".env.local" (
    echo Creating .env.local from example...
    copy .env.local.example .env.local
)

REM Start the development server
echo Starting Next.js development server...
call npm run dev

pause

