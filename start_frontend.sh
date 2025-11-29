#!/bin/bash
# Start the frontend development server

echo "Starting Sudoku Solver Frontend..."
echo "==================================="

cd frontend

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "Installing dependencies..."
    npm install
fi

# Create .env.local if it doesn't exist
if [ ! -f ".env.local" ]; then
    echo "Creating .env.local from example..."
    cp .env.local.example .env.local
fi

# Start the development server
echo "Starting Next.js development server..."
npm run dev

