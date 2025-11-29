# Sudoku Solver - Frontend

Next.js frontend with TypeScript and TailwindCSS for interacting with the Sudoku solver API.

## Quick Start

1. **Install dependencies**:
```bash
npm install
```

2. **Configure environment**:
```bash
cp .env.local.example .env.local
```

Edit `.env.local`:
```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

3. **Start development server**:
```bash
npm run dev
```

Frontend will be available at `http://localhost:3000`

## Features

- **Interactive Sudoku Board**: Click and type to enter numbers
- **Visual Distinction**: User inputs (black) vs AI solutions (blue)
- **Model Selection**: Switch between CNN Baseline, CNN Advanced, and GNN
- **Confidence Display**: See model confidence in solution
- **Import/Export**: Save and load boards as JSON
- **Random Puzzles**: Generate sample puzzles
- **API Status**: Real-time connection indicator

## Components

### `SudokuBoard.tsx`
Interactive 9×9 Sudoku grid with:
- Cell input handling
- Visual styling for user/AI cells
- Thick borders for 3×3 boxes

### `Controls.tsx`
Control panel with:
- Solve button
- Clear board
- Random puzzle generation
- Model selection
- Import/Export functionality

### `api.ts`
API client for backend communication:
- Solve puzzle
- Switch models
- Get model info
- Health checks

## Building for Production

```bash
npm run build
npm start
```

## Development

```bash
# Run development server
npm run dev

# Lint code
npm run lint
```

