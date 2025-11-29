'use client';

import React from 'react';

export type SudokuBoardData = number[][];

export interface CellState {
  value: number;
  isUserInput: boolean;
  isError?: boolean;
}

interface SudokuBoardProps {
  board: CellState[][];
  onCellChange: (row: number, col: number, value: string) => void;
  isLoading?: boolean;
}

export default function SudokuBoard({ board, onCellChange, isLoading = false }: SudokuBoardProps) {
  const handleInputChange = (row: number, col: number, value: string) => {
    // Only allow digits 0-9 or empty
    if (value === '' || /^[0-9]$/.test(value)) {
      onCellChange(row, col, value);
    }
  };

  const getCellClassName = (row: number, col: number, cell: CellState): string => {
    const baseClass = 'sudoku-cell';
    const classes = [baseClass];

    // Add user input or AI solution class
    if (cell.value !== 0) {
      if (cell.isUserInput) {
        classes.push('user-input');
      } else {
        classes.push('ai-solution');
      }
    }

    // Add error class
    if (cell.isError) {
      classes.push('error');
    }

    // Add thick borders for 3x3 boxes
    if ((col + 1) % 3 === 0 && col < 8) {
      classes.push('sudoku-cell-border-right');
    }
    if ((row + 1) % 3 === 0 && row < 8) {
      classes.push('sudoku-cell-border-bottom');
    }

    return classes.join(' ');
  };

  return (
    <div className="inline-block p-4 bg-white rounded-lg shadow-lg border-4 border-gray-800">
      <div className="grid grid-cols-9 gap-0">
        {board.map((row, rowIndex) =>
          row.map((cell, colIndex) => (
            <input
              key={`${rowIndex}-${colIndex}`}
              type="text"
              inputMode="numeric"
              maxLength={1}
              value={cell.value === 0 ? '' : cell.value}
              onChange={(e) => handleInputChange(rowIndex, colIndex, e.target.value)}
              disabled={isLoading}
              className={getCellClassName(rowIndex, colIndex, cell)}
              aria-label={`Cell ${rowIndex + 1}, ${colIndex + 1}`}
            />
          ))
        )}
      </div>
      
      {/* Legend */}
      <div className="mt-4 flex gap-4 text-sm">
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 bg-white border border-gray-300"></div>
          <span className="text-gray-700">User Input</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 bg-blue-50 border border-gray-300"></div>
          <span className="text-gray-700">AI Solution</span>
        </div>
      </div>
    </div>
  );
}

