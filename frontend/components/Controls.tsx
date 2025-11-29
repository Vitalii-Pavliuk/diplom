'use client';

import React from 'react';
import { Sparkles, Trash2, Shuffle, Download, Upload } from 'lucide-react';

interface ControlsProps {
  onSolve: () => void;
  onClear: () => void;
  onRandom: () => void;
  onExport?: () => void;
  onImport?: () => void;
  isLoading?: boolean;
  currentModel?: string;
  onModelChange?: (model: string) => void;
  availableModels?: string[];
  // Нові пропси для складності
  difficulty: string;
  onDifficultyChange: (diff: string) => void;
}

export default function Controls({
  onSolve,
  onClear,
  onRandom,
  onExport,
  onImport,
  isLoading = false,
  currentModel = 'baseline',
  onModelChange,
  availableModels = ['baseline', 'advanced', 'gnn'],
  difficulty,
  onDifficultyChange,
}: ControlsProps) {
  return (
    <div className="bg-white rounded-lg shadow-lg p-6 space-y-6">
      <h2 className="text-xl font-bold text-gray-800 border-b pb-2">Control Panel</h2>
      
      {/* 1. Вибір Моделі */}
      {onModelChange && (
        <div className="space-y-2">
          <label className="block text-sm font-bold text-gray-700">
            Neural Architecture
          </label>
          <select
            value={currentModel}
            onChange={(e) => onModelChange(e.target.value)}
            disabled={isLoading}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 bg-gray-50"
          >
            {availableModels.map((model) => (
              <option key={model} value={model}>
                {model === 'baseline' && '⚡ CNN Baseline (Fast)'}
                {model === 'advanced' && '🧠 CNN Advanced (ResNet)'}
                {model === 'gnn' && '🕸️ GNN (Graph Network)'}
              </option>
            ))}
          </select>
        </div>
      )}

      {/* 2. Генерація Пазлів (Оновлено) */}
      <div className="space-y-2 pt-2">
        <label className="block text-sm font-bold text-gray-700">
          Puzzle Generation
        </label>
        <div className="flex gap-2">
          <select
            value={difficulty}
            onChange={(e) => onDifficultyChange(e.target.value)}
            disabled={isLoading}
            className="flex-1 px-3 py-2 border border-gray-300 rounded-lg text-sm bg-gray-50"
          >
            <option value="easy">Easy</option>
            <option value="medium">Medium</option>
            <option value="hard">Hard</option>
            <option value="impossible">🔥 Impossible</option>
          </select>
          
          <button
            onClick={onRandom}
            disabled={isLoading}
            className="flex-1 flex items-center justify-center gap-2 px-3 py-2 bg-green-600 text-white font-semibold rounded-lg hover:bg-green-700 transition-colors disabled:opacity-50"
          >
            <Shuffle size={18} /> Generate
          </button>
        </div>
      </div>
      
      {/* 3. Головні дії */}
      <div className="space-y-3 pt-2">
        <button
          onClick={onSolve}
          disabled={isLoading}
          className="w-full flex items-center justify-center gap-2 px-6 py-4 bg-blue-600 text-white font-bold text-lg rounded-xl hover:bg-blue-700 shadow-md hover:shadow-lg transition-all disabled:opacity-50"
        >
          <Sparkles size={24} />
          {isLoading ? 'Solving...' : 'SOLVE PUZZLE'}
        </button>
        
        <button
          onClick={onClear}
          disabled={isLoading}
          className="w-full flex items-center justify-center gap-2 px-6 py-2 bg-gray-100 text-gray-600 font-semibold rounded-lg hover:bg-gray-200 transition-colors"
        >
          <Trash2 size={18} />
          Clear Board
        </button>
      </div>
      
      {/* Import/Export (Small buttons) */}
      {(onExport || onImport) && (
        <div className="pt-4 border-t border-gray-200 grid grid-cols-2 gap-2">
            {onImport && (
              <button onClick={onImport} className="flex justify-center items-center gap-2 px-3 py-2 text-xs text-gray-500 border rounded hover:bg-gray-50">
                <Upload size={14} /> Import JSON
              </button>
            )}
            {onExport && (
              <button onClick={onExport} className="flex justify-center items-center gap-2 px-3 py-2 text-xs text-gray-500 border rounded hover:bg-gray-50">
                <Download size={14} /> Export JSON
              </button>
            )}
        </div>
      )}
    </div>
  );
}