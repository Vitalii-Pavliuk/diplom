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
  difficulty: string;
  onDifficultyChange: (diff: string) => void;
  // НОВІ ПРОПСИ
  confidence: number | null;
  realAccuracy: number | null;
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
  // Деструктуризація
  confidence,
  realAccuracy,
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

      {/* 2. Генерація Пазлів */}
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
            <option value="easypeasy">Easy-peasy</option>
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

      {/* --- 4. ВІЗУАЛІЗАЦІЯ МЕТРИК (НОВЕ) --- */}
      {(confidence !== null || realAccuracy !== null) && (
        <div className="pt-4 border-t border-gray-100 space-y-4">
            
            {/* Графік Впевненості */}
            {confidence !== null && (
            <div className="bg-gray-50 rounded-xl p-3 border border-gray-200">
                <div className="flex justify-between items-end mb-1">
                    <h3 className="text-gray-500 font-bold uppercase text-[10px] tracking-wider">Model Confidence</h3>
                    <span className="text-sm font-bold text-gray-700">{(confidence * 100).toFixed(1)}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2 overflow-hidden">
                <div 
                    className={`h-full transition-all duration-500 ${confidence > 0.8 ? 'bg-yellow-400' : 'bg-yellow-600'}`}
                    style={{ width: `${confidence * 100}%` }}
                />
                </div>
            </div>
            )}

            {/* Графік Реальної Точності */}
            {realAccuracy !== null && (
            <div className="bg-blue-50 rounded-xl p-3 border border-blue-200">
                <div className="flex justify-between items-end mb-1">
                    <h3 className="text-blue-600 font-bold uppercase text-[10px] tracking-wider">Real Accuracy</h3>
                    <span className={`text-sm font-black ${realAccuracy === 1 ? 'text-green-600' : 'text-blue-900'}`}>
                        {(realAccuracy * 100).toFixed(1)}%
                    </span>
                </div>
                <div className="w-full bg-blue-200 rounded-full h-2 overflow-hidden">
                <div 
                    className={`h-full transition-all duration-500 ${
                        realAccuracy === 1 ? 'bg-green-500' : realAccuracy > 0.9 ? 'bg-blue-500' : 'bg-red-500'
                    }`}
                    style={{ width: `${realAccuracy * 100}%` }}
                />
                </div>
                {realAccuracy < 1 && (
                    <p className="text-[10px] text-red-500 mt-1 font-medium text-center">
                        ⚠️ Mistakes found vs Ground Truth
                    </p>
                )}
            </div>
            )}
        </div>
      )}
      
      {/* 5. Import/Export */}
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