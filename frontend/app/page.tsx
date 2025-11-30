'use client';

import React, { useState, useEffect } from 'react';
import SudokuBoard, { CellState } from '@/components/SudokuBoard';
import Controls from '@/components/Controls';
import { solveSudoku, getModelInfo, switchModel, checkHealth } from '@/lib/api';
// ! ВАЖЛИВО: Переконайся, що ти створив файл lib/sudokuLogic.ts з кодом, який я давав раніше
import { solveSudokuClassic, calculateAccuracy } from '@/lib/sudokuLogic';
import { AlertCircle, CheckCircle2, Loader2, BrainCircuit } from 'lucide-react';

// --- БАЗА ДАНИХ ПАЗЛІВ ---
const PUZZLE_DB: Record<string, string[]> = {
  easypeasy: [
    '435260781680571493197830562826195047374082915951743620509326874248957130763410259',
  ],
  easy: [
    '000000000000003085001020000000507000004000100090000000500000073002010000000040009',
    '530070000600195000098000060800060003400803001700020006060000280000419005000080079',
    '000000200080007090602000500070060000000901000000020040005000602020000070009000000',
  ],
  medium: [
    '003020600900305001001806400008102900700000008006708200002609500800203009005010300',
    '020608000580009700000040000370000500600000004008000013000020000009800036000306090',
  ],
  hard: [
    '200080300060070084030500209000105408000000000402706000301007040720040060004010003',
    '000000000000003085001020000000507000004000100090000000500000073002010000000040009',
  ],
  impossible: [
    '800000000003600000070090200050007000000045700000100030001000068008500010090000400',
    '000000000000003085001020000000507000004000100090000000500000073002010000000040009',
  ]
};

export default function Home() {
  // Стан дошки
  const [board, setBoard] = useState<CellState[][]>(() => createEmptyBoard());
  
  // --- НОВІ СТАНИ ДЛЯ ТОЧНОСТІ ---
  const [groundTruth, setGroundTruth] = useState<number[][] | null>(null);
  const [realAccuracy, setRealAccuracy] = useState<number | null>(null);
  
  // UI стани
  const [isLoading, setIsLoading] = useState(false);
  const [message, setMessage] = useState<{ type: 'success' | 'error' | 'info'; text: string } | null>(null);
  
  // API та Модель
  const [currentModel, setCurrentModel] = useState('baseline');
  const [availableModels, setAvailableModels] = useState<string[]>(['baseline', 'advanced', 'gnn']);
  const [apiHealthy, setApiHealthy] = useState(false);
  const [confidence, setConfidence] = useState<number | null>(null);

  // Складність
  const [difficulty, setDifficulty] = useState('easy');
  
  // Ініціалізація при завантаженні
  useEffect(() => {
    const initialize = async () => {
      try {
        const healthy = await checkHealth();
        setApiHealthy(healthy);
        
        if (healthy) {
          const info = await getModelInfo();
          setCurrentModel(info.current_model);
          setAvailableModels(info.available_models);
          setMessage({ type: 'success', text: 'Connected to API' });
          // Завантажуємо простий пазл при старті
          loadRandomPuzzle('easy');
        } else {
           setMessage({ type: 'error', text: 'Backend is offline' });
        }
      } catch (error) {
        setApiHealthy(false);
        setMessage({ type: 'error', text: 'Failed to connect to API' });
      }
    };
    initialize();
  }, []);
  
  // --- Допоміжні функції ---

  function createEmptyBoard(): CellState[][] {
    return Array(9).fill(null).map(() =>
      Array(9).fill(null).map(() => ({ 
        value: 0, 
        isUserInput: true, 
        isError: false 
      }))
    );
  }
  
  const boardToNumberArray = (b: CellState[][]): number[][] => {
    return b.map(row => row.map(cell => cell.value));
  };

  // --- Обробники подій ---
  
  const handleCellChange = (row: number, col: number, value: string) => {
    const newBoard = board.map((r, rIdx) =>
      r.map((cell, cIdx) => {
        if (rIdx === row && cIdx === col) {
          return { 
            ...cell, 
            value: value === '' ? 0 : parseInt(value), 
            isUserInput: true 
          };
        }
        return cell;
      })
    );
    setBoard(newBoard);
    // Скидаємо метрики при ручній зміні
    setConfidence(null);
    setRealAccuracy(null);
  };
  
  const handleSolve = async () => {
    if (!apiHealthy) {
      setMessage({ type: 'error', text: 'API is not available. Please start backend.' });
      return;
    }
    
    setIsLoading(true);
    setMessage({ type: 'info', text: 'Thinking...' });
    
    try {
      const numberBoard = boardToNumberArray(board);
      const response = await solveSudoku(numberBoard);
      
      const newBoard = board.map((row, rIdx) =>
        row.map((cell, cIdx) => ({
          value: response.solution[rIdx][cIdx],
          isUserInput: cell.value !== 0 && cell.value === response.solution[rIdx][cIdx],
          isError: false,
        }))
      );
      
      setBoard(newBoard);
      setConfidence(response.confidence || null);
      
      // --- РАХУЄМО РЕАЛЬНУ ТОЧНІСТЬ ---
      let truth = groundTruth;
      // Якщо раптом groundTruth немає (наприклад, ручне введення), спробуємо знайти зараз
      if (!truth) {
         truth = solveSudokuClassic(numberBoard);
      }
      
      if (truth) {
          const acc = calculateAccuracy(response.solution, truth);
          setRealAccuracy(acc);
      } else {
          setRealAccuracy(null); // Не вдалося знайти еталонне рішення
      }

      setMessage({
        type: 'success',
        text: `Solved using ${response.model_used} model`,
      });
    } catch (error) {
      setMessage({ 
        type: 'error', 
        text: error instanceof Error ? error.message : 'Failed to solve' 
      });
    } finally {
      setIsLoading(false);
    }
  };
  
  const handleClear = () => {
    setBoard(createEmptyBoard());
    setConfidence(null);
    setRealAccuracy(null);
    setGroundTruth(null);
    setMessage(null);
  };
  
  // Логіка вибору випадкового пазла
  const loadRandomPuzzle = (diff: string) => {
    const puzzles = PUZZLE_DB[diff] || PUZZLE_DB['easy'];
    const randomString = puzzles[Math.floor(Math.random() * puzzles.length)];
    
    // Формуємо числовий масив для пошуку рішення
    const newBoardNumbers = Array(9).fill(null).map((_, row) =>
        Array(9).fill(0).map((_, col) => parseInt(randomString[row * 9 + col]))
    );

    const newBoard = newBoardNumbers.map(row => 
        row.map(val => ({ 
            value: val, 
            isUserInput: val !== 0, 
            isError: false 
        }))
    );
    
    setBoard(newBoard);
    
    // --- ЗНАХОДИМО ІДЕАЛЬНЕ РІШЕННЯ ОДРАЗУ ---
    const truth = solveSudokuClassic(newBoardNumbers);
    setGroundTruth(truth);

    setConfidence(null);
    setRealAccuracy(null);
    setMessage({ type: 'info', text: `Loaded ${diff.toUpperCase()} puzzle` });
  };

  const handleRandom = () => {
    loadRandomPuzzle(difficulty);
  };
  
  const handleModelChange = async (modelName: string) => {
    if (!apiHealthy) return;
    setIsLoading(true);
    try {
      await switchModel(modelName);
      setCurrentModel(modelName);
      setMessage({ type: 'success', text: `Switched to ${modelName}` });
    } catch (error) {
      setMessage({ type: 'error', text: 'Failed to switch model' });
    } finally {
      setIsLoading(false);
    }
  };

  // --- ЕКСПОРТ ---
  const handleExport = () => {
    const numberBoard = boardToNumberArray(board);
    const json = JSON.stringify(numberBoard, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'sudoku-board.json';
    a.click();
    URL.revokeObjectURL(url);
    setMessage({ type: 'success', text: 'Board exported' });
  };

  // --- ІМПОРТ ---
  const handleImport = () => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'application/json';
    input.onchange = (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
          try {
            const json = JSON.parse(e.target?.result as string);
            if (Array.isArray(json) && json.length === 9 && json.every((row: any) => Array.isArray(row) && row.length === 9)) {
              const newBoard = json.map((row: number[]) =>
                row.map((value: number) => ({
                  value: typeof value === 'number' ? value : 0,
                  isUserInput: value !== 0,
                  isError: false,
                }))
              );
              setBoard(newBoard);
              // Спробуємо розв'язати імпортований пазл для Ground Truth
              const numBoard = json as number[][];
              const truth = solveSudokuClassic(numBoard);
              setGroundTruth(truth);
              setConfidence(null);
              setRealAccuracy(null);
              
              setMessage({ type: 'success', text: 'Board imported' });
            } else {
              setMessage({ type: 'error', text: 'Invalid board format' });
            }
          } catch (error) {
            setMessage({ type: 'error', text: 'Failed to parse JSON' });
          }
        };
        reader.readAsText(file);
      }
    };
    input.click();
  };
  
  return (
    <main className="min-h-screen p-4 sm:p-8 flex flex-col items-center">
      <div className="max-w-7xl w-full mx-auto">
        
        {/* Хедер */}
        <header className="text-center mb-10">
            <div className="flex justify-center items-center gap-3 mb-2">
                <BrainCircuit size={40} className="text-blue-600" />
                <h1 className="text-4xl font-extrabold text-gray-900 tracking-tight">
                    NeuroSudoku
                </h1>
            </div>
            <p className="text-lg text-gray-600 max-w-2xl mx-auto">
                Thesis Project: Comparative Analysis of CNN vs GNN Architectures
            </p>
          
            <div className={`mt-4 inline-flex items-center gap-2 px-4 py-1.5 rounded-full text-sm font-medium ${
                apiHealthy ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'
            }`}>
                {apiHealthy ? <CheckCircle2 size={16} /> : <AlertCircle size={16} />}
                {apiHealthy ? 'Backend Connected' : 'Backend Disconnected'}
            </div>
        </header>
        
        {/* Повідомлення */}
        {message && (
          <div className={`max-w-2xl mx-auto mb-6 p-4 rounded-lg flex items-center gap-3 shadow-sm ${
            message.type === 'success' ? 'bg-green-50 text-green-800 border border-green-200' :
            message.type === 'error' ? 'bg-red-50 text-red-800 border border-red-200' :
            'bg-blue-50 text-blue-800 border border-blue-200'
          }`}>
            {message.type === 'info' && <Loader2 size={20} className="animate-spin" />}
            <span className="font-medium">{message.text}</span>
          </div>
        )}
        
        {/* Основна сітка */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 lg:gap-12 items-start">
          
          {/* Дошка Судоку */}
          <div className="lg:col-span-2 flex justify-center order-2 lg:order-1">
            <SudokuBoard
              board={board}
              onCellChange={handleCellChange}
              isLoading={isLoading}
            />
          </div>
          
          {/* Панель керування */}
          <div className="lg:col-span-1 order-1 lg:order-2">
            <Controls
              onSolve={handleSolve}
              onClear={handleClear}
              onRandom={handleRandom}
              onExport={handleExport}
              onImport={handleImport}
              isLoading={isLoading}
              currentModel={currentModel}
              onModelChange={handleModelChange}
              availableModels={availableModels}
              difficulty={difficulty}
              onDifficultyChange={setDifficulty}
              // Передаємо нові пропси
              confidence={confidence}
              realAccuracy={realAccuracy}
            />
          </div>
        </div>
      </div>
    </main>
  );
}