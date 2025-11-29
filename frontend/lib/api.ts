/**
 * API Client for Sudoku Solver Backend
 */

import axios from 'axios';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export type SudokuBoard = number[][];

export interface SolveSudokuResponse {
  solution: SudokuBoard;
  model_used: string;
  confidence?: number;
}

export interface ModelInfo {
  current_model: string;
  available_models: string[];
  model_loaded: boolean;
}

/**
 * Solve a Sudoku puzzle
 */
export async function solveSudoku(board: SudokuBoard): Promise<SolveSudokuResponse> {
  try {
    const response = await apiClient.post<SolveSudokuResponse>('/solve', {
      board,
    });
    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      throw new Error(error.response?.data?.detail || 'Failed to solve Sudoku');
    }
    throw error;
  }
}

/**
 * Get current model information
 */
export async function getModelInfo(): Promise<ModelInfo> {
  try {
    const response = await apiClient.get<ModelInfo>('/model');
    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      throw new Error(error.response?.data?.detail || 'Failed to get model info');
    }
    throw error;
  }
}

/**
 * Switch to a different model
 */
export async function switchModel(modelName: string): Promise<void> {
  try {
    await apiClient.post('/model/switch', null, {
      params: { model_name: modelName },
    });
  } catch (error) {
    if (axios.isAxiosError(error)) {
      throw new Error(error.response?.data?.detail || 'Failed to switch model');
    }
    throw error;
  }
}

/**
 * Check API health
 */
export async function checkHealth(): Promise<boolean> {
  try {
    const response = await apiClient.get('/health');
    return response.data.status === 'healthy';
  } catch (error) {
    return false;
  }
}

