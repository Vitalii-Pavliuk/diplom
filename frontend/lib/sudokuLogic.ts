// frontend/lib/sudokuLogic.ts

export function isValid(board: number[][], row: number, col: number, num: number): boolean {
    for (let x = 0; x < 9; x++) {
        if (board[row][x] === num && x !== col) return false;
        if (board[x][col] === num && x !== row) return false;
        
        const boxRow = 3 * Math.floor(row / 3) + Math.floor(x / 3);
        const boxCol = 3 * Math.floor(col / 3) + x % 3;
        if (board[boxRow][boxCol] === num && (boxRow !== row || boxCol !== col)) return false;
    }
    return true;
}

// Класичний алгоритм Backtracking для пошуку ідеального рішення
export function solveSudokuClassic(board: number[][]): number[][] | null {
    const solvedBoard = board.map(row => [...row]);
    
    function solve(): boolean {
        for (let row = 0; row < 9; row++) {
            for (let col = 0; col < 9; col++) {
                if (solvedBoard[row][col] === 0) {
                    for (let num = 1; num <= 9; num++) {
                        if (isValid(solvedBoard, row, col, num)) {
                            solvedBoard[row][col] = num;
                            if (solve()) return true;
                            solvedBoard[row][col] = 0;
                        }
                    }
                    return false;
                }
            }
        }
        return true;
    }

    if (solve()) return solvedBoard;
    return null;
}

// Функція порівняння відповіді АІ з ідеалом
export function calculateAccuracy(aiBoard: number[][], trueBoard: number[][]): number {
    let correct = 0;
    let total = 0;
    
    for (let r = 0; r < 9; r++) {
        for (let c = 0; c < 9; c++) {
            // Рахуємо всі клітинки (або можна рахувати тільки пусті, якщо хочеш)
            total++; 
            if (aiBoard[r][c] === trueBoard[r][c]) {
                correct++;
            }
        }
    }
    
    return (correct / total);
}