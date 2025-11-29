# Sudoku Dataset Directory

This directory should contain the Sudoku dataset for training the neural networks.

## Required File

**File name**: `sudoku.csv`

**Format**: CSV with two columns
- `quizzes`: String of 81 digits (0 for empty cells)
- `solutions`: String of 81 digits (complete solution)

## Where to Get the Dataset

### Option 1: Kaggle (Recommended)
Search for "1 million sudoku games" on Kaggle

### Option 2: Other Sources
Any Sudoku dataset in the same format will work

## Example Format

```csv
quizzes,solutions
004300209005009001070060043006002087190007400050083000600000105003508690042910300,864371259325849761971265843436192587198657432257483916689734125713528694542916378
020608000580009700000040000370000500600000004008000013000020000009800036000306090,123678945584239761796145823371964528652387194498512613834521279219857436567396482
```

## Dataset Statistics

Expected dataset properties:
- **Size**: ~1 million puzzles
- **File size**: ~150-200 MB
- **Difficulty**: Various (easy to hard)
- **Format**: Each puzzle and solution as 81-character string

## Verifying Your Dataset

After downloading, verify the format:

```bash
python -c "
import pandas as pd
df = pd.read_csv('data/sudoku.csv')
print(f'Rows: {len(df)}')
print(f'Columns: {list(df.columns)}')
print(f'Sample quiz length: {len(df.iloc[0][\"quizzes\"])}')
print(f'Sample solution length: {len(df.iloc[0][\"solutions\"])}')
"
```

Expected output:
```
Rows: 1000000 (or similar)
Columns: ['quizzes', 'solutions']
Sample quiz length: 81
Sample solution length: 81
```

## Alternative: Small Test Dataset

For testing purposes, you can create a small test dataset:

```python
import pandas as pd

# Create a small test dataset (10 puzzles)
quizzes = [
    "530070000600195000098000060800060003400803001700020006060000280000419005000080079",
    "003020600900305001001806400008102900700000008006708200002609500800203009005010300",
    # Add more...
]

solutions = [
    "534678912672195348198342567859761423426853791713924856961537284287419635345286179",
    "483921657967345821251876493548132976729564138136798245372689514814253769695417382",
    # Add more...
]

df = pd.DataFrame({'quizzes': quizzes, 'solutions': solutions})
df.to_csv('data/sudoku.csv', index=False)
print("Test dataset created!")
```

## Important Notes

1. **Do NOT commit large datasets to git** (they're in .gitignore)
2. The CSV file should have NO index column
3. Each string must be exactly 81 characters
4. Digits 0-9 only (0 = empty cell)
5. UTF-8 encoding recommended

## Troubleshooting

### "File not found" error
- Make sure the file is named exactly `sudoku.csv`
- Check it's in the `backend/data/` directory
- Use absolute path if needed

### "Invalid format" error
- Verify columns are named `quizzes` and `solutions`
- Check that strings are exactly 81 characters
- Ensure no extra columns or index

### "Out of memory" error
- The full dataset (~1M rows) requires ~4-8GB RAM
- Use a smaller subset for testing
- Increase system swap space if needed

## Using a Subset

To train on a smaller subset (faster for testing):

```python
# In train.py or dataset.py
df = pd.read_csv('data/sudoku.csv', nrows=10000)  # Only first 10K rows
```

Or use the `--train-split` parameter to adjust the training size.

---

Once you have the dataset in place, you're ready to train models!

See `train.py --help` for training options.

