"""
Повне порівняння всіх метрик моделей
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Завантаження даних
weights_dir = Path("weights")

with open(weights_dir / "baseline_history.json", "r") as f:
    baseline_history = json.load(f)

with open(weights_dir / "advanced_history.json", "r") as f:
    advanced_history = json.load(f)

models = {
    "CNN Baseline": baseline_history,
    "CNN Advanced": advanced_history,
}

colors = {
    "CNN Baseline": "#FF6B6B",
    "CNN Advanced": "#4ECDC4",
}

# === КОМПЛЕКСНЕ ПОРІВНЯННЯ: 2x2 ГРАФІКИ ===
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# === 1. Training Cell Accuracy ===
for model_name, history in models.items():
    epochs = [h["epoch"] for h in history]
    train_cell_acc = [h["train"]["cell_accuracy"] * 100 for h in history]
    
    axes[0, 0].plot(
        epochs, 
        train_cell_acc, 
        label=model_name, 
        marker='o', 
        markersize=3,
        linewidth=2,
        color=colors[model_name],
        alpha=0.8
    )

axes[0, 0].set_xlabel("Epoch", fontsize=11, fontweight='bold')
axes[0, 0].set_ylabel("Cell Accuracy (%)", fontsize=11, fontweight='bold')
axes[0, 0].set_title("Training Cell Accuracy", fontsize=13, fontweight='bold')
axes[0, 0].legend(fontsize=10, loc='lower right')
axes[0, 0].grid(True, alpha=0.3, linestyle='--')

# === 2. Validation Cell Accuracy ===
for model_name, history in models.items():
    epochs = [h["epoch"] for h in history]
    val_cell_acc = [h["val"]["cell_accuracy"] * 100 for h in history]
    
    axes[0, 1].plot(
        epochs, 
        val_cell_acc, 
        label=model_name, 
        marker='s', 
        markersize=3,
        linewidth=2,
        color=colors[model_name],
        alpha=0.8
    )

axes[0, 1].set_xlabel("Epoch", fontsize=11, fontweight='bold')
axes[0, 1].set_ylabel("Cell Accuracy (%)", fontsize=11, fontweight='bold')
axes[0, 1].set_title("Validation Cell Accuracy", fontsize=13, fontweight='bold')
axes[0, 1].legend(fontsize=10, loc='lower right')
axes[0, 1].grid(True, alpha=0.3, linestyle='--')

# === 3. Training Board Accuracy ===
for model_name, history in models.items():
    epochs = [h["epoch"] for h in history]
    train_board_acc = [h["train"]["board_accuracy"] * 100 for h in history]
    
    axes[1, 0].plot(
        epochs, 
        train_board_acc, 
        label=model_name, 
        marker='o', 
        markersize=3,
        linewidth=2,
        color=colors[model_name],
        alpha=0.8
    )

axes[1, 0].set_xlabel("Epoch", fontsize=11, fontweight='bold')
axes[1, 0].set_ylabel("Board Accuracy (%)", fontsize=11, fontweight='bold')
axes[1, 0].set_title("Training Board Accuracy", fontsize=13, fontweight='bold')
axes[1, 0].legend(fontsize=10, loc='lower right')
axes[1, 0].grid(True, alpha=0.3, linestyle='--')
axes[1, 0].set_ylim(bottom=0)

# === 4. Validation Board Accuracy ===
for model_name, history in models.items():
    epochs = [h["epoch"] for h in history]
    val_board_acc = [h["val"]["board_accuracy"] * 100 for h in history]
    
    axes[1, 1].plot(
        epochs, 
        val_board_acc, 
        label=model_name, 
        marker='s', 
        markersize=3,
        linewidth=2,
        color=colors[model_name],
        alpha=0.8
    )

axes[1, 1].set_xlabel("Epoch", fontsize=11, fontweight='bold')
axes[1, 1].set_ylabel("Board Accuracy (%)", fontsize=11, fontweight='bold')
axes[1, 1].set_title("Validation Board Accuracy", fontsize=13, fontweight='bold')
axes[1, 1].legend(fontsize=10, loc='lower right')
axes[1, 1].grid(True, alpha=0.3, linestyle='--')
axes[1, 1].set_ylim(bottom=0)

plt.suptitle("CNN Models Comparison: All Metrics", fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(weights_dir / "models_all_metrics_comparison.png", dpi=300, bbox_inches='tight')
print(f"[OK] Chart saved: {weights_dir / 'models_all_metrics_comparison.png'}")

# === ДЕТАЛЬНА СТАТИСТИКА ===
print("\n" + "="*80)
print("FULL METRICS COMPARISON")
print("="*80)

for model_name, history in models.items():
    train_cell_acc = [h["train"]["cell_accuracy"] for h in history]
    val_cell_acc = [h["val"]["cell_accuracy"] for h in history]
    train_board_acc = [h["train"]["board_accuracy"] for h in history]
    val_board_acc = [h["val"]["board_accuracy"] for h in history]
    
    print(f"\n{model_name}")
    print(f"  Epochs:                         {len(history)}")
    print(f"\n  CELL ACCURACY:")
    print(f"    Best Train:                   {max(train_cell_acc)*100:.2f}%")
    print(f"    Best Val:                     {max(val_cell_acc)*100:.2f}%")
    print(f"    Final Train:                  {train_cell_acc[-1]*100:.2f}%")
    print(f"    Final Val:                    {val_cell_acc[-1]*100:.2f}%")
    print(f"\n  BOARD ACCURACY:")
    print(f"    Best Train:                   {max(train_board_acc)*100:.2f}%")
    print(f"    Best Val:                     {max(val_board_acc)*100:.2f}%")
    print(f"    Final Train:                  {train_board_acc[-1]*100:.2f}%")
    print(f"    Final Val:                    {val_board_acc[-1]*100:.2f}%")

print("\n" + "="*80)

# Абсолютна різниця
baseline_val_cell = max([h["val"]["cell_accuracy"] for h in baseline_history])
advanced_val_cell = max([h["val"]["cell_accuracy"] for h in advanced_history])
baseline_val_board = max([h["val"]["board_accuracy"] for h in baseline_history])
advanced_val_board = max([h["val"]["board_accuracy"] for h in advanced_history])

cell_improvement = (advanced_val_cell - baseline_val_cell) / baseline_val_cell * 100
board_improvement_abs = (advanced_val_board - baseline_val_board) * 100  # Абсолютна різниця

print(f"\nIMPROVEMENTS (CNN Advanced vs Baseline):")
print(f"  Cell Accuracy:   {advanced_val_cell*100:.2f}% vs {baseline_val_cell*100:.2f}%")
print(f"                   (+{cell_improvement:.1f}% relative improvement)")
print(f"\n  Board Accuracy:  {advanced_val_board*100:.2f}% vs {baseline_val_board*100:.2f}%")
print(f"                   (+{board_improvement_abs:.2f} percentage points)")
print("\n" + "="*80)

print("\nKEY INSIGHTS:")
print("  - CNN Baseline has ~77% cell accuracy, but only ~0.1% board accuracy")
print("  - CNN Advanced has ~96% cell accuracy, which gives ~52% board accuracy")
print("  - For fully correct sudoku, very high cell accuracy is needed!")
print("  - Difference between 77% and 96% cell accuracy = 520x difference in board accuracy")
print("="*80)
