"""
Скрипт для порівняння board accuracy різних моделей
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Завантаження даних
weights_dir = Path("weights")

# Завантаження історій
with open(weights_dir / "baseline_history.json", "r") as f:
    baseline_history = json.load(f)

with open(weights_dir / "advanced_history.json", "r") as f:
    advanced_history = json.load(f)

# Підготовка даних
models = {
    "CNN Baseline": baseline_history,
    "CNN Advanced": advanced_history,
}

# Створення графіка
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

# Кольори для моделей
colors = {
    "CNN Baseline": "#FF6B6B",
    "CNN Advanced": "#4ECDC4",
}

# === ГРАФІК 1: Training Board Accuracy ===
for model_name, history in models.items():
    epochs = [h["epoch"] for h in history]
    train_board_acc = [h["train"]["board_accuracy"] * 100 for h in history]  # Конвертація в %
    
    ax1.plot(
        epochs, 
        train_board_acc, 
        label=model_name, 
        marker='o', 
        markersize=4,
        linewidth=2,
        color=colors[model_name],
        alpha=0.8
    )

ax1.set_xlabel("Epoch", fontsize=12, fontweight='bold')
ax1.set_ylabel("Board Accuracy (%)", fontsize=12, fontweight='bold')
ax1.set_title("Training Board Accuracy", fontsize=14, fontweight='bold')
ax1.legend(fontsize=11, loc='lower right')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_ylim(bottom=0)

# === ГРАФІК 2: Validation Board Accuracy ===
for model_name, history in models.items():
    epochs = [h["epoch"] for h in history]
    val_board_acc = [h["val"]["board_accuracy"] * 100 for h in history]  # Конвертація в %
    
    ax2.plot(
        epochs, 
        val_board_acc, 
        label=model_name, 
        marker='s', 
        markersize=4,
        linewidth=2,
        color=colors[model_name],
        alpha=0.8
    )

ax2.set_xlabel("Epoch", fontsize=12, fontweight='bold')
ax2.set_ylabel("Board Accuracy (%)", fontsize=12, fontweight='bold')
ax2.set_title("Validation Board Accuracy", fontsize=14, fontweight='bold')
ax2.legend(fontsize=11, loc='lower right')
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_ylim(bottom=0)

plt.tight_layout()
plt.savefig(weights_dir / "models_board_accuracy_comparison.png", dpi=300, bbox_inches='tight')
print(f"[OK] Chart saved: {weights_dir / 'models_board_accuracy_comparison.png'}")

# === Additional chart: All 4 lines on one plot ===
fig2, ax = plt.subplots(figsize=(14, 8))

for model_name, history in models.items():
    epochs = [h["epoch"] for h in history]
    train_board_acc = [h["train"]["board_accuracy"] * 100 for h in history]
    val_board_acc = [h["val"]["board_accuracy"] * 100 for h in history]
    
    # Train (суцільна лінія)
    ax.plot(
        epochs, 
        train_board_acc, 
        label=f"{model_name} (Train)", 
        marker='o', 
        markersize=3,
        linewidth=2.5,
        linestyle='-',
        color=colors[model_name],
        alpha=0.9
    )
    
    # Validation (пунктирна лінія)
    ax.plot(
        epochs, 
        val_board_acc, 
        label=f"{model_name} (Val)", 
        marker='s', 
        markersize=3,
        linewidth=2.5,
        linestyle='--',
        color=colors[model_name],
        alpha=0.7
    )

ax.set_xlabel("Epoch", fontsize=13, fontweight='bold')
ax.set_ylabel("Board Accuracy (%)", fontsize=13, fontweight='bold')
ax.set_title("Models Comparison: Training vs Validation Board Accuracy", fontsize=15, fontweight='bold')
ax.legend(fontsize=11, loc='lower right', framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_ylim(bottom=0)

# Додавання текстових анотацій з фінальними результатами
y_offset = 2
for model_name, history in models.items():
    final_train = history[-1]["train"]["board_accuracy"] * 100
    final_val = history[-1]["val"]["board_accuracy"] * 100
    final_epoch = history[-1]["epoch"]
    
    # Анотація для validation (більш важлива)
    ax.annotate(
        f"{final_val:.2f}%",
        xy=(final_epoch, final_val),
        xytext=(5, y_offset),
        textcoords='offset points',
        fontsize=9,
        fontweight='bold',
        color=colors[model_name],
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=colors[model_name], alpha=0.8)
    )
    y_offset += 3

plt.tight_layout()
plt.savefig(weights_dir / "models_board_accuracy_combined.png", dpi=300, bbox_inches='tight')
print(f"[OK] Chart saved: {weights_dir / 'models_board_accuracy_combined.png'}")

# === СТАТИСТИКА ===
print("\n" + "="*70)
print("PORIVNYANNYA MODELEY - BOARD ACCURACY")
print("="*70)

for model_name, history in models.items():
    train_board_acc = [h["train"]["board_accuracy"] for h in history]
    val_board_acc = [h["val"]["board_accuracy"] for h in history]
    
    print(f"\n{model_name}")
    print(f"   Epochs trained:              {len(history)}")
    print(f"   Best Train Board Accuracy:   {max(train_board_acc)*100:.2f}%")
    print(f"   Best Val Board Accuracy:     {max(val_board_acc)*100:.2f}%")
    print(f"   Final Train Board Accuracy:  {train_board_acc[-1]*100:.2f}%")
    print(f"   Final Val Board Accuracy:    {val_board_acc[-1]*100:.2f}%")

print("\n" + "="*70)

# Порівняння
baseline_best_val = max([h["val"]["board_accuracy"] for h in baseline_history])
advanced_best_val = max([h["val"]["board_accuracy"] for h in advanced_history])

improvement = (advanced_best_val - baseline_best_val) / baseline_best_val * 100

print(f"\nVYSNOVOK:")
print(f"   CNN Advanced pokazuye na {improvement:.1f}% krashchu board accuracy nizh Baseline")
print(f"   ({advanced_best_val*100:.2f}% vs {baseline_best_val*100:.2f}%)")
print("="*70)

# plt.show()  # Закоментовано, щоб не блокувати виконання
