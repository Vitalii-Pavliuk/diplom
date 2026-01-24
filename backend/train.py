"""
Training Script for Sudoku Neural Networks
Supports all three model architectures: Baseline CNN, Advanced CNN, and GNN
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import os
import json
from datetime import datetime

# Переконайся, що dataset.py та models/ існують
from dataset import SudokuDataset
from models import CNNBaseline, CNNAdvanced, GNNModel, SudokuRNN


def calculate_accuracy(predictions: torch.Tensor, targets: torch.Tensor, inputs: torch.Tensor) -> dict:
    """
    Calculate accuracy metrics
    """
    # Overall cell accuracy
    correct = (predictions == targets).float()
    cell_accuracy = correct.mean().item()
    
    # Accuracy only on initially empty cells (0s in input)
    # Reshape inputs if needed (for RNN which uses flattened input)
    if inputs.dim() == 2 and inputs.size(1) == 81:
        # RNN flattened input: reshape to (Batch, 9, 9)
        inputs_reshaped = inputs.view(inputs.size(0), 9, 9)
    else:
        inputs_reshaped = inputs
    
    empty_mask = (inputs_reshaped == 0)
    if empty_mask.sum() > 0:
        empty_cell_accuracy = correct[empty_mask].mean().item()
    else:
        empty_cell_accuracy = 0.0
    
    # Board accuracy (entire puzzle solved correctly)
    board_correct = correct.view(predictions.size(0), -1).all(dim=1)
    board_accuracy = board_correct.float().mean().item()
    
    return {
        'cell_accuracy': cell_accuracy,
        'empty_cell_accuracy': empty_cell_accuracy,
        'board_accuracy': board_accuracy
    }


def train_epoch(model: nn.Module, 
                dataloader: DataLoader, 
                criterion: nn.Module, 
                optimizer: optim.Optimizer,
                device: str,
                model_name: str = None) -> dict:
    """
    Train for one epoch
    """
    model.train()
    
    total_loss = 0
    total_cell_acc = 0
    total_empty_cell_acc = 0
    total_board_acc = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc="Training")
    
    for inputs, targets in pbar:
        # inputs: (Batch, 9, 9)
        # targets: (Batch, 9, 9)
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # RNN requires flattened input (1D sequence)
        if model_name == 'rnn':
            # Flatten from (Batch, 9, 9) to (Batch, 81)
            inputs = inputs.view(inputs.size(0), -1)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)  # (Batch, 9, 9, 9)
        
        # Compute loss
        outputs_flat = outputs.reshape(-1, 9)
        targets_flat = targets.reshape(-1)
        
        loss = criterion(outputs_flat, targets_flat)
        
        # Backward pass
        loss.backward()
        
        # !!! ВАЖЛИВО ДЛЯ GNN: Обрізання градієнтів !!!
        # Це рятує від стрибків loss і NaN
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Calculate accuracy
        predictions = torch.argmax(outputs, dim=-1)
        metrics = calculate_accuracy(predictions, targets, inputs)
        
        # Update metrics
        total_loss += loss.item()
        total_cell_acc += metrics['cell_accuracy']
        total_empty_cell_acc += metrics['empty_cell_accuracy']
        total_board_acc += metrics['board_accuracy']
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'cell_acc': f"{metrics['cell_accuracy']:.4f}",
            'board_acc': f"{metrics['board_accuracy']:.4f}"
        })
    
    return {
        'loss': total_loss / num_batches,
        'cell_accuracy': total_cell_acc / num_batches,
        'empty_cell_accuracy': total_empty_cell_acc / num_batches,
        'board_accuracy': total_board_acc / num_batches
    }


@torch.no_grad()
def validate(model: nn.Module, 
             dataloader: DataLoader, 
             criterion: nn.Module,
             device: str,
             model_name: str = None) -> dict:
    """
    Validate the model
    """
    model.eval()
    
    total_loss = 0
    total_cell_acc = 0
    total_empty_cell_acc = 0
    total_board_acc = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc="Validation")
    
    for inputs, targets in pbar:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # RNN requires flattened input (1D sequence)
        if model_name == 'rnn':
            # Flatten from (Batch, 9, 9) to (Batch, 81)
            inputs = inputs.view(inputs.size(0), -1)
        
        # Forward pass
        outputs = model(inputs)  # (Batch, 9, 9, 9)
        
        # Compute loss
        outputs_flat = outputs.reshape(-1, 9)
        targets_flat = targets.reshape(-1)
        loss = criterion(outputs_flat, targets_flat)
        
        # Calculate accuracy
        predictions = torch.argmax(outputs, dim=-1)
        metrics = calculate_accuracy(predictions, targets, inputs)
        
        # Update metrics
        total_loss += loss.item()
        total_cell_acc += metrics['cell_accuracy']
        total_empty_cell_acc += metrics['empty_cell_accuracy']
        total_board_acc += metrics['board_accuracy']
        num_batches += 1
        
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'cell_acc': f"{metrics['cell_accuracy']:.4f}",
            'board_acc': f"{metrics['board_accuracy']:.4f}"
        })
    
    return {
        'loss': total_loss / num_batches,
        'cell_accuracy': total_cell_acc / num_batches,
        'empty_cell_accuracy': total_empty_cell_acc / num_batches,
        'board_accuracy': total_board_acc / num_batches
    }


def main():
    parser = argparse.ArgumentParser(description="Train Sudoku Neural Network")
    
    # Model selection
    parser.add_argument('--model', type=str, default='baseline', 
                        choices=['baseline', 'advanced', 'gnn', 'rnn'],
                        help='Model architecture to use')
    
    # Data parameters
    parser.add_argument('--data', type=str, default='data/sudoku.csv',
                        help='Path to CSV dataset')
    parser.add_argument('--train-split', type=float, default=0.8,
                        help='Training split ratio')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of samples (for debugging/testing)')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='Weight decay')
    
    # Model-specific parameters
    parser.add_argument('--hidden-channels', type=int, default=128,
                        help='Number of hidden channels')
    parser.add_argument('--num-residual-blocks', type=int, default=20,
                        help='Number of residual blocks (for advanced model)')
    parser.add_argument('--num-gnn-layers', type=int, default=8,
                        help='Number of GNN layers (for GNN model)')
    
    # Resume training argument
    # !!! НОВИЙ АРГУМЕНТ !!!
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from (e.g., weights/gnn_best.pth)')
    
    # Other parameters
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='Number of data loading workers')
    parser.add_argument('--save-dir', type=str, default='weights',
                        help='Directory to save model weights')
    
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("=" * 60)
    print("Sudoku Neural Network Training")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    if args.resume:
        print(f"Resuming from: {args.resume}")
    print("=" * 60)
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = SudokuDataset(args.data, train=True, train_split=args.train_split, limit=args.limit)
    val_dataset = SudokuDataset(args.data, train=False, train_split=args.train_split, limit=args.limit)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True if args.device == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True if args.device == 'cuda' else False
    )
    
    # Create model
    print("\nInitializing model...")
    if args.model == 'baseline':
        model = CNNBaseline(hidden_channels=args.hidden_channels)
    elif args.model == 'advanced':
        model = CNNAdvanced(
            hidden_channels=args.hidden_channels,
            num_residual_blocks=args.num_residual_blocks
        )
    elif args.model == 'gnn':
        model = GNNModel(
            hidden_channels=args.hidden_channels,
            num_layers=args.num_gnn_layers
        )
    elif args.model == 'rnn':
        model = SudokuRNN(
            embedding_dim=64,
            hidden_size=128,
            num_layers=2
        )
    
    model = model.to(args.device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # --- RESUME LOGIC ---
    start_epoch = 1
    best_val_loss = float('inf')
    
    if args.resume and os.path.exists(args.resume):
        print(f"\n🔄 Loading checkpoint from {args.resume}...")
        try:
            checkpoint = torch.load(args.resume, map_location=args.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Відновлюємо епоху і loss
            start_epoch = checkpoint['epoch'] + 1
            if 'val_loss' in checkpoint:
                best_val_loss = checkpoint['val_loss']
            
            print(f"✅ Checkpoint loaded. Resuming from epoch {start_epoch}.")
            print(f"   Previous best validation loss: {best_val_loss:.4f}")
        except Exception as e:
            print(f"⚠️ Error loading checkpoint: {e}")
            print("   Starting training from scratch.")
    else:
        print("\n⭐ Starting fresh training")
    
    history = []
    
    # --- TRAINING LOOP ---
    print("\nStarting training...")
    # Змінено range, щоб враховувати start_epoch
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 60)
        
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, args.device, args.model)
        val_metrics = validate(model, val_loader, criterion, args.device, args.model)
        
        scheduler.step(val_metrics['loss'])
        
        print(f"\nTrain Loss: {train_metrics['loss']:.4f} | "
              f"Cell Acc: {train_metrics['cell_accuracy']:.4f} | "
              f"Board Acc: {train_metrics['board_accuracy']:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f} | "
              f"Cell Acc: {val_metrics['cell_accuracy']:.4f} | "
              f"Board Acc: {val_metrics['board_accuracy']:.4f}")
        
        history.append({
            'epoch': epoch,
            'train': train_metrics,
            'val': val_metrics,
            'lr': optimizer.param_groups[0]['lr']
        })
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            save_path = os.path.join(args.save_dir, f'{args.model}_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'val_accuracy': val_metrics['cell_accuracy'],
                'args': vars(args)
            }, save_path)
            print(f"\n✓ Saved best model to {save_path}")
        
        # Save checkpoint (last state)
        # Це корисно, якщо навчання перерветься не на найкращій епосі
        checkpoint_path = os.path.join(args.save_dir, f'{args.model}_last.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_metrics['loss'],
            'val_accuracy': val_metrics['cell_accuracy'],
            'args': vars(args)
        }, checkpoint_path)
    
    # Save training history
    history_path = os.path.join(args.save_dir, f'{args.model}_history.json')
    # Якщо відновлюємо, треба б дописати в старий файл, але поки просто перезаписуємо
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()