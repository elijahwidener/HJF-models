import torch
import torch.nn.functional as F
import numpy as np
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, average_precision_score
import matplotlib.pyplot as plt
import os
import json
from config import DEVICE
import time


def train_epoch(model, dataloader, optimizer, scheduler, device, class_weights=None):
    """
    Train the model for one epoch
    """
    model.train()
    total_loss = 0
    data_load_time = 0
    forward_time = 0
    backward_time = 0
    total_time = 0
    
    epoch_start = time.time()
    batch_start = time.time()
    
    # Initialize gradient scaler for AMP
    scaler = GradScaler() if torch.cuda.is_available() else None
    
    for batch_idx, batch in enumerate(dataloader):
        # Time data loading
        data_load_end = time.time()
        data_load_time += (data_load_end - batch_start)
        
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        temporal_weights = batch['temporal_weights'].to(device)
        
        # Clear gradients
        optimizer.zero_grad()
        
        # Forward pass with timing
        forward_start = time.time()
        
        # Use AMP for faster training on GPU
        if torch.cuda.is_available():
            with autocast():
                outputs, _ = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    temporal_weights=temporal_weights
                )
                # Apply class weights if provided
                if class_weights is not None:
                    loss = F.binary_cross_entropy(outputs, labels, weight=class_weights.to(device))  
                else:
                    loss = F.binary_cross_entropy(outputs, labels)            
            forward_end = time.time()
            forward_time += (forward_end - forward_start)
            
            # Backward pass with timing
            backward_start = time.time()
            
            # Scale gradients and backward pass
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            # Regular training for CPU
            outputs, _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                temporal_weights=temporal_weights
            )
            if class_weights is not None:
                loss = F.binary_cross_entropy(outputs, labels, weight=class_weights.to(device))
            else:
                loss = F.binary_cross_entropy(outputs, labels)            
            forward_end = time.time()
            forward_time += (forward_end - forward_start)
            
            # Backward pass with timing
            backward_start = time.time()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        scheduler.step()
        
        backward_end = time.time()
        backward_time += (backward_end - backward_start)
        
        # Update total loss
        total_loss += loss.item()

        # Print progress every 10 batches
        if (batch_idx + 1) % 10 == 0:
            batch_end = time.time()
            batch_time = batch_end - batch_start
            total_time += batch_time
            
            print(f"  Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}")
            print(f"    Times: Batch={batch_time:.2f}s, Data={data_load_time/10:.2f}s, Forward={forward_time/10:.2f}s, Backward={backward_time/10:.2f}s")
            
            # Reset timers
            data_load_time = 0
            forward_time = 0
            backward_time = 0
            batch_start = time.time()
    
    epoch_time = time.time() - epoch_start
    avg_loss = total_loss / len(dataloader)
    
    print(f"Epoch completed in {epoch_time:.2f}s, Avg loss: {avg_loss:.4f}")
    
    return avg_loss


def evaluate(model, dataloader, device):
    """
    Evaluate the model
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader for evaluation data
        device: Device to use
        
    Returns:
        results: Dictionary of evaluation metrics
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            temporal_weights = batch['temporal_weights'].to(device)
            
            # Forward pass
            outputs, _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                temporal_weights=temporal_weights
            )
            
            # Store predictions and labels
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    # Calculate metrics for each window
    results = {}
    windows = [30, 60, 180, 365]
    
    for i, window in enumerate(windows):
        window_preds = all_preds[:, i]
        window_labels = all_labels[:, i]
        
        # Convert to binary predictions
        binary_preds = (window_preds > 0.5).astype(int)
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            window_labels, binary_preds, average='binary', zero_division=0
        )
        
        # Calculate ROC AUC and PR AUC
        roc_auc = roc_auc_score(window_labels, window_preds)
        pr_auc = average_precision_score(window_labels, window_preds)
        
        results[f'{window}_day'] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc
        }
    
    return results

def train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, device, 
                num_epochs=3, patience=2, class_weights_list=None):
    """
    Train the model with early stopping
    
    Args:
        model: Model to train
        train_dataloader: DataLoader for training data
        val_dataloader: DataLoader for validation data
        optimizer: Optimizer to use
        scheduler: Learning rate scheduler
        device: Device to use
        num_epochs: Maximum number of epochs to train
        patience: Patience for early stopping
        
    Returns:
        model: Trained model
        history: Training history
    """
    best_val_auc = 0
    best_model = None
    patience_counter = 0
    history = {
        'train_loss': [],
        'val_metrics': []
    }
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Train for one epoch
        avg_train_loss = train_epoch(model, train_dataloader, optimizer, scheduler, device, class_weights_list)        
        history['train_loss'].append(avg_train_loss)
        
        print(f"  Train Loss: {avg_train_loss:.4f}")
        
        # Evaluate on validation set
        val_metrics = evaluate(model, val_dataloader, device)
        history['val_metrics'].append(val_metrics)
        
        # Print validation metrics
        print("  Validation Metrics:")
        for window, metrics in val_metrics.items():
            print(f"    {window}: ROC AUC: {metrics['roc_auc']:.4f}, PR AUC: {metrics['pr_auc']:.4f}")
        
        # Calculate average ROC AUC across all windows
        avg_val_auc = np.mean([metrics['roc_auc'] for metrics in val_metrics.values()])
        
        # Check if this is the best model
        if avg_val_auc > best_val_auc:
            best_val_auc = avg_val_auc
            best_model = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping after {epoch+1} epochs")
            break
    
    # Load best model
    if best_model is not None:
        model.load_state_dict(best_model)
    
    return model, history

def plot_training_history(history, output_dir):
    """Plot training loss and validation metrics"""
    
    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(f"{output_dir}/training_loss.png")
    plt.close()
    
    # Plot validation ROC AUC for each window
    plt.figure(figsize=(10, 5))
    
    windows = ['30_day', '60_day', '180_day', '365_day']
    for window in windows:
        auc_values = [metrics[window]['roc_auc'] for metrics in history['val_metrics']]
        plt.plot(auc_values, label=window)
    
    plt.title('Validation ROC AUC')
    plt.xlabel('Epoch')
    plt.ylabel('ROC AUC')
    plt.legend()
    plt.savefig(f"{output_dir}/validation_auc.png")
    plt.close()
