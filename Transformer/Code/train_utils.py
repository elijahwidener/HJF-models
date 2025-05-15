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
from predict import predict_with_window_ensembling


# In train_utils.py

def train_epoch(model, dataloader, optimizer, scheduler, device, class_weights=None):
    """
    Train the model for one epoch with support for class weighting to handle imbalanced data.
    
    This function handles a complete training epoch including:
    1. Forward and backward passes through the model
    2. Application of window-specific class weights to address class imbalance
    3. Gradient clipping to prevent exploding gradients
    4. Learning rate scheduling
    5. Progress reporting
    
    The class weighting is particularly important for medical prediction tasks where
    positive cases (patients developing mental health conditions) are typically much
    rarer than negative cases, creating a class imbalance.
    
    Args:
        model (MultiTaskTBIPredictor): The model to train
        dataloader (DataLoader): DataLoader containing training data
        optimizer (torch.optim.Optimizer): Optimizer for parameter updates
        scheduler: Learning rate scheduler
        device (torch.device): Device to use for computation
        class_weights (list, optional): List of weights for positive class in each window
            to address class imbalance
            
    Returns:
        float: Average loss for the epoch
    """
    model.train()
    total_loss = 0
    
    for batch_idx, batch in enumerate(dataloader):
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        temporal_weights = batch['temporal_weights'].to(device) if 'temporal_weights' in batch else None
        
        # Clear gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs, _ = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            temporal_weights=temporal_weights
        )
        
        # Apply class weights based on window-specific imbalance
        if class_weights is not None:
            # Calculate weighted BCE loss for each window prediction
            loss = 0
            for i in range(labels.shape[1]):
                window_labels = labels[:, i]
                window_outputs = outputs[:, i]
                window_weight = class_weights[i]
                
                # Create weights tensor for each sample based on its class
                sample_weights = torch.ones_like(window_labels)
                sample_weights[window_labels == 1] = window_weight
                
                # Apply weighted BCE loss
                window_loss = F.binary_cross_entropy(
                    window_outputs, 
                    window_labels, 
                    weight=sample_weights
                )
                loss += window_loss
            
            # Average loss across windows
            loss = loss / labels.shape[1]
        else:
            # Standard BCE loss if no weights provided
            loss = F.binary_cross_entropy(outputs, labels)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        # Update total loss
        total_loss += loss.item()
        
        # Print progress every 10 batches
        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch completed, Avg loss: {avg_loss:.4f}")
    
    return avg_loss

def train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, device, 
               num_epochs=3, patience=2):
    """
    Train the model with early stopping and class weighting
    """
    best_val_auc = 0
    best_model = None
    patience_counter = 0
    history = {
        'train_loss': [],
        'val_metrics': []
    }
    
    # Define class weights based on imbalance rates
    # Using inverse of positive class rate to weight rare positive examples higher
    pos_rates = [0.236, 0.294, 0.371, 0.444]  # 30, 60, 180, 365 days
    class_weights = [1.0 / rate for rate in pos_rates]
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Train for one epoch with class weights
        avg_train_loss = train_epoch(
            model, train_dataloader, optimizer, scheduler, device, 
            class_weights=class_weights
        )
        
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
    
    all_preds, all_labels = predict_with_window_ensembling(model, dataloader, device)
    
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
