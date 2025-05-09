import os
import json
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, logging
from torch.optim import AdamW
import numpy as np

# Local imports
from config import (
    PRETRAINED_MODEL, BATCH_SIZE, EVAL_BATCH_SIZE, 
    NUM_EPOCHS, PATIENCE, DEVICE, LEARNING_RATE, WEIGHT_DECAY, RANDOM_SEED
)
from data_utils import load_patient_data, prepare_encounter_data, split_dataset
from dataset import TBIEncounterDataset
from models import MultiTaskTBIPredictor
from train_utils import train_epoch, evaluate, plot_training_history
from predict import risk_stratify_patients

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    print(f"CUDA available: Using {torch.cuda.get_device_name(0)}")
else:
    print("CUDA not available: Using CPU")
    
# Set up logging
logging.set_verbosity_info()

# Set random seeds for reproducibility
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

def collate_fn(batch, max_length=512):
    """
    Custom collate function that truncates sequences to specified max_length
    """
    input_ids = []
    attention_masks = []
    labels = []
    temporal_weights = []
    patient_ids = []
    
    for item in batch:
        # Truncate input_ids and attention_mask
        input_ids.append(item['input_ids'][:max_length])
        attention_masks.append(item['attention_mask'][:max_length])
        
        # Other items don't need truncation
        labels.append(item['labels'])
        temporal_weights.append(item['temporal_weights'])
        patient_ids.append(item['patient_id'])
    
    # Pad inputs to max_length
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True)
    attention_masks = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True)
    
    # Stack other items
    labels = torch.stack(labels)
    temporal_weights = torch.stack(temporal_weights)
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_masks,
        'labels': labels,
        'temporal_weights': temporal_weights,
        'patient_id': patient_ids
    }

def main():
    # 1. Load and prepare data
    print("Loading data...")
    patient_data, outcomes = load_patient_data()
    
    print("Preparing encounter data...")
    processed_data = prepare_encounter_data(patient_data, outcomes)
    
    print("Splitting dataset...")
    train_data, val_data, test_data = split_dataset(processed_data)
    
    # 2. Initialize tokenizer and model
    print("Initializing tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
    model = MultiTaskTBIPredictor(PRETRAINED_MODEL)
    model.to(DEVICE)
    
    # 3. Create datasets and dataloaders
    print("Creating datasets and dataloaders...")
    train_dataset = TBIEncounterDataset(train_data, tokenizer)
    val_dataset = TBIEncounterDataset(val_data, tokenizer)
    test_dataset = TBIEncounterDataset(test_data, tokenizer)
    
    # Print shapes to help debug
    sample = train_dataset[0]
    print(f"Sample input_ids shape: {sample['input_ids'].shape}")
    print(f"Sample attention_mask shape: {sample['attention_mask'].shape}")
    print(f"Sample labels shape: {sample['labels'].shape}")
    print(f"Sample temporal_weights shape: {sample['temporal_weights'].shape}")
    
    # Define sequence lengths for progressive resizing
    sequence_lengths = [128, 256, 384, 512]
    
    # 4. Set up optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Learning rate scheduler
    total_steps = sum([len(train_data) // BATCH_SIZE * 2 for _ in sequence_lengths])  # Estimate total steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # 5. Progressive resizing training
    print("Starting progressive resizing training...")
    best_val_auc = 0
    best_model_state = None
    patience_counter = 0
    history = {'train_loss': [], 'val_metrics': []}
    
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        
        # Select sequence length for this epoch
        len_idx = min(epoch // 2, len(sequence_lengths) - 1)  # Change length every 2 epochs
        curr_seq_len = sequence_lengths[len_idx]
        print(f"Using sequence length: {curr_seq_len}")
        
        # Create new dataloaders with current sequence length
        train_loader = DataLoader(
            train_dataset, 
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=True if torch.cuda.is_available() else False
        )

        val_loader = DataLoader(
            val_dataset, 
            batch_size=EVAL_BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        # Define class weights based on imbalance rates
        pos_rates = [0.236, 0.294, 0.371, 0.444]  # 30, 60, 180, 365 days
        class_weights = [1.0 / rate for rate in pos_rates]
        
        # Train for one epoch
        avg_train_loss = train_epoch(
            model, 
            train_loader, 
            optimizer, 
            scheduler, 
            DEVICE, 
            class_weights=class_weights
        )
        history['train_loss'].append(avg_train_loss)
        
        # Evaluate
        val_metrics = evaluate(model, val_loader, DEVICE)
        history['val_metrics'].append(val_metrics)
        
        # Print results
        print(f"Train Loss: {avg_train_loss:.4f}")
        print("Validation Metrics:")
        for window, metrics in val_metrics.items():
            print(f"  {window}: ROC AUC: {metrics['roc_auc']:.4f}, PR AUC: {metrics['pr_auc']:.4f}")
        
        # Calculate average AUC across all windows
        avg_val_auc = np.mean([metrics['roc_auc'] for metrics in val_metrics.values()])
        
        # Early stopping check
        if avg_val_auc > best_val_auc:
            best_val_auc = avg_val_auc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= PATIENCE:
            print(f"Early stopping after {epoch+1} epochs")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # 6. Evaluate on test set
    print("Evaluating on test set...")
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=EVAL_BATCH_SIZE, 
        shuffle=False,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_metrics = evaluate(model, test_dataloader, DEVICE)
    
    print("Test Metrics:")
    for window, metrics in test_metrics.items():
        print(f"  {window}: ROC AUC: {metrics['roc_auc']:.4f}, PR AUC: {metrics['pr_auc']:.4f}")
    
    # 7. Save model and results
    output_dir = "tbi_bert_model"
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = f"{output_dir}/model.pt"
    torch.save(model.state_dict(), model_path)
    
    # Save tokenizer for future use
    tokenizer.save_pretrained(output_dir)
    
    # Save test metrics
    with open(f"{output_dir}/test_metrics.json", 'w') as f:
        json.dump({k: {m: float(v) for m, v in metrics.items()} 
                  for k, metrics in test_metrics.items()}, f, indent=2)
    
    # Plot training history
    plot_training_history(history, output_dir)
    
    # 8. Risk stratification
    print("Performing risk stratification...")
    stratified_patients, predictions = risk_stratify_patients(model, test_dataloader, DEVICE)
    
    # Save stratification results
    with open(f"{output_dir}/risk_stratification.json", 'w') as f:
        json.dump({k: {risk: patients for risk, patients in groups.items()} 
                  for k, groups in stratified_patients.items()}, f, indent=2)
    
    print(f"Model and results saved to {output_dir}")
    
    return model, tokenizer, history, test_metrics


if __name__ == "__main__":
    main()