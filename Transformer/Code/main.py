import os
import json
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup, logging

# Local imports
from config import (
    PRETRAINED_MODEL, BATCH_SIZE, EVAL_BATCH_SIZE, 
    NUM_EPOCHS, PATIENCE, DEVICE, LEARNING_RATE, WEIGHT_DECAY, RANDOM_SEED
)
from data_utils import load_patient_data, prepare_encounter_data, split_dataset
from dataset import TBIEncounterDataset
from models import MultiTaskTBIPredictor
from train_utils import train_model, evaluate, plot_training_history
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
    
    # DataLoaders with appropriate batch sizes and worker settings
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=4,  
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=EVAL_BATCH_SIZE, 
        shuffle=False,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=EVAL_BATCH_SIZE, 
        shuffle=False,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # 4. Set up optimizer and scheduler
    # Use AdamW optimizer with weight decay
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Learning rate scheduler
    total_steps = len(train_dataloader) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # 5. Train the model
    print("Training model...")
    model, history = train_model(
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        scheduler,
        DEVICE,
        num_epochs=NUM_EPOCHS,
        patience=PATIENCE
    )
    
    # 6. Evaluate on test set
    print("Evaluating on test set...")
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