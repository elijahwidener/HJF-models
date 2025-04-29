import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

# 1. Load the tokenizer for Clinical BERT
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

# 2. Create a custom dataset class for your encounter data
class EncounterDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Convert to pytorch tensors
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float)
        }
        return item

# 3. Load and prepare your encounter data
def prepare_patient_data(patient_data, window_days=60):
    texts = []
    labels = []
    
    for patient_id, patient in patient_data.items():
        # Combine all pre-TBI encounters into a single text
        # You might want to add special tokens to separate encounters
        combined_text = " [SEP] ".join(patient['pre_tbi_tokens'])
        texts.append(combined_text)
        
        # Get the label for this window
        # This assumes you have a way to look up labels from your outcomes data
        has_mh = get_outcome_label(patient_id, window_days)
        labels.append(has_mh)
    
    return texts, labels

# 4. Define custom metrics for evaluation
def compute_metrics(pred):
    predictions = (pred.predictions > 0).astype(np.int32)
    labels = pred.label_ids
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary'
    )
    
    # Calculate ROC AUC
    auc = roc_auc_score(labels, pred.predictions)
    
    return {
        'accuracy': (predictions == labels).mean(),
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }

# 5. Load the model with a classification head for your task
model = AutoModelForSequenceClassification.from_pretrained(
    "emilyalsentzer/Bio_ClinicalBERT", 
    num_labels=1  # Binary classification
)

# 6. Split data and create datasets
train_texts, train_labels = prepare_patient_data(train_patient_data)
val_texts, val_labels = prepare_patient_data(val_patient_data)

train_dataset = EncounterDataset(train_texts, train_labels, tokenizer)
val_dataset = EncounterDataset(val_texts, val_labels, tokenizer)

# 7. Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="auc",
    fp16=True  # Use mixed precision training
)

# 8. Create Trainer and train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# 9. Fine-tune the model
trainer.train()

# 10. Save the fine-tuned model
model.save_pretrained("./tbi_mental_health_model")
tokenizer.save_pretrained("./tbi_mental_health_model")

# 11. Make predictions on test data
test_texts, test_labels = prepare_patient_data(test_patient_data)
test_dataset = EncounterDataset(test_texts, test_labels, tokenizer)

predictions = trainer.predict(test_dataset)
print(compute_metrics(predictions))