import torch
from torch.utils.data import Dataset
import numpy as np

class TBIEncounterDataset(Dataset):
    """Dataset for TBI patient encounters"""
    
    def __init__(self, processed_data, tokenizer, max_seq_length=512, max_encounters=30):
        """
        Initialize the dataset
        
        Args:
            processed_data: List of dictionaries with processed patient data
            tokenizer: Tokenizer for encoding text
            max_seq_length: Maximum sequence length for each encounter
            max_encounters: Maximum number of encounters to consider
        """
        self.processed_data = processed_data
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.max_encounters = max_encounters
        
        # Windows to predict
        self.windows = [30, 60, 180, 365]
        
    def __len__(self):
        return len(self.processed_data)
    
    def __getitem__(self, idx):
        patient_data = self.processed_data[idx]
        
        # Get patient encounters and truncate if too many
        encounters = patient_data['encounters'][-self.max_encounters:]
        
        # Get weights and pad or truncate to max_encounters
        weights = patient_data['weights'][-self.max_encounters:]
        
        # Create a fixed-size tensor and fill it with weights or zeros
        fixed_weights = torch.zeros(self.max_encounters, dtype=torch.float)
        fixed_weights[:len(weights)] = torch.tensor(weights, dtype=torch.float)
        
        # Combine encounters with separators
        combined_text = " " + self.tokenizer.sep_token + " ".join(encounters)
        
        # Tokenize the combined text
        encoding = self.tokenizer(
            combined_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_seq_length,
            return_tensors='pt'
        )
        
        # Get labels for each time window
        labels = torch.tensor([patient_data['outcomes'][window] for window in self.windows], 
                            dtype=torch.float)
        
        # Create sample with all the needed information
        sample = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': labels,
            'temporal_weights': fixed_weights,
            'patient_id': patient_data['patient_id']
        }
        
        return sample
