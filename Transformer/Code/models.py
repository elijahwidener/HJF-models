import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel

class TemporalAttention(nn.Module):
    """Attention mechanism that incorporates temporal weights"""
    
    def __init__(self, hidden_size):
        super(TemporalAttention, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)
        
    def forward(self, hidden_states, attention_mask, temporal_weights=None):
        # Standard attention scores
        attention_scores = self.attention(hidden_states).squeeze(-1)
        
        # Apply attention mask (set masked positions to large negative value)
        # Changed from -1e9 to -1e4 to avoid overflow in half precision
        attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e4)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # If temporal weights provided, incorporate them
        if temporal_weights is not None:
            # We need to map the per-encounter weights to token-level weights
            # This is a simplified approach - in real implementation,
            # you would map weights based on [SEP] token positions
            
            # For now, we'll just use the first N weights where N is the number of tokens
            # This isn't ideal but prevents errors with variable-size temporal weights
            token_count = attention_weights.shape[1]
            
            # Create token-level weights (initialized to 1.0)
            token_weights = torch.ones(attention_weights.shape, device=attention_weights.device)
            
            # Set token weights based on the first encounter weight
            # This is simplified - ideally weights would be assigned per encounter
            for i in range(attention_weights.shape[0]):  # For each item in batch
                token_weights[i] = token_weights[i] * temporal_weights[i][0]
            
            # Combine with attention weights and re-normalize
            attention_weights = attention_weights * token_weights
            attention_weights = attention_weights / (attention_weights.sum(dim=1, keepdim=True) + 1e-9)
        
        # Apply attention weights to hidden states
        context_vector = torch.bmm(attention_weights.unsqueeze(1), hidden_states).squeeze(1)
        
        return context_vector, attention_weights

class MultiTaskTBIPredictor(nn.Module):
    """Multi-task model for predicting mental health diagnosis across time windows"""
    
    def __init__(self, pretrained_model_name, num_windows=4, dropout_rate=0.1):
        super(MultiTaskTBIPredictor, self).__init__()
        
        # Load pre-trained model
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        hidden_size = self.bert.config.hidden_size
        
        # Temporal attention layer
        self.temporal_attention = TemporalAttention(hidden_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Shared layers
        self.shared_layer = nn.Linear(hidden_size, hidden_size)
        
        # Add layers for incorporating previous window predictions
        self.prev_projection = nn.Linear(num_windows-1, hidden_size // 2)
        self.combined_layer = nn.Linear(hidden_size + hidden_size // 2, hidden_size)
        
        # Task-specific layers for each window
        self.task_layers = nn.ModuleList([
            nn.Linear(hidden_size, 1) for _ in range(num_windows)
        ])
    
    def forward(self, input_ids, attention_mask, temporal_weights=None, prev_window_preds=None):
        """
        Forward pass with optional previous window predictions
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            temporal_weights: Temporal weights for attention
            prev_window_preds: Previous window predictions (batch_size, num_prev_windows)
                For 30-day window: None (no previous windows)
                For 60-day window: 30-day predictions (batch_size, 1) 
                For 180-day window: 30,60-day predictions (batch_size, 2)
                For 365-day window: 30,60,180-day predictions (batch_size, 3)
        """
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Get sequence output (hidden states for all tokens)
        sequence_output = outputs.last_hidden_state
        
        # Apply temporal attention
        context_vector, attention_weights = self.temporal_attention(
            sequence_output, 
            attention_mask,
            temporal_weights
        )
        
        # Process previous window predictions if provided
        if prev_window_preds is not None:
            # Project previous predictions to higher dimension
            prev_features = self.prev_projection(prev_window_preds)
            
            # Combine with context vector
            combined_vector = torch.cat([context_vector, prev_features], dim=1)
            
            # Project back to original dimension
            context_vector = self.combined_layer(combined_vector)
        
        # Apply shared layer
        shared_features = F.relu(self.shared_layer(context_vector))
        shared_features = self.dropout(shared_features)
        
        # Apply task-specific layers
        task_outputs = []
        for task_layer in self.task_layers:
            task_output = torch.sigmoid(task_layer(shared_features))
            task_outputs.append(task_output)
        
        # Stack outputs for all tasks/windows
        stacked_outputs = torch.cat(task_outputs, dim=1)
        
        return stacked_outputs, attention_weights