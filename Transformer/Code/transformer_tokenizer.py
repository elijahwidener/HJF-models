import numpy as np
import json
import os
import tensorflow as tf
from keras import Tokenizer
from keras import pad_sequences

class EncounterTokenizer:
    """
    Specialized tokenizer for converting medical encounter text tokens to numerical sequences.
    
    This tokenizer is designed specifically for clinical encounter data, with features for:
    1. Handling patient-level sequences of encounters
    2. Creating padded sequences with proper attention masks
    3. Building transformer-ready inputs with special tokens (CLS, SEP)
    4. Managing out-of-vocabulary tokens in clinical text
    
    Unlike standard NLP tokenizers, this tokenizer preserves the hierarchical structure
    of patient data (patients → encounters → tokens) and provides utilities to flatten
    this structure appropriately for transformer models.
    
    Args:
        vocab_size (int): Maximum size of vocabulary to use
        max_sequence_length (int): Maximum sequence length to pad/truncate to
        oov_token (str): Token to use for out-of-vocabulary tokens
    """
    
    def __init__(self, vocab_size=10000, max_sequence_length=100, oov_token="<UNK>"):
        """
        Initialize the tokenizer
        
        Args:
            vocab_size: Maximum size of vocabulary to use
            max_sequence_length: Maximum sequence length to pad/truncate to
            oov_token: Token to use for out-of-vocabulary tokens
        """
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.oov_token = oov_token
        self.tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
        self.word_index = None
        self.is_fit = False
    
    def fit_on_texts(self, encounter_texts):
        """
        Fit the tokenizer on a list of encounter texts
        
        Args:
            encounter_texts: List of encounter text strings
        """
        # Flatten list if it contains patient sequences
        if isinstance(encounter_texts, dict) or (isinstance(encounter_texts, list) and 
                                              isinstance(encounter_texts[0], list)):
            # If input is patient data dictionary
            if isinstance(encounter_texts, dict):
                texts = []
                for patient in encounter_texts.values():
                    texts.extend(patient['pre_tbi_tokens'])
            else:
                # If input is list of patient sequences
                texts = [text for patient in encounter_texts for text in patient]
        else:
            texts = encounter_texts
        
        print(f"Fitting tokenizer on {len(texts)} encounter texts")
        self.tokenizer.fit_on_texts(texts)
        self.word_index = self.tokenizer.word_index
        self.is_fit = True
        
        print(f"Vocabulary size: {len(self.word_index)} unique tokens")
        print(f"Using top {self.vocab_size} tokens")
        
        # Add special tokens if needed
        if "<PAD>" not in self.word_index:
            print("Warning: <PAD> token not found in vocabulary")
        if self.oov_token not in self.word_index:
            print(f"Warning: OOV token {self.oov_token} not found in vocabulary")
    
    def texts_to_sequences(self, encounter_texts):
        """
        Convert encounter texts to integer sequences
        
        Args:
            encounter_texts: List of encounter text strings
            
        Returns:
            List of integer sequences
        """
        if not self.is_fit:
            raise ValueError("Tokenizer must be fit before converting texts to sequences")
        
        return self.tokenizer.texts_to_sequences(encounter_texts)
    
    def patient_data_to_sequences(self, patient_data):
        """
        Convert patient data dictionary to padded sequences and labels
        
        Args:
            patient_data: Dictionary of patient data with pre_tbi_tokens
            
        Returns:
            Tuple of (padded_sequences, attention_mask)
        """
        if not self.is_fit:
            raise ValueError("Tokenizer must be fit before converting patient data")
        
        sequences = []
        patient_ids = []
        
        for patient_id, patient in patient_data.items():
            # Convert each encounter to a sequence of integers
            patient_sequence = self.tokenizer.texts_to_sequences(patient['pre_tbi_tokens'])
            sequences.append(patient_sequence)
            patient_ids.append(patient_id)
        
        # Pad/truncate sequences to a uniform length
        return self.create_padded_sequences(sequences), patient_ids
    
    def create_padded_sequences(self, sequences):
        """
        Pad sequences to a uniform length and create attention masks
        
        Args:
            sequences: List of list of integer sequences
            
        Returns:
            Tuple of (padded_sequences, attention_mask)
        """
        # First, determine the maximum encounter sequence length
        max_encounters = min(self.max_sequence_length, max(len(seq) for seq in sequences))
        
        # Now, for each patient, pad/truncate their encounter sequences
        padded_data = []
        attention_masks = []
        
        for patient_sequence in sequences:
            # If too long, truncate
            if len(patient_sequence) > max_encounters:
                patient_sequence = patient_sequence[-max_encounters:]
                mask = [1] * max_encounters
            else:
                # If too short, create mask and append padding
                mask = [1] * len(patient_sequence) + [0] * (max_encounters - len(patient_sequence))
                # Pad with empty lists
                patient_sequence = patient_sequence + [[]] * (max_encounters - len(patient_sequence))
            
            # Now pad each encounter to a uniform token length
            padded_encounters = []
            for encounter in patient_sequence:
                # Pad/truncate to uniform length
                if len(encounter) > 0:
                    padded = pad_sequences([encounter], maxlen=512, padding='post', truncating='post')[0]
                else:
                    padded = [0] * 512  # Empty padding for missing encounters
                
                padded_encounters.append(padded)
            
            padded_data.append(padded_encounters)
            attention_masks.append(mask)
        
        return np.array(padded_data), np.array(attention_masks)
    
    def create_transformer_inputs(self, patient_data, outcomes_window=60):
        """
        Create input tensors ready to feed into a transformer model
        
        Args:
            patient_data: Dictionary of patient data with pre_tbi_tokens
            outcomes_window: Window (in days) for outcome labels
            
        Returns:
            Dictionary with model inputs and labels
        """
        # Convert patient data to sequences
        sequences, patient_ids = self.patient_data_to_sequences(patient_data)
        padded_data, attention_masks = sequences
        
        # Flatten the 3D patient-encounter-token data to 2D input format
        # Instead of [patients, encounters, tokens] we need [patients, tokens]
        # where each patient's encounters are concatenated with special tokens
        
        max_tokens_per_patient = self.max_sequence_length * 512
        flattened_data = np.zeros((len(padded_data), max_tokens_per_patient))
        
        # Special tokens
        CLS_token = 1  # We'll use 1 for [CLS]
        SEP_token = 2  # We'll use 2 for [SEP] between encounters
        
        for i, patient in enumerate(padded_data):
            position = 0
            flattened_data[i, position] = CLS_token
            position += 1
            
            for j, encounter in enumerate(patient):
                # Only process real encounters (where mask is 1)
                if j < len(attention_masks[i]) and attention_masks[i][j] == 1:
                    # Add tokens from this encounter (skip padding)
                    for token in encounter:
                        if token != 0:  # Skip padding tokens
                            if position < max_tokens_per_patient:
                                flattened_data[i, position] = token
                                position += 1
                    
                    # Add separator token
                    if position < max_tokens_per_patient:
                        flattened_data[i, position] = SEP_token
                        position += 1
        
        # Create new attention mask for the flattened data
        flattened_masks = (flattened_data > 0).astype(np.int32)
        
        # Get outcome labels
        labels = {}
        for window in [30, 60, 180, 365]:
            window_labels = []
            for patient_id in patient_ids:
                # Look up outcome for this patient and window
                found = False
                outcome_file = f"text_token_analysis/outcomes_{window}day.csv"
                if os.path.exists(outcome_file):
                    outcomes_df = pd.read_csv(outcome_file)
                    patient_outcome = outcomes_df[outcomes_df['patient_id'] == patient_id]
                    if len(patient_outcome) > 0:
                        window_labels.append(int(patient_outcome.iloc[0]['has_mh']))
                        found = True
                
                if not found:
                    # If we can't find it, assume negative
                    window_labels.append(0)
            
            labels[f'window_{window}'] = np.array(window_labels)
        
        print(f"Created transformer inputs: {flattened_data.shape}")
        print(f"Attention masks: {flattened_masks.shape}")
        
        return {
            'input_ids': flattened_data,
            'attention_mask': flattened_masks,
            'labels': labels,
            'patient_ids': np.array(patient_ids)
        }
    
    def save(self, filepath="tokenizer_data"):
        """
        Save the tokenizer data to JSON
        
        Args:
            filepath: Directory to save the tokenizer data
        """
        if not self.is_fit:
            raise ValueError("Tokenizer must be fit before saving")
        
        os.makedirs(filepath, exist_ok=True)
        
        # Save tokenizer config
        tokenizer_json = self.tokenizer.to_json()
        with open(f"{filepath}/tokenizer.json", 'w') as f:
            f.write(tokenizer_json)
        
        # Save config
        config = {
            'vocab_size': self.vocab_size,
            'max_sequence_length': self.max_sequence_length,
            'oov_token': self.oov_token
        }
        
        with open(f"{filepath}/tokenizer_config.json", 'w') as f:
            json.dump(config, f)
        
        print(f"Tokenizer saved to {filepath}")
    
    @classmethod
    def load(cls, filepath="tokenizer_data"):
        """
        Load a saved tokenizer
        
        Args:
            filepath: Directory containing the tokenizer data
            
        Returns:
            EncounterTokenizer instance
        """
        # Load config
        with open(f"{filepath}/tokenizer_config.json", 'r') as f:
            config = json.load(f)
        
        # Create instance
        tokenizer = cls(
            vocab_size=config['vocab_size'],
            max_sequence_length=config['max_sequence_length'],
            oov_token=config['oov_token']
        )
        
        # Load tokenizer
        with open(f"{filepath}/tokenizer.json", 'r') as f:
            tokenizer_json = f.read()
            
        tokenizer.tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_json)
        tokenizer.word_index = tokenizer.tokenizer.word_index
        tokenizer.is_fit = True
        
        print(f"Tokenizer loaded from {filepath}")
        print(f"Vocabulary size: {len(tokenizer.word_index)} unique tokens")
        
        return tokenizer
    
    def get_vocab_size(self):
        """
        Get the actual vocabulary size
        
        Returns:
            int: Actual vocabulary size
        """
        if not self.is_fit:
            return 0
        
        return min(self.vocab_size, len(self.word_index) + 1)
        
    def decode_sequence(self, sequence):
        """
        Decode a sequence of token IDs back to text
        
        Args:
            sequence: List of token IDs
            
        Returns:
            String representing the decoded sequence
        """
        if not self.is_fit:
            raise ValueError("Tokenizer must be fit before decoding")
        
        # Create inverse word index
        inverse_word_index = {v: k for k, v in self.word_index.items()}
        
        # Decode sequence
        decoded = []
        for token_id in sequence:
            if token_id == 0:  # Skip padding
                continue
            word = inverse_word_index.get(token_id, self.oov_token)
            decoded.append(word)
        
        return " ".join(decoded)