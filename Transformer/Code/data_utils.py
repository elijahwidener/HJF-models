import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from config import RANDOM_SEED

def load_patient_data(token_data_dir="text_token_analysis"):
    """
    Load patient data from the token data directory
    
    Args:
        token_data_dir: Directory containing token data
        
    Returns:
        patient_data: Dictionary of patient data
        outcomes: Dictionary of outcomes by window
    """
    # Load patient token data
    with open(f"{token_data_dir}/patient_tokens.json", 'r') as f:
        patient_data = json.load(f)
    
    # Load outcomes for different windows
    outcomes = {}
    for window in [30, 60, 180, 365]:
        outcomes_df = pd.read_csv(f"{token_data_dir}/outcomes_{window}day.csv")
        # Convert to dictionary for faster lookup
        outcomes[window] = dict(zip(outcomes_df['patient_id'].astype(str), 
                                   outcomes_df['has_mh'].astype(int)))
    
    return patient_data, outcomes

def calculate_encounter_weights(days_from_tbi, lambda_param=0.01):
    """
    Calculate time-based weights for encounters
    
    Args:
        days_from_tbi: List/array of days from TBI for each encounter
        lambda_param: Decay parameter for exponential weighting
        
    Returns:
        weights: Array of weights
    """
    days = np.abs(np.array(days_from_tbi))
    return np.exp(-lambda_param * days)

def prepare_encounter_data(patient_data, outcomes):
    """
    Prepare encounter data with temporal information
    
    Args:
        patient_data: Dictionary of patient data
        outcomes: Dictionary of outcomes by window
        
    Returns:
        processed_data: List of dictionaries with processed data
    """
    processed_data = []
    
    for patient_id, patient in patient_data.items():
        # Skip patients with no pre-TBI data
        if len(patient['pre_tbi_tokens']) == 0:
            continue
        
        # Combine sequential encounters with special tokens
        encounters = patient['pre_tbi_tokens']
        
        # Estimate days from TBI based on encounter order (if not available)
        # In a real implementation, you'd use actual days_from_tbi
        # Here we're just using encounter index as a proxy
        days_from_tbi = [-i for i in range(len(encounters), 0, -1)]
        
        # Calculate encounter weights (higher weight for encounters closer to TBI)
        weights = calculate_encounter_weights(days_from_tbi)
        
        # Get outcomes for each time window
        patient_outcomes = {}
        for window in [30, 60, 180, 365]:
            patient_outcomes[window] = outcomes[window].get(patient_id, 0)
        
        processed_data.append({
            'patient_id': patient_id,
            'encounters': encounters,
            'days_from_tbi': days_from_tbi,
            'weights': weights.tolist(),
            'outcomes': patient_outcomes
        })
    
    return processed_data

def split_dataset(processed_data, test_size=0.2, val_size=0.1):
    """
    Split the dataset into train, validation, and test sets
    
    Args:
        processed_data: List of processed patient data
        test_size: Fraction of data to use for testing
        val_size: Fraction of data to use for validation
        
    Returns:
        train_data, val_data, test_data: Split datasets
    """
    # First split into train+val and test
    train_val, test = train_test_split(processed_data, test_size=test_size, random_state=RANDOM_SEED)
    
    # Then split train+val into train and val
    val_size_adjusted = val_size / (1 - test_size)
    train, val = train_test_split(train_val, test_size=val_size_adjusted, random_state=RANDOM_SEED)
    
    print(f"Dataset split: {len(train)} train, {len(val)} validation, {len(test)} test")
    
    return train, val, test