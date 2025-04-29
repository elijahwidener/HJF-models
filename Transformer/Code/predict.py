import torch
import numpy as np
from config import DEVICE

def risk_stratify_patients(model, dataloader, device, thresholds={'low': 0.3, 'high': 0.7}):
    """
    Stratify patients into risk categories
    
    Args:
        model: Trained model
        dataloader: DataLoader with patient data
        device: Device to use
        thresholds: Dictionary with thresholds for low/high risk
        
    Returns:
        stratified_patients: Dictionary with patient IDs by risk category and window
    """
    model.eval()
    
    predictions = {}
    patient_ids = []
    windows = [30, 60, 180, 365]
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            temporal_weights = batch['temporal_weights'].to(device)
            batch_patient_ids = batch['patient_id']
            
            # Forward pass
            outputs, _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                temporal_weights=temporal_weights
            )
            
            # Store predictions and patient IDs
            for i, patient_id in enumerate(batch_patient_ids):
                patient_ids.append(patient_id)
                predictions[patient_id] = outputs[i].cpu().numpy()
    
    # Stratify patients by risk
    stratified_patients = {}
    
    for i, window in enumerate(windows):
        stratified_patients[window] = {
            'low_risk': [],
            'medium_risk': [],
            'high_risk': []
        }
        
        for patient_id in patient_ids:
            risk_score = predictions[patient_id][i]
            
            if risk_score < thresholds['low']:
                stratified_patients[window]['low_risk'].append(patient_id)
            elif risk_score >= thresholds['high']:
                stratified_patients[window]['high_risk'].append(patient_id)
            else:
                stratified_patients[window]['medium_risk'].append(patient_id)
    
    # Print stratification summary
    print("Risk Stratification Summary:")
    for window in windows:
        total = len(patient_ids)
        low = len(stratified_patients[window]['low_risk'])
        medium = len(stratified_patients[window]['medium_risk'])
        high = len(stratified_patients[window]['high_risk'])
        
        print(f"  {window}-day window:")
        print(f"    Low risk: {low} patients ({low/total:.1%})")
        print(f"    Medium risk: {medium} patients ({medium/total:.1%})")
        print(f"    High risk: {high} patients ({high/total:.1%})")
    
    return stratified_patients, predictions

def predict_for_new_patient(model, tokenizer, patient_encounters, device):
    """
    Make predictions for a new patient
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        patient_encounters: List of patient encounter texts
        device: Device to use
        
    Returns:
        risk_scores: Risk scores for each window
    """
    model.eval()
    
    # Combine encounters with separators
    combined_text = " " + tokenizer.sep_token + " ".join(patient_encounters)
    
    # Tokenize
    encoding = tokenizer(
        combined_text,
        truncation=True,
        padding='max_length',
        max_length=512,
        return_tensors='pt'
    )
    
    # Create temporal weights (assume linear decay with most recent = 1.0)
    weights = np.linspace(0.5, 1.0, len(patient_encounters))
    temporal_weights = torch.tensor(weights, dtype=torch.float).unsqueeze(0).to(device)
    
    # Move inputs to device
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs, _ = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            temporal_weights=temporal_weights
        )
    
    # Get risk scores
    risk_scores = outputs[0].cpu().numpy()
    
    # Map to window days
    windows = [30, 60, 180, 365]
    window_scores = dict(zip(windows, risk_scores))
    
    return window_scores