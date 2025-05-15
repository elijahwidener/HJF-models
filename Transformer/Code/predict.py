import torch
import numpy as np
from config import DEVICE

def risk_stratify_patients(model, dataloader, device, thresholds={'low': 0.3, 'high': 0.7}):
    """
    Stratify patients into risk categories based on model predictions.
    
    This function categorizes patients into three risk tiers:
    - Low risk: Prediction score < low threshold (default 0.3)
    - Medium risk: Score between low and high thresholds
    - High risk: Prediction score >= high threshold (default 0.7)
    
    Risk stratification is performed for each prediction window (30, 60, 180, 365 days),
    allowing for time-specific risk assessment and potential intervention planning.
    
    Args:
        model (MultiTaskTBIPredictor): The trained model
        dataloader (DataLoader): DataLoader containing patient data
        device (torch.device): Device to use for computation
        thresholds (dict): Dictionary with 'low' and 'high' threshold values
        
    Returns:
        tuple: (stratified_patients, all_preds)
            - stratified_patients: Dictionary mapping windows to risk category groupings
            - all_preds: Raw prediction scores
    """
    # Use the window ensembling function instead of direct model calls
    all_preds, all_labels = predict_with_window_ensembling(model, dataloader, device)
    
    # Get patient IDs from dataloader
    patient_ids = []
    for batch in dataloader:
        patient_ids.extend(batch['patient_id'])
    
    # Process stratification based on predictions
    windows = ['30_day', '60_day', '180_day', '365_day']
    stratified_patients = {}
    
    for window in windows:
        stratified_patients[window] = {
            'low_risk': [],
            'medium_risk': [],
            'high_risk': []
        }
        
        for i, patient_id in enumerate(patient_ids):
            risk_score = all_preds[window][i]
            
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
        
        print(f"  {window} window:")
        print(f"    Low risk: {low} patients ({low/total:.1%})")
        print(f"    Medium risk: {medium} patients ({medium/total:.1%})")
        print(f"    High risk: {high} patients ({high/total:.1%})")
    
    return stratified_patients, all_preds

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

def predict_with_window_ensembling(model, dataloader, device):
    """
    Make predictions with a sequential ensembling approach where predictions for earlier time
    windows are used to inform predictions for later time windows.
    
    This function implements a cascading prediction approach:
    1. First predicts the 30-day window outcome
    2. Uses the 30-day prediction to help predict the 60-day window
    3. Uses both 30 and 60-day predictions to predict the 180-day window
    4. Uses all previous predictions to predict the 365-day window
    
    This approach leverages the intuition that mental health outcomes are temporally 
    correlated - a positive prediction in an earlier window may increase the likelihood
    of a positive prediction in later windows.
    
    Args:
        model (MultiTaskTBIPredictor): The trained model
        dataloader (DataLoader): DataLoader containing the evaluation data
        device (torch.device): Device to use for computation
        
    Returns:
        tuple: (all_preds, all_labels)
            - all_preds: Dictionary mapping window names to prediction arrays
            - all_labels: Dictionary mapping window names to label arrays
    """
    model.eval()
    
    all_preds = {
        '30_day': [],
        '60_day': [],
        '180_day': [],
        '365_day': []
    }
    all_labels = {
        '30_day': [],
        '60_day': [],
        '180_day': [],
        '365_day': []
    }
    
    windows = ['30_day', '60_day', '180_day', '365_day']
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            temporal_weights = batch['temporal_weights'].to(device) if 'temporal_weights' in batch else None
            
            # First predict 30-day window (no previous predictions)
            outputs, _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                temporal_weights=temporal_weights,
                prev_window_preds=None
            )
            
            preds_30day = outputs[:, 0].cpu().numpy()
            
            # Then predict 60-day window using 30-day predictions
            prev_preds_60day = outputs[:, 0].unsqueeze(1)  # 30-day predictions
            outputs_60day, _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                temporal_weights=temporal_weights,
                prev_window_preds=prev_preds_60day
            )
            
            preds_60day = outputs_60day[:, 1].cpu().numpy()
            
            # Predict 180-day window using 30 and 60-day predictions
            prev_preds_180day = outputs_60day[:, :2]  # 30 and 60-day predictions
            outputs_180day, _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                temporal_weights=temporal_weights,
                prev_window_preds=prev_preds_180day
            )
            
            preds_180day = outputs_180day[:, 2].cpu().numpy()
            
            # Predict 365-day window using all previous predictions
            prev_preds_365day = outputs_180day[:, :3]  # 30, 60, and 180-day predictions
            outputs_365day, _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                temporal_weights=temporal_weights,
                prev_window_preds=prev_preds_365day
            )
            
            preds_365day = outputs_365day[:, 3].cpu().numpy()
            
            # Store predictions and labels
            all_preds['30_day'].append(preds_30day)
            all_preds['60_day'].append(preds_60day)
            all_preds['180_day'].append(preds_180day)
            all_preds['365_day'].append(preds_365day)
            
            all_labels['30_day'].append(labels[:, 0].cpu().numpy())
            all_labels['60_day'].append(labels[:, 1].cpu().numpy())
            all_labels['180_day'].append(labels[:, 2].cpu().numpy())
            all_labels['365_day'].append(labels[:, 3].cpu().numpy())
    
    # Concatenate predictions and labels for each window
    for window in windows:
        all_preds[window] = np.concatenate(all_preds[window])
        all_labels[window] = np.concatenate(all_labels[window])
    
    return all_preds, all_labels