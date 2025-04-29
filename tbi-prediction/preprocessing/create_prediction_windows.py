import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

def create_prediction_windows(df, window_days):
    """
    Create prediction windows for mental health diagnosis prediction
    """
    features = []
    labels = []
    patient_ids = []
    
    print(f"Creating {window_days}-day window labels...")
    
    for patient_id, patient_data in df.groupby('patient_id'):
        # Get all data up to TBI (days_from_tbi <= 0)
        pre_tbi_data = patient_data[patient_data['days_from_tbi'] <= 0]
        
        # Get data in prediction window
        window_data = patient_data[
            (patient_data['days_from_tbi'] > 0) & 
            (patient_data['days_from_tbi'] <= window_days)
        ]
        
        if len(pre_tbi_data) > 0:  # Only include if we have pre-TBI data
            # Aggregate pre-TBI features
            patient_features = aggregate_patient_features(pre_tbi_data)
            
            # Create label
            has_mh = window_data['has_mh_diagnosis'].any()
            
            features.append(patient_features)
            labels.append(has_mh)
            patient_ids.append(patient_id)
    
    print(f"Created features for {len(patient_ids)} patients")
    feature_df = pd.DataFrame(features)
    return feature_df, np.array(labels), np.array(patient_ids)

def aggregate_patient_features(patient_data):
    """
    Aggregate patient features from multiple encounters
    """
    # Get last values for static features
    static_features = {
        'age': patient_data['age'].iloc[-1],
        'gender': patient_data['gender'].iloc[-1],
        'race': patient_data['race'].iloc[-1],
        'sponservice': patient_data['sponservice'].iloc[-1],
        'sponrankgrp': patient_data['sponrankgrp'].iloc[-1]
    }
    
    # Aggregate dynamic features with time weights
    weighted_features = {}
    for col in patient_data.select_dtypes(include=[np.number]).columns:
        if col not in ['patient_id', 'days_from_tbi', 'time_weight'] and \
           col not in static_features:
            # Weighted mean
            weighted_features[f'{col}_wmean'] = np.average(
                patient_data[col], 
                weights=patient_data['time_weight']
            )
            # Also get max values for certain features
            if any(x in col for x in ['cost', 'diag', 'proc']):
                weighted_features[f'{col}_max'] = patient_data[col].max()
    
    # Combine all features
    return {**static_features, **weighted_features}

def prepare_model_data(df, window_days):
    """
    Prepare train/test splits for a specific prediction window
    """
    print(f"\nPreparing {window_days}-day prediction window...")
    
    # Create features and labels
    X, y, patient_ids = create_prediction_windows(df, window_days)
    
    # Reshape patient_ids for sklearn
    patient_ids_reshaped = patient_ids.reshape(-1, 1)
    
    # Split by patient
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, groups=patient_ids_reshaped))
    
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Positive cases in train: {y_train.mean():.2%}")
    print(f"Positive cases in test: {y_test.mean():.2%}")
    
    return X_train, X_test, y_train, y_test

def main():
    # Load cleaned data
    print("Loading cleaned data...")
    df = pd.read_csv('cleaned_temporal_data.csv')
    
    # Create prediction windows
    windows = [30, 60, 180, 365]
    model_data = {}
    
    for window in windows:
        X_train, X_test, y_train, y_test = prepare_model_data(df, window)
        model_data[window] = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }
        
        # Save window-specific data
        output_prefix = f'window_{window}d'
        X_train.to_csv(f'{output_prefix}_X_train.csv', index=False)
        X_test.to_csv(f'{output_prefix}_X_test.csv', index=False)
        np.save(f'{output_prefix}_y_train.npy', y_train)
        np.save(f'{output_prefix}_y_test.npy', y_test)
        
        print(f"\nSaved {window}-day window data")

if __name__ == "__main__":
    main()