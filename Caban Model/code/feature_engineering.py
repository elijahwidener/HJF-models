import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from datetime import timedelta

def create_rolling_features(df):
    """
    Create rolling window features for each patient
    """
    # Sort by patient and date
    df = df.sort_values(['patient_id', 'days_from_tbi'])
    
    # Define windows in days
    windows = [30, 60, 90]
    
    # Initialize features dictionary
    rolling_features = {}
    
    for window in windows:
        # Create window mask
        window_mask = (df['days_from_tbi'] >= -window) & (df['days_from_tbi'] <= 0)
        
        # Group by patient and calculate features
        window_stats = df[window_mask].groupby('patient_id').agg({
            'total_cost': ['mean', 'sum', 'count'],
            'clinical_salary_cost': ['mean', 'sum'],
            'pharmacy_cost': ['mean', 'sum'],
            'has_mh_diagnosis': 'sum',
            'severity': ['mean', 'max']
        })
        
        # Flatten column names
        window_stats.columns = [f'{col[0]}_{col[1]}_{window}d' 
                              for col in window_stats.columns]
        
        # Add to features dictionary
        rolling_features[window] = window_stats
    
    # Combine all windows
    rolling_df = pd.concat([df for df in rolling_features.values()], axis=1)
    
    return rolling_df

def create_enhanced_features(patient_data, time_weights):
    """Create even more sophisticated features from patient data with time weighting"""
    features = {}
    
    # Professional cost features (most important predictor)
    prof_costs = patient_data['professional_salary_cost']
    features.update({
        'prof_cost_max': prof_costs.max(),
        'prof_cost_recent_avg': (prof_costs * time_weights).tail(3).mean(),
        'prof_cost_growth': (prof_costs.diff() * time_weights).mean(),
        'prof_cost_volatility': prof_costs.std() / (prof_costs.mean() + 1),
        'prof_cost_trend': (prof_costs.iloc[-3:].mean() - prof_costs.iloc[:-3].mean()) / (prof_costs.iloc[:-3].mean() + 1)
    })
    
    # Enhanced sponsor rank interactions (second most important)
    features.update({
        'sponsor_prof_cost': patient_data['sponrankgrp'] * prof_costs.mean(),
        'sponsor_visit_rate': patient_data['sponrankgrp'] * len(patient_data),
        'sponsor_severity': patient_data['sponrankgrp'] * patient_data['severity'].mean()
    })
    
    # Transform study period features into meaningful predictors
    study_periods = ['90day', '180day', '270day', '365day']
    for period in study_periods:
        study_col = f'study{period}'
        encounter_col = f'study{period}_encounter'
        
        if study_col in patient_data.columns and encounter_col in patient_data.columns:
            # Create engagement metrics
            engagement = patient_data[encounter_col].sum() / (patient_data[study_col].sum() + 1)
            features[f'engagement_{period}'] = engagement
            
            # Interaction with professional costs
            features[f'prof_cost_per_encounter_{period}'] = prof_costs.sum() / (patient_data[encounter_col].sum() + 1)
            
            # Interaction with severity
            if 'severity' in patient_data.columns:
                features[f'severity_per_encounter_{period}'] = patient_data['severity'].mean() * engagement

    # Enhanced diagnosis features
    diag_cols = [col for col in patient_data.columns if col.startswith('diag') and col.endswith('_encoded')]
    for col in diag_cols:
        unique_diags = patient_data[col].nunique()
        features.update({
            f'{col}_unique': unique_diags,
            f'{col}_recent': patient_data[col].iloc[-1],
            f'{col}_repeats': len(patient_data[col]) - unique_diags,
            f'{col}_prof_cost_interaction': unique_diags * prof_costs.mean(),
            f'{col}_severity_interaction': unique_diags * patient_data['severity'].mean() if 'severity' in patient_data.columns else 0
        })
        
        # Add recency weighted diagnosis features
        recent_diags = (patient_data[col] * time_weights).sum() / time_weights.sum()
        features[f'{col}_recent_weighted'] = recent_diags
    
    # Visit patterns and intensity
    features.update({
        'visit_count': len(patient_data),
        'days_between_visits': patient_data['days_from_tbi'].diff().mean(),
        'visit_frequency': len(patient_data) / (abs(patient_data['days_from_tbi'].min()) + 1),
        'recent_visit_intensity': time_weights.sum() / len(time_weights),
        'visit_regularity': 1 / (patient_data['days_from_tbi'].diff().std() + 1)
    })
    
    # Cost ratios and trends for all cost types
    cost_cols = [col for col in patient_data.columns if 'cost' in col.lower()]
    for i, col1 in enumerate(cost_cols):
        for col2 in cost_cols[i+1:]:
            ratio_name = f'{col1}_to_{col2}_ratio'
            features[ratio_name] = patient_data[col1].sum() / (patient_data[col2].sum() + 1)
    
    # Treatment intensity and progression
    features['treatment_intensity'] = (
        patient_data.filter(like='_cost').sum(axis=1) * time_weights
    ).mean()
    
    # Severity progression if available
    if 'severity' in patient_data.columns:
        features.update({
            'severity_progression': (patient_data['severity'] * time_weights).diff().mean(),
            'high_severity_ratio': (patient_data['severity'] >= patient_data['severity'].median()).mean(),
            'severity_trend': (patient_data['severity'].iloc[-3:].mean() - 
                             patient_data['severity'].iloc[:-3].mean()) / 
                            (patient_data['severity'].iloc[:-3].mean() + 1)
        })
    
    return features

def create_interactions(df):
    """Create interaction features from aggregated data"""
    interactions = pd.DataFrame(index=df.index)
    
    # Get numerical columns
    num_cols = df.select_dtypes(include=[np.number]).columns
    
    # Cost ratios
    cost_cols = [col for col in num_cols if 'cost' in col.lower()]
    for i, col1 in enumerate(cost_cols):
        for col2 in cost_cols[i+1:]:
            ratio_name = f'{col1}_to_{col2}_ratio'
            interactions[ratio_name] = df[col1] / (df[col2] + 1)
    
    # Severity interactions
    if 'severity' in df.columns:
        severity_cols = ['visit_count', 'prof_cost_max', 'treatment_intensity']
        for col in severity_cols:
            if col in df.columns:
                interactions[f'severity_{col}_interaction'] = df['severity'] * df[col]
    
    # Diagnosis density
    diag_cols = [col for col in df.columns if 'diag' in col.lower()]
    if diag_cols:
        interactions['diagnosis_density'] = df[diag_cols].mean(axis=1)
        
    # Visit pattern interactions
    if 'visit_frequency' in df.columns:
        for col in ['prof_cost_max', 'severity', 'treatment_intensity']:
            if col in df.columns:
                interactions[f'visit_freq_{col}_interaction'] = df['visit_frequency'] * df[col]
    
    return interactions