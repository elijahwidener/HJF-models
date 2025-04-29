import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def get_important_features(patient_data, time_weights):
    """Create features focusing only on the most predictive ones"""
    feature_dict = {}
    
    # Professional salary cost features (most important predictor)
    prof_costs = patient_data['professional_salary_cost']
    feature_dict.update({
        'prof_cost_max': prof_costs.max(),
        'prof_cost_weighted_mean': (prof_costs * time_weights).mean(),
        'prof_cost_recent': prof_costs.iloc[-3:].mean(),  # Average of last 3 visits
        'prof_cost_growth': prof_costs.pct_change().mean(),
        'prof_cost_volatility': prof_costs.std() / (prof_costs.mean() + 1)
    })
    
    # Sponsor rank features (second most important)
    if 'sponrankgrp' in patient_data.columns:
        sponsor_rank = patient_data['sponrankgrp'].iloc[0]  # Should be constant per patient
        feature_dict.update({
            'sponrankgrp': sponsor_rank,
            'sponsor_cost_interaction': sponsor_rank * prof_costs.mean(),
            'sponsor_visit_interaction': sponsor_rank * len(patient_data)
        })
    
    # Diagnosis features (third most important)
    diag_cols = [col for col in patient_data.columns if col.startswith('diag') and col.endswith('_encoded')]
    for col in diag_cols[:4]:  # Focus on first 4 diagnoses as they're most important
        feature_dict.update({
            f'{col}_max': patient_data[col].max(),
            f'{col}_wmean': (patient_data[col] * time_weights).mean(),
            f'{col}_unique': patient_data[col].nunique(),
            f'{col}_recent': patient_data[col].iloc[-1],
            f'{col}_cost_interaction': patient_data[col].nunique() * prof_costs.mean()
        })
    
    # Procedure features (shown some importance)
    proc_cols = [col for col in patient_data.columns if col.startswith('proc')]
    for col in proc_cols[:2]:  # Focus on first 2 procedures
        feature_dict.update({
            f'{col}_max': patient_data[col].max(),
            f'{col}_recent': patient_data[col].iloc[-1]
        })
    
    # Product line features (shown some importance)
    if 'prodline' in patient_data.columns:
        feature_dict['prodline_wmean'] = (patient_data['prodline'] * time_weights).mean()
    
    # Visit patterns (derived from important features)
    feature_dict.update({
        'visit_count': len(patient_data),
        'visit_frequency': len(patient_data) / (abs(patient_data['days_from_tbi'].min()) + 1),
        'recent_visit_intensity': time_weights[-3:].sum() / 3
    })
    
    return feature_dict

def load_window_data(window_days):
    """Load and prepare data focusing on important features"""
    print(f"\nLoading data for {window_days}-day window...")
    
    try:
        # Try to load preprocessed data
        X_train = pd.read_csv(f'window_{window_days}d_X_train.csv')
        X_test = pd.read_csv(f'window_{window_days}d_X_test.csv')
        y_train = np.load(f'window_{window_days}d_y_train.npy')
        y_test = np.load(f'window_{window_days}d_y_test.npy')
        
    except FileNotFoundError:
        df = pd.read_csv('cleaned_temporal_data.csv')
        
        # Remove study period columns (they showed no importance)
        study_cols = [col for col in df.columns if 'study' in col]
        df = df.drop(columns=study_cols)
        
        # Split data into pre and post TBI
        pre_tbi = df[df['days_from_tbi'] <= 0]
        post_tbi = df[df['days_from_tbi'] > 0]
        
        features = []
        labels = []
        patient_ids = []
        
        for patient_id, patient_data in pre_tbi.groupby('patient_id'):
            if len(patient_data) == 0:
                continue
            
            # Calculate time weights with more emphasis on recent events
            days_from_tbi = abs(patient_data['days_from_tbi'])
            time_weights = np.exp(-0.01 * days_from_tbi)  # Exponential decay
            
            # Get important features
            feature_dict = get_important_features(patient_data, time_weights)
            feature_dict['patient_id'] = patient_id
            
            # Get label
            patient_post = post_tbi[
                (post_tbi['patient_id'] == patient_id) & 
                (post_tbi['days_from_tbi'] <= window_days)
            ]
            has_mh = patient_post['has_mh_diagnosis'].any()
            
            features.append(feature_dict)
            labels.append(has_mh)
            patient_ids.append(patient_id)
        
        # Convert to DataFrame
        X = pd.DataFrame(features)
        y = np.array(labels)
        
        # Split the data
        unique_patients = np.unique(patient_ids)
        np.random.shuffle(unique_patients)
        split_idx = int(len(unique_patients) * 0.8)
        
        train_patients = unique_patients[:split_idx]
        test_patients = unique_patients[split_idx:]
        
        X_train = X[X['patient_id'].isin(train_patients)].drop('patient_id', axis=1)
        X_test = X[X['patient_id'].isin(test_patients)].drop('patient_id', axis=1)
        y_train = y[X['patient_id'].isin(train_patients)]
        y_test = y[X['patient_id'].isin(test_patients)]
        
        # Save preprocessed data
        X_train.to_csv(f'window_{window_days}d_X_train.csv', index=False)
        X_test.to_csv(f'window_{window_days}d_X_test.csv', index=False)
        np.save(f'window_{window_days}d_y_train.npy', y_train)
        np.save(f'window_{window_days}d_y_test.npy', y_test)
    
    return X_train, X_test, y_train, y_test

def train_xgboost(X_train, X_test, y_train, y_test, window_days):
    """Train XGBoost with optimized parameters"""
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Calculate class weights
    scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    
    # Optimized XGBoost parameters
    xgb_params = {
        'n_estimators': 1000,  # Increased for better convergence
        'max_depth': 6,
        'learning_rate': 0.01,  # Slower learning rate
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'gamma': 0.2,
        'scale_pos_weight': scale_pos_weight,
        'reg_alpha': 0.1,  # L1 regularization
        'reg_lambda': 1.0,  # L2 regularization
        'random_state': 42,
        'tree_method': 'hist'  # Faster histogram-based algorithm
    }
    
    # Initialize model
    model = xgb.XGBClassifier(**xgb_params)
    
    # Train with early stopping
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)],
        verbose=True
    )
    
    # Make predictions
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Calculate metrics
    results = {
        'auc_roc': roc_auc_score(y_test, y_pred_proba),
        'auc_pr': average_precision_score(y_test, y_pred_proba),
        'classification_report': classification_report(y_test, y_pred),
        'feature_importance': pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
    }
    
    print(f"\nResults for {window_days}-day window:")
    print(f"ROC AUC: {results['auc_roc']:.3f}")
    print(f"PR AUC: {results['auc_pr']:.3f}")
    print("\nClassification Report:")
    print(results['classification_report'])
    
    return model, results

def plot_feature_importance(model, feature_names, window_days):
    """Plot feature importance with better visualization"""
    importance = model.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    })
    
    # Sort by importance and filter out near-zero importance
    feature_importance = feature_importance[feature_importance['importance'] > 0.001]
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    # Plot important features
    plt.figure(figsize=(12, 8))
    sns.barplot(data=feature_importance, x='importance', y='feature')
    plt.title(f'Important Features ({window_days}-day window)')
    plt.xlabel('Feature Importance')
    plt.tight_layout()
    plt.savefig(f'xgboost_importance_{window_days}d.png')
    plt.close()
    
    return feature_importance

def main():
    windows = [30, 60, 180, 365]
    results_summary = {}
    
    for window in windows:
        # Load data
        X_train, X_test, y_train, y_test = load_window_data(window)
        
        # Train and evaluate
        model, results = train_xgboost(
            X_train, X_test, y_train, y_test, window
        )
        
        # Plot feature importance
        feature_importance = plot_feature_importance(model, X_train.columns, window)
        
        # Store results
        results_summary[window] = {
            'metrics': results,
            'feature_importance': feature_importance
        }
    
    # Save summary results
    pd.DataFrame(results_summary).to_csv('xgboost_results_summary.csv')

if __name__ == "__main__":
    main()