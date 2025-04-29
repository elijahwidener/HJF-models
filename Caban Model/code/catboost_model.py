import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def load_window_data(window_days):
    """Load data with minimal preprocessing since CatBoost handles categories"""
    print(f"\nLoading data for {window_days}-day window...")
    
    try:
        # Try to load preprocessed data
        X_train = pd.read_csv(f'catboost_{window_days}d_X_train.csv')
        X_test = pd.read_csv(f'catboost_{window_days}d_X_test.csv')
        y_train = np.load(f'catboost_{window_days}d_y_train.npy')
        y_test = np.load(f'catboost_{window_days}d_y_test.npy')
        
    except FileNotFoundError:
        # Load raw data
        df = pd.read_csv('cleaned_temporal_data.csv')
        
        # Split into pre and post TBI
        pre_tbi = df[df['days_from_tbi'] <= 0]
        post_tbi = df[df['days_from_tbi'] > 0]
        
        features = []
        labels = []
        patient_ids = []
        
        # Process each patient
        for patient_id, patient_data in pre_tbi.groupby('patient_id'):
            if len(patient_data) == 0:
                continue
            
            # Calculate time weights
            days_from_tbi = abs(patient_data['days_from_tbi'])
            time_weights = np.exp(-0.01 * days_from_tbi)
            
            # Basic features
            feature_dict = {
                'patient_id': patient_id,
                'visit_count': len(patient_data),
                'days_in_system': patient_data['days_from_tbi'].max() - patient_data['days_from_tbi'].min(),
                'last_severity': patient_data['severity'].iloc[-1],
                'max_severity': patient_data['severity'].max(),
                'recent_visits': sum(time_weights[-3:]),
                'age': patient_data['age'].iloc[0],
                'gender': patient_data['gender'].iloc[0],
                'race': patient_data['race'].iloc[0],
                'sponrankgrp': patient_data['sponrankgrp'].iloc[0],
                'sponservice': patient_data['sponservice'].iloc[0]
            }
            
            # Cost features
            cost_cols = [col for col in patient_data.columns if 'cost' in col.lower()]
            for col in cost_cols:
                weighted_costs = patient_data[col] * time_weights
                feature_dict.update({
                    f'{col}_recent': patient_data[col].iloc[-3:].mean(),
                    f'{col}_max': patient_data[col].max(),
                    f'{col}_total': patient_data[col].sum(),
                    f'{col}_weighted': weighted_costs.sum() / time_weights.sum()
                })
            
            # Diagnosis features
            diag_cols = [col for col in patient_data.columns if col.startswith('diag') and col.endswith('_encoded')]
            for col in diag_cols:
                feature_dict.update({
                    f'{col}_recent': patient_data[col].iloc[-1],
                    f'{col}_unique': patient_data[col].nunique(),
                    f'{col}_total': len(patient_data[col])
                })
            
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
        
        # Split data
        unique_patients = np.unique(patient_ids)
        np.random.shuffle(unique_patients)
        split_idx = int(len(unique_patients) * 0.8)
        
        train_patients = unique_patients[:split_idx]
        test_patients = unique_patients[split_idx:]
        
        # Create train/test sets
        X_train = X[X['patient_id'].isin(train_patients)]
        X_test = X[X['patient_id'].isin(test_patients)]
        y_train = y[X['patient_id'].isin(train_patients)]
        y_test = y[X['patient_id'].isin(test_patients)]
        
        # Save processed data
        X_train.to_csv(f'catboost_{window_days}d_X_train.csv', index=False)
        X_test.to_csv(f'catboost_{window_days}d_X_test.csv', index=False)
        np.save(f'catboost_{window_days}d_y_train.npy', y_train)
        np.save(f'catboost_{window_days}d_y_test.npy', y_test)
    
    return X_train, X_test, y_train, y_test

def train_catboost(X_train, X_test, y_train, y_test, window_days):
    """Train CatBoost model with temporal awareness"""
    print(f"\nTraining model for {window_days}-day window...")
    
    # Identify categorical columns
    cat_features = ['gender', 'race', 'sponrankgrp', 'sponservice']
    
    # Create CatBoost pools
    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    test_pool = Pool(X_test, y_test, cat_features=cat_features)
    
    # Calculate class weights
    class_weights = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    
    # Initialize model
    model = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.02,
        depth=6,
        l2_leaf_reg=3,
        loss_function='Logloss',
        eval_metric='AUC',
        metric_period=50,
        od_type='Iter',
        od_wait=50,
        random_seed=42,
        class_weights=[1, class_weights],
        verbose=50  # Print metrics every 50 iterations
    )
    
    # Train model
    model.fit(
        train_pool,
        eval_set=test_pool,
        use_best_model=True,
        plot=False  # Disable plotting
    )
    
    # Make predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
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
    """Plot feature importance"""
    importance = model.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    })
    
    # Sort and filter important features
    feature_importance = feature_importance[feature_importance['importance'] > 0]
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    # Plot
    plt.figure(figsize=(12, 8))
    sns.barplot(data=feature_importance.head(20), x='importance', y='feature')
    plt.title(f'Top 20 Important Features ({window_days}-day window)')
    plt.xlabel('Feature Importance')
    plt.tight_layout()
    plt.savefig(f'catboost_importance_{window_days}d.png')
    plt.close()
    
    return feature_importance

def main():
    windows = [30, 60, 180, 365]
    results_summary = {}
    
    for window in windows:
        # Load data
        X_train, X_test, y_train, y_test = load_window_data(window)
        
        # Train and evaluate
        model, results = train_catboost(
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
    pd.DataFrame(results_summary).to_csv('catboost_results_summary.csv')

if __name__ == "__main__":
    main()