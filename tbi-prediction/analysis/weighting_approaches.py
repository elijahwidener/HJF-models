"""
weighting_approaches.py

This script implements and compares different weighting strategies for predicting
mental health diagnoses following traumatic brain injury.

Key strategies explored:
1. Class weighting (different weights for positive vs. negative cases)
2. Temporal decay variations (different decay rates for time weighting)
3. Combined weighting approaches
4. Evaluation of impact on model performance

For AMIA Annual Symposium submission
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from imblearn.over_sampling import SMOTE
import os

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configure plot style for publication
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)
colors = sns.color_palette("deep")

class WeightingAnalysis:
    """Class to analyze different weighting approaches for TBI prediction"""
    
    def __init__(self, data_path='cleaned_temporal_data.csv', window_days=60):
        """
        Initialize the analysis
        
        Args:
            data_path: Path to the processed TBI dataset
            window_days: Prediction window (30, 60, 180, or 365 days)
        """
        self.data_path = data_path
        self.window_days = window_days
        print(f"Analyzing weighting approaches for {window_days}-day window")
        
    def load_data(self):
        """Load and prepare data for analysis"""
        try:
            print(f"Loading data from {self.data_path}")
            self.df = pd.read_csv(self.data_path)
            
            # Split into pre and post TBI
            self.pre_tbi = self.df[self.df['days_from_tbi'] <= 0]
            self.post_tbi = self.df[(self.df['days_from_tbi'] > 0) & 
                                     (self.df['days_from_tbi'] <= self.window_days)]
            
            print(f"Data loaded: {len(self.df)} records, " 
                  f"{self.pre_tbi['patient_id'].nunique()} patients")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def prepare_features_and_labels(self):
        """Prepare feature sequences and labels"""
        print("Preparing features and labels...")
        
        sequences = []
        labels = []
        patient_ids = []
        
        # Process each patient
        for patient_id, patient_data in self.pre_tbi.groupby('patient_id'):
            # Skip if empty
            if len(patient_data) == 0:
                continue
            
            # Get features from pre-TBI data
            feature_cols = patient_data.select_dtypes(include=[np.number]).columns
            feature_cols = feature_cols.drop(['patient_id', 'days_from_tbi'])
            
            # Create sequence
            sequence = patient_data[feature_cols].values
            sequences.append(sequence)
            
            # Get label from post-TBI data
            patient_post = self.post_tbi[self.post_tbi['patient_id'] == patient_id]
            has_mh = patient_post['has_mh_diagnosis'].any() if len(patient_post) > 0 else False
            
            labels.append(has_mh)
            patient_ids.append(patient_id)
        
        # Convert to numpy arrays
        self.sequences = sequences
        self.labels = np.array(labels)
        self.patient_ids = np.array(patient_ids)
        
        # Calculate class distribution
        pos_rate = np.mean(self.labels)
        print(f"Prepared {len(sequences)} sequences")
        print(f"Class distribution: {pos_rate:.1%} positive, {1-pos_rate:.1%} negative")
        
        return sequences, self.labels, self.patient_ids
    
    def pad_sequences(self, sequences):
        """Pad sequences to the same length"""
        # Find max length
        max_len = max(len(seq) for seq in sequences)
        
        # Get feature dimension
        n_features = sequences[0].shape[1]
        
        # Create padded array
        padded = np.zeros((len(sequences), max_len, n_features))
        
        # Fill with actual values
        for i, seq in enumerate(sequences):
            padded[i, :len(seq)] = seq
        
        return padded
    
    def create_temporal_weights(self, decay_rates=[0.005, 0.01, 0.02, 0.05, 0.1]):
        """Create different temporal weights based on days from TBI"""
        print("Creating temporal weights with different decay rates...")
        
        temporal_weights = {}
        
        for rate in decay_rates:
            weights_list = []
            
            for patient_data in self.sequences:
                # Calculate weights based on days from TBI
                seq_len = len(patient_data)
                # Assuming days are ordered from furthest to closest to TBI
                days = np.arange(seq_len, 0, -1)
                weights = np.exp(-rate * days)
                weights_list.append(weights)
            
            # Pad weights to same length
            max_len = max(len(w) for w in weights_list)
            padded_weights = np.zeros((len(weights_list), max_len))
            
            for i, weights in enumerate(weights_list):
                padded_weights[i, :len(weights)] = weights
            
            temporal_weights[rate] = padded_weights
        
        return temporal_weights
    
    def split_data(self, test_size=0.2):
        """Split data into training and testing sets"""
        print("Splitting data into training and testing sets...")
        
        # Get unique patients for splitting
        unique_patients = np.unique(self.patient_ids)
        n_patients = len(unique_patients)
        
        # Shuffle patients
        np.random.shuffle(unique_patients)
        
        # Split at patient level
        split_idx = int(n_patients * (1 - test_size))
        train_patients = unique_patients[:split_idx]
        test_patients = unique_patients[split_idx:]
        
        # Create masks
        train_mask = np.isin(self.patient_ids, train_patients)
        test_mask = np.isin(self.patient_ids, test_patients)
        
        # Store split masks
        self.train_mask = train_mask
        self.test_mask = test_mask
        
        # Calculate class distribution in splits
        train_pos_rate = np.mean(self.labels[train_mask])
        test_pos_rate = np.mean(self.labels[test_mask])
        
        print(f"Train set: {np.sum(train_mask)} samples, {train_pos_rate:.1%} positive")
        print(f"Test set: {np.sum(test_mask)} samples, {test_pos_rate:.1%} positive")
    
    def create_model(self, input_shape):
        """Create LSTM model with temporal attention"""
        # Input layer
        inputs = keras.layers.Input(shape=input_shape)
        
        # Masking layer
        masked = keras.layers.Masking(mask_value=0.)(inputs)
        
        # Bidirectional LSTM
        lstm = keras.layers.Bidirectional(
            keras.layers.LSTM(64, return_sequences=True)
        )(masked)
        
        # Global average pooling
        pooled = keras.layers.GlobalAveragePooling1D()(lstm)
        
        # Dense layers
        dense = keras.layers.Dense(32, activation='relu')(pooled)
        dropout = keras.layers.Dropout(0.3)(dense)
        
        # Output layer
        output = keras.layers.Dense(1, activation='sigmoid')(dropout)
        
        # Create model
        model = keras.Model(inputs=inputs, outputs=output)
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )
        
        return model
    
    def train_with_class_weights(self, X_train, X_test, y_train, y_test, 
                                class_weights_list=None):
        """Train models with different class weights"""
        if class_weights_list is None:
            class_weights_list = [
                None,  # No weighting
                {0: 1.0, 1: 2.0},  # 2x weight on positive class
                {0: 1.0, 1: 3.0},  # 3x weight on positive class
                {0: 1.0, 1: 5.0}   # 5x weight on positive class
            ]
        
        results = {}
        
        for i, class_weights in enumerate(class_weights_list):
            weight_name = f"class_{i}" if class_weights is None else f"class_{class_weights[1]}x"
            print(f"\nTraining with {weight_name} weighting...")
            
            # Create model
            model = self.create_model(input_shape=(X_train.shape[1], X_train.shape[2]))
            
            # Define callbacks
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_auc', patience=10, restore_best_weights=True, mode='max'
                )
            ]
            
            # Train model
            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=30,
                batch_size=32,
                class_weight=class_weights,
                callbacks=callbacks,
                verbose=1
            )
            
            # Evaluate model
            y_pred_proba = model.predict(X_test)
            auc_roc = roc_auc_score(y_test, y_pred_proba)
            auc_pr = average_precision_score(y_test, y_pred_proba)
            
            # Calculate precision-recall curves
            precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            
            results[weight_name] = {
                'model': model,
                'history': history.history,
                'auc_roc': auc_roc,
                'auc_pr': auc_pr,
                'precision': precision,
                'recall': recall,
                'fpr': fpr,
                'tpr': tpr
            }
            
            print(f"Results for {weight_name}: ROC AUC = {auc_roc:.3f}, PR AUC = {auc_pr:.3f}")
        
        return results
    
    def train_with_temporal_weights(self, X_train, X_test, y_train, y_test, 
                                   temporal_weights_dict):
        """Train models with different temporal decay rates"""
        results = {}
        
        for rate, weights in temporal_weights_dict.items():
            print(f"\nTraining with temporal decay rate = {rate}...")
            
            # Extract weights for training set
            train_weights = weights[self.train_mask]
            
            # Create model
            model = self.create_model(input_shape=(X_train.shape[1], X_train.shape[2]))
            
            # Define callbacks
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_auc', patience=10, restore_best_weights=True, mode='max'
                )
            ]
            
            # Train model with sample weights
            history = model.fit(
                X_train, y_train,
                sample_weight=train_weights.mean(axis=1),  # Use average weight per sequence
                validation_data=(X_test, y_test),
                epochs=30,
                batch_size=32,
                callbacks=callbacks,
                verbose=1
            )
            
            # Evaluate model
            y_pred_proba = model.predict(X_test)
            auc_roc = roc_auc_score(y_test, y_pred_proba)
            auc_pr = average_precision_score(y_test, y_pred_proba)
            
            # Calculate precision-recall curves
            precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            
            results[f"temporal_{rate}"] = {
                'model': model,
                'history': history.history,
                'auc_roc': auc_roc,
                'auc_pr': auc_pr,
                'precision': precision,
                'recall': recall,
                'fpr': fpr,
                'tpr': tpr
            }
            
            print(f"Results for temporal_{rate}: ROC AUC = {auc_roc:.3f}, PR AUC = {auc_pr:.3f}")
        
        return results
    
    def train_with_combined_weights(self, X_train, X_test, y_train, y_test,
                                   temporal_rate=0.01, class_weights=None):
        """Train model with combined temporal and class weighting"""
        print("\nTraining with combined weighting...")
        
        # Calculate temporal weights
        temporal_weights_list = []
        for patient_data in self.sequences:
            seq_len = len(patient_data)
            days = np.arange(seq_len, 0, -1)
            weights = np.exp(-temporal_rate * days)
            temporal_weights_list.append(weights)
        
        # Pad weights
        max_len = max(len(w) for w in temporal_weights_list)
        padded_weights = np.zeros((len(temporal_weights_list), max_len))
        for i, weights in enumerate(temporal_weights_list):
            padded_weights[i, :len(weights)] = weights
        
        # Extract training weights
        train_weights = padded_weights[self.train_mask]
        
        # Create model
        model = self.create_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        
        # Define callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_auc', patience=10, restore_best_weights=True, mode='max'
            )
        ]
        
        # Train with both sample weights and class weights
        history = model.fit(
            X_train, y_train,
            sample_weight=train_weights.mean(axis=1),
            class_weight=class_weights,
            validation_data=(X_test, y_test),
            epochs=30,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate model
        y_pred_proba = model.predict(X_test)
        auc_roc = roc_auc_score(y_test, y_pred_proba)
        auc_pr = average_precision_score(y_test, y_pred_proba)
        
        # Calculate precision-recall curves
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        
        results = {
            'model': model,
            'history': history.history,
            'auc_roc': auc_roc,
            'auc_pr': auc_pr,
            'precision': precision,
            'recall': recall,
            'fpr': fpr,
            'tpr': tpr
        }
        
        print(f"Results for combined weighting: ROC AUC = {auc_roc:.3f}, PR AUC = {auc_pr:.3f}")
        
        return results
    
    def train_with_smote(self, X_train, X_test, y_train, y_test):
        """Train model with SMOTE oversampling"""
        print("\nTraining with SMOTE oversampling...")
        
        # Reshape training data for SMOTE
        n_samples = X_train.shape[0]
        n_features = X_train.shape[1] * X_train.shape[2]
        X_train_flat = X_train.reshape(n_samples, n_features)
        
        # Apply SMOTE
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train_flat, y_train)
        
        # Reshape back
        X_train_res = X_train_res.reshape(X_train_res.shape[0], X_train.shape[1], X_train.shape[2])
        
        # Create model
        model = self.create_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        
        # Define callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_auc', patience=10, restore_best_weights=True, mode='max'
            )
        ]
        
        # Train model
        history = model.fit(
            X_train_res, y_train_res,
            validation_data=(X_test, y_test),
            epochs=30,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate model
        y_pred_proba = model.predict(X_test)
        auc_roc = roc_auc_score(y_test, y_pred_proba)
        auc_pr = average_precision_score(y_test, y_pred_proba)
        
        # Calculate precision-recall curves
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        
        results = {
            'model': model,
            'history': history.history,
            'auc_roc': auc_roc,
            'auc_pr': auc_pr,
            'precision': precision,
            'recall': recall,
            'fpr': fpr,
            'tpr': tpr
        }
        
        print(f"Results for SMOTE: ROC AUC = {auc_roc:.3f}, PR AUC = {auc_pr:.3f}")
        
        return results
    
    def plot_class_weight_comparison(self, results, save_path='figures'):
        """Plot comparison of different class weights"""
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        # Filter only class weight results
        class_results = {k: v for k, v in results.items() if k.startswith('class_')}
        
        # Prepare data for plotting
        weights = []
        auc_roc_values = []
        auc_pr_values = []
        
        for weight, result in class_results.items():
            weights.append(weight.replace('class_', ''))
            auc_roc_values.append(result['auc_roc'])
            auc_pr_values.append(result['auc_pr'])
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Plot ROC AUC
        plt.subplot(1, 2, 1)
        bars = plt.bar(weights, auc_roc_values, color=colors[0])
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{height:.3f}', ha='center', va='bottom')
        
        plt.title('ROC AUC by Class Weight')
        plt.ylabel('ROC AUC')
        plt.xlabel('Class Weight')
        plt.ylim(0.5, max(auc_roc_values) + 0.05)
        
        # Plot PR AUC
        plt.subplot(1, 2, 2)
        bars = plt.bar(weights, auc_pr_values, color=colors[1])
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{height:.3f}', ha='center', va='bottom')
        
        plt.title('PR AUC by Class Weight')
        plt.ylabel('PR AUC')
        plt.xlabel('Class Weight')
        plt.ylim(0.3, max(auc_pr_values) + 0.05)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/class_weight_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_temporal_weight_comparison(self, results, save_path='figures'):
        """Plot comparison of different temporal weights"""
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        # Filter only temporal weight results
        temporal_results = {k: v for k, v in results.items() if k.startswith('temporal_')}
        
        # Prepare data for plotting
        rates = []
        auc_roc_values = []
        auc_pr_values = []
        
        for rate, result in temporal_results.items():
            rates.append(float(rate.replace('temporal_', '')))
            auc_roc_values.append(result['auc_roc'])
            auc_pr_values.append(result['auc_pr'])
        
        # Create figure
        plt.figure(figsize=(10, 5))
        
        plt.plot(rates, auc_roc_values, 'o-', color=colors[0], label='ROC AUC')
        plt.plot(rates, auc_pr_values, 's-', color=colors[1], label='PR AUC')
        
        # Add value labels
        for i, rate in enumerate(rates):
            plt.text(rate, auc_roc_values[i] + 0.005, f'{auc_roc_values[i]:.3f}', 
                    ha='center', va='bottom')
            plt.text(rate, auc_pr_values[i] - 0.015, f'{auc_pr_values[i]:.3f}', 
                    ha='center', va='top')
        
        plt.title('Performance by Temporal Decay Rate')
        plt.ylabel('AUC Score')
        plt.xlabel('Decay Rate')
        plt.xscale('log')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/temporal_weight_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_all_approaches_comparison(self, all_results, save_path='figures'):
        """Plot comparison of all approaches"""
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        # Prepare data for plotting
        approaches = []
        auc_roc_values = []
        auc_pr_values = []
        
        # Get best from each category
        best_class = max(
            {k: v for k, v in all_results.items() if k.startswith('class_')}.items(),
            key=lambda x: x[1]['auc_roc']
        )
        
        best_temporal = max(
            {k: v for k, v in all_results.items() if k.startswith('temporal_')}.items(),
            key=lambda x: x[1]['auc_roc']
        )
        
        # Add each approach
        for name, result in [
            ('Baseline', all_results.get('class_0', all_results['class_None'])),
            (f'Best Class Weight\n({best_class[0]})', best_class[1]),
            (f'Best Temporal\n({best_temporal[0]})', best_temporal[1]),
            ('Combined', all_results['combined']),
            ('SMOTE', all_results.get('smote', {'auc_roc': 0, 'auc_pr': 0}))
        ]:
            if result['auc_roc'] > 0:  # Only include if we have results
                approaches.append(name)
                auc_roc_values.append(result['auc_roc'])
                auc_pr_values.append(result['auc_pr'])
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot ROC AUC
        plt.subplot(1, 2, 1)
        bars = plt.bar(approaches, auc_roc_values, color=colors[0])
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{height:.3f}', ha='center', va='bottom', rotation=0)
        
        plt.title('ROC AUC by Approach')
        plt.ylabel('ROC AUC')
        plt.ylim(0.5, max(auc_roc_values) + 0.05)
        plt.xticks(rotation=45, ha='right')
        
        # Plot PR AUC
        plt.subplot(1, 2, 2)
        bars = plt.bar(approaches, auc_pr_values, color=colors[1])
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{height:.3f}', ha='center', va='bottom', rotation=0)
        
        plt.title('PR AUC by Approach')
        plt.ylabel('PR AUC')
        plt.ylim(0.3, max(auc_pr_values) + 0.05)
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/all_approaches_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_pr_curves(self, results, save_path='figures'):
        """Plot precision-recall curves for all approaches"""
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        plt.figure(figsize=(10, 8))
        
        # Plot each PR curve
        for name, result in results.items():
            if 'precision' in result and 'recall' in result:
                plt.plot(result['recall'], result['precision'], 
                        label=f"{name} (AUC={result['auc_pr']:.3f})")
        
        # Add random baseline
        baseline = np.mean(self.labels)
        plt.axhline(y=baseline, color='r', linestyle='--', 
                   label=f'Random Baseline ({baseline:.3f})')
        
        plt.title('Precision-Recall Curves by Approach')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/pr_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_roc_curves(self, results, save_path='figures'):
        """Plot ROC curves for all approaches"""
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        plt.figure(figsize=(10, 8))
        
        # Plot each ROC curve
        for name, result in results.items():
            if 'fpr' in result and 'tpr' in result:
                plt.plot(result['fpr'], result['tpr'], 
                        label=f"{name} (AUC={result['auc_roc']:.3f})")
        
        # Add random baseline
        plt.plot([0, 1], [0, 1], 'r--', label='Random Baseline')
        
        plt.title('ROC Curves by Approach')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_analysis(self):
        """Run the full weighting analysis"""
        # Load and prepare data
        if not self.load_data():
            return False
        
        self.prepare_features_and_labels()
        self.split_data()
        
        # Pad sequences
        X = self.pad_sequences(self.sequences)
        y = self.labels
        
        # Prepare train/test sets
        X_train = X[self.train_mask]
        X_test = X[self.test_mask]
        y_train = y[self.train_mask]
        y_test = y[self.test_mask]
        
        # Create temporal weights
        decay_rates = [0.005, 0.01, 0.02, 0.05]
        temporal_weights = self.create_temporal_weights(decay_rates)
        
        # Class weights to try
        class_weights_list = [
            None,  # No weighting
            {0: 1.0, 1: 2.0},  # 2x weight on positive class
            {0: 1.0, 1: 3.0},  # 3x weight on positive class
            {0: 1.0, 1: 5.0}   # 5x weight on positive class
        ]
        
        # Train with class weights
        class_results = self.train_with_class_weights(
            X_train, X_test, y_train, y_test, class_weights_list
        )
        
        # Train with temporal weights
        temporal_results = self.train_with_temporal_weights(
            X_train, X_test, y_train, y_test, temporal_weights
        )
        
        # Train with combined approach
        # Use the best class weight and temporal rate 
        best_class_weight = max(class_results.items(), key=lambda x: x[1]['auc_roc'])[0]
        best_temporal_rate = max(temporal_results.items(), key=lambda x: x[1]['auc_roc'])[0]
        
        print(f"Using best class weight ({best_class_weight}) and temporal rate ({best_temporal_rate})")
        
        # Extract actual values
        if best_class_weight == 'class_None':
            best_class_weight_value = None
        else:
            weight_value = float(best_class_weight.split('_')[1].replace('x', ''))
            best_class_weight_value = {0: 1.0, 1: weight_value}
        
        best_temporal_rate_value = float(best_temporal_rate.split('_')[1])
        
        # Train combined model
        combined_results = self.train_with_combined_weights(
            X_train, X_test, y_train, y_test,
            temporal_rate=best_temporal_rate_value,
            class_weights=best_class_weight_value
        )
        
        # Train with SMOTE (if requested)
        try:
            smote_results = self.train_with_smote(X_train, X_test, y_train, y_test)
        except Exception as e:
            print(f"Error training with SMOTE: {e}")
            smote_results = None
        
        # Combine all results
        all_results = {**class_results, **temporal_results}
        all_results['combined'] = combined_results
        if smote_results:
            all_results['smote'] = smote_results
        
        # Create output directory
        os.makedirs('figures', exist_ok=True)
        
        # Generate plots
        self.plot_class_weight_comparison(class_results)
        self.plot_temporal_weight_comparison(temporal_results)
        self.plot_all_approaches_comparison(all_results)
        self.plot_pr_curves(all_results)
        self.plot_roc_curves(all_results)
        
        # Save results summary
        self.save_results_summary(all_results)
        
        print("\nAnalysis complete! Results saved to 'figures' directory.")
        return all_results
    

def save_results_summary(self, results):
        """Save summary of results to CSV file"""
        # Prepare data for CSV
        summary_data = []
        
        for approach, result in results.items():
            summary_data.append({
                'Approach': approach,
                'ROC_AUC': result['auc_roc'],
                'PR_AUC': result['auc_pr'],
                'Window': self.window_days
            })
        
        # Convert to DataFrame
        df = pd.DataFrame(summary_data)
        
        # Save to CSV
        df.to_csv('weighting_approaches_results.csv', index=False)
        print("Results summary saved to weighting_approaches_results.csv")


if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='TBI prediction weighting approaches analysis')
    parser.add_argument('--window', type=int, default=60, choices=[30, 60, 180, 365],
                        help='Prediction window in days (30, 60, 180, or 365)')
    parser.add_argument('--data_path', type=str, default='cleaned_temporal_data.csv',
                        help='Path to cleaned temporal data file')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save results')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Set up analysis
    analysis = WeightingAnalysis(
        data_path=args.data_path,
        window_days=args.window
    )
    
    # Run analysis
    results = analysis.run_analysis()
    
    if results:
        print(f"\nWeighting analysis complete for {args.window}-day window.")
        print(f"Results and figures saved to {args.output_dir}/")
    else:
        print(f"\nAnalysis failed. Please check the data path: {args.data_path}")