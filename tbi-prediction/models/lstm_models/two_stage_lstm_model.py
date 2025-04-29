import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from keras import backend as K

class MaskedAttentionLayer(keras.layers.Layer):
    """Attention layer that properly supports masking"""
    
    def __init__(self, **kwargs):
        super(MaskedAttentionLayer, self).__init__(**kwargs)
        self.supports_masking = True  # Enable mask support
    
    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], 1),
            initializer='glorot_uniform',
            trainable=True
        )
        super(MaskedAttentionLayer, self).build(input_shape)
    
    def call(self, inputs, mask=None):
        # Calculate attention scores
        attention = tf.keras.backend.tanh(tf.keras.backend.dot(inputs, self.W))
        
        # Apply mask if provided
        if mask is not None:
            # Convert mask to float and add a dimension for broadcasting
            mask = tf.keras.backend.cast(mask, tf.keras.backend.dtype(inputs))
            mask = tf.keras.backend.expand_dims(mask, axis=-1)
            # Set attention scores to a large negative value for masked positions
            attention = attention * mask - 1e10 * (1 - mask)
        
        # Get attention weights
        attention_weights = tf.keras.backend.softmax(attention, axis=1)
        
        # Apply attention weights to input
        weighted_input = inputs * attention_weights
        return tf.keras.backend.sum(weighted_input, axis=1)
    
    def compute_mask(self, inputs, mask=None):
        # Return None as we don't need to propagate the mask further
        return None
    
class TwoStagePredictor:
    """Improved two-stage prediction model"""
    
    def __init__(self, input_shape, negative_threshold=0.8):
        self.negative_threshold = negative_threshold
        self.stage1_model = self._create_stage_model(input_shape, name='stage1')
        self.stage2_model = self._create_stage_model(input_shape, name='stage2')
    
    def _create_stage_model(self, input_shape, name):
        """Create improved single stage model"""
        inputs = keras.layers.Input(shape=input_shape, name=f'{name}_input')
        
        # Masking layer
        masked = keras.layers.Masking(mask_value=0., name=f'{name}_masking')(inputs)
        
        # Bidirectional LSTM with proper mask propagation
        lstm = keras.layers.Bidirectional(
            keras.layers.LSTM(128, return_sequences=True),
            name=f'{name}_bilstm'
        )(masked)
        
        # Batch normalization with proper masking
        normalized = keras.layers.BatchNormalization(name=f'{name}_batchnorm')(lstm)
        dropout1 = keras.layers.Dropout(0.3, name=f'{name}_dropout1')(normalized)
        
        # Attention mechanism
        attention = MaskedAttentionLayer(name=f'{name}_attention')(dropout1)
        
        # Dense layers
        dense = keras.layers.Dense(128, activation='relu', name=f'{name}_dense1')(attention)
        normalized2 = keras.layers.BatchNormalization(name=f'{name}_batchnorm2')(dense)
        dropout2 = keras.layers.Dropout(0.2, name=f'{name}_dropout2')(normalized2)
        
        # Output layer
        output = keras.layers.Dense(1, activation='sigmoid', name=f'{name}_output')(dropout2)
        
        # Create model
        model = keras.Model(inputs=inputs, outputs=output)
        
        # Compile with proper metrics
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.AUC(name='auc'),  # Changed from 'AUC' to 'auc'
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall')
            ]
        )
        return model
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """Train both stages with proper validation"""
        # Train stage 1 (negative detector)
        print("Training Stage 1 (Negative Detector)...")
        history1 = self.stage1_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val) if X_val is not None else None,
            **kwargs
        )
        
        # Get predictions from stage 1
        stage1_pred = self.stage1_model.predict(X_train)
        uncertain_mask = stage1_pred.flatten() >= (1 - self.negative_threshold)
        
        if uncertain_mask.any():
            # Train stage 2 on uncertain cases
            print("\nTraining Stage 2 (Uncertain Cases)...")
            X_train_s2 = X_train[uncertain_mask]
            y_train_s2 = y_train[uncertain_mask]
            
            if X_val is not None:
                stage1_val_pred = self.stage1_model.predict(X_val)
                val_uncertain_mask = stage1_val_pred.flatten() >= (1 - self.negative_threshold)
                X_val_s2 = X_val[val_uncertain_mask]
                y_val_s2 = y_val[val_uncertain_mask]
            else:
                X_val_s2, y_val_s2 = None, None
            
            history2 = self.stage2_model.fit(
                X_train_s2, y_train_s2,
                validation_data=(X_val_s2, y_val_s2) if X_val is not None else None,
                **kwargs
            )
            return history1, history2
        return history1, None
    
    def predict(self, X):
        """Make predictions using both stages"""
        # Stage 1 predictions
        stage1_pred = self.stage1_model.predict(X)
        
        # Initialize final predictions
        final_pred = np.zeros_like(stage1_pred)
        
        # Find uncertain cases
        uncertain_mask = stage1_pred.flatten() >= (1 - self.negative_threshold)
        
        if uncertain_mask.any():
            # Get stage 2 predictions for uncertain cases
            stage2_pred = self.stage2_model.predict(X[uncertain_mask])
            final_pred[uncertain_mask] = stage2_pred
        
        return final_pred
    
def pad_sequences(sequences, max_seq_length=None):
    """
    Enhanced padding with better handling of variable lengths
    """
    # If max_length not specified, use 95th percentile
    if max_seq_length is None:
        lengths = [len(seq) for seq in sequences]
        max_seq_length = int(np.percentile(lengths, 95))
    
    n_features = sequences[0].shape[1]
    padded = np.zeros((len(sequences), max_seq_length, n_features))
    
    for i, seq in enumerate(sequences):
        # Center the sequence in the padded array
        if len(seq) > max_seq_length:
            start_idx = (len(seq) - max_seq_length) // 2
            padded[i] = seq[start_idx:start_idx + max_seq_length]
        else:
            start_idx = (max_seq_length - len(seq)) // 2
            padded[i, start_idx:start_idx + len(seq)] = seq
    
    return padded

def prepare_temporal_data(window_start, window_end, sliding_window=False, previous_predictions=None):
    """
    Prepare temporal data with option for sliding or growing windows
    
    Args:
        window_start (int): Start day for prediction window
        window_end (int): End day for prediction window
        sliding_window (bool): If True, use sliding windows. If False, use growing windows
        previous_predictions (dict): Dictionary of predictions from previous windows
    """
    print(f"\nPreparing sequences for {window_start}-{window_end} day window...")
    df = pd.read_csv('cleaned_temporal_data.csv')
    
    # Sort by patient and time
    df = df.sort_values(['patient_id', 'days_from_tbi'])
    
    sequences = []
    labels = []
    patient_ids = []
    
    for patient_id, patient_data in df.groupby('patient_id'):
        # Get pre-TBI data based on window type
        if sliding_window:
            pre_tbi = patient_data[
                (patient_data['days_from_tbi'] >= -window_start) & 
                (patient_data['days_from_tbi'] <= 0)
            ].copy()
        else:
            pre_tbi = patient_data[patient_data['days_from_tbi'] <= 0].copy()
        
        if len(pre_tbi) > 0:
            # Get features
            sequence_features = pre_tbi.select_dtypes(include=[np.number]).drop(
                ['patient_id'], axis=1
            )
            
            # Add previous window predictions if available
            if previous_predictions is not None:
                for window, preds in previous_predictions.items():
                    if patient_id in preds:
                        sequence_features[f'pred_window_{window}'] = np.full(len(sequence_features), preds[patient_id])
            
            sequence = sequence_features.values
            
            # Get label from post-TBI window
            post_tbi = patient_data[
                (patient_data['days_from_tbi'] > window_start) & 
                (patient_data['days_from_tbi'] <= window_end)
            ]
            has_mh = post_tbi['has_mh_diagnosis'].any()
            
            sequences.append(sequence)
            labels.append(has_mh)
            patient_ids.append(patient_id)
    
    return pad_sequences(sequences), np.array(labels), np.array(patient_ids)

def train_temporal_windows(windows, sliding=False):
    """
    Train models for multiple temporal windows with improved callbacks
    
    Args:
        windows (list): List of (start, end) tuples defining windows
        sliding (bool): If True, use sliding windows. If False, use growing windows
    """
    previous_predictions = {}
    results = {}
    
    for start, end in windows:
        print(f"\nProcessing window {start}-{end} days")
        
        # Prepare data with previous predictions
        X, y, patient_ids = prepare_temporal_data(
            start, end, 
            sliding_window=sliding,
            previous_predictions=previous_predictions
        )
        
        # Split data
        unique_patients = np.unique(patient_ids)
        np.random.shuffle(unique_patients)
        split_idx = int(len(unique_patients) * 0.8)
        
        train_patients = unique_patients[:split_idx]
        test_patients = unique_patients[split_idx:]
        
        train_mask = np.isin(patient_ids, train_patients)
        test_mask = np.isin(patient_ids, test_patients)
        
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        
        # Calculate class weights
        class_weights = {
            0: 1.0,
            1: len(y_train[y_train == 0]) / len(y_train[y_train == 1])
        }
        
        # Create callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_auc',
                patience=10,
                mode='max',
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=f'lstm_model_{start}_{end}_' + '{epoch:02d}-{val_auc:.2f}.keras',
                monitor='val_auc',
                mode='max',
                save_best_only=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_auc',
                factor=0.5,
                patience=5,
                min_lr=0.0001,
                verbose=1
            )
        ]
        
        # Create and train two-stage model
        model = TwoStagePredictor(
            input_shape=(X_train.shape[1], X_train.shape[2])
        )
        
        # Train the model
        history1, history2 = model.fit(
            X_train, y_train,
            X_val=X_test, y_val=y_test,
            epochs=50,
            batch_size=32,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )
        
        # Get predictions
        y_pred_proba = model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Store predictions for next window
        test_patient_ids = patient_ids[test_mask]
        previous_predictions[f"{start}_{end}"] = dict(zip(test_patient_ids, y_pred_proba))
        
        # Save detailed predictions for analysis
        pred_df = pd.DataFrame({
            'patient_id': test_patient_ids,
            'true_label': y_test,
            'predicted_prob': y_pred_proba.flatten(),
            'predicted_label': y_pred.flatten()
        })
        pred_df.to_csv(f'predictions_{start}_{end}d.csv', index=False)
        
        # Calculate metrics
        results[(start, end)] = {
            'auc_roc': roc_auc_score(y_test, y_pred_proba),
            'auc_pr': average_precision_score(y_test, y_pred_proba),
            'classification_report': classification_report(y_test, y_pred),
            'predictions': pred_df,
            'history1': history1.history,
            'history2': history2.history if history2 is not None else None
        }
        
        print(f"\nResults for {start}-{end} day window:")
        print(f"ROC AUC: {results[(start, end)]['auc_roc']:.3f}")
        print(f"PR AUC: {results[(start, end)]['auc_pr']:.3f}")
        print("\nClassification Report:")
        print(results[(start, end)]['classification_report'])
    
    return results

def main():
    # Define windows
    growing_windows = [(0, 30), (0, 60), (0, 180), (0, 365)]
    sliding_windows = [(0, 30), (31, 60), (61, 180), (181, 365)]
    
    # Train with growing windows
    print("\nTraining with growing windows...")
    growing_results = train_temporal_windows(growing_windows, sliding=False)
    
    # Train with sliding windows
    print("\nTraining with sliding windows...")
    sliding_results = train_temporal_windows(sliding_windows, sliding=True)
    
    # Save all results
    np.save('growing_window_results.npy', growing_results)
    np.save('sliding_window_results.npy', sliding_results)

if __name__ == "__main__":
    main()