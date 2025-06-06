import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
import os

class TemporalLSTM:
    """Single stage LSTM with temporal weighting"""
    
    def __init__(self, input_shape):
        self.model = self._create_model(input_shape)
    
    def _create_model(self, input_shape):
        """Create temporally-aware LSTM model"""
        inputs = keras.layers.Input(shape=input_shape)
        
        # Masking layer for variable length sequences
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
        
        # Create and compile model
        model = keras.Model(inputs=inputs, outputs=output)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )
        return model

def prepare_temporal_data(df, window_days, previous_predictions=None):
    """
    Prepare data with temporal weighting and previous window predictions.
    """
    print(f"\nPreparing data for {window_days}-day window...")

    # Sort by patient and time
    df = df.sort_values(['patient_id', 'days_from_tbi'])

    sequences = []
    temporal_weights = []
    labels = []
    patient_ids = []
    
    all_feature_cols = set()  # Track all feature columns across patients

    for patient_id, patient_data in df.groupby('patient_id'):
        pre_tbi = patient_data[patient_data['days_from_tbi'] <= 0].copy()

        if len(pre_tbi) > 0:
            # Get features from pre-TBI period
            days = abs(pre_tbi['days_from_tbi'])
            weights = np.exp(-0.01 * days)

            feature_cols = pre_tbi.select_dtypes(include=[np.number]).columns
            feature_cols = feature_cols.drop(['patient_id', 'days_from_tbi'])
            
            # Remove any diagnosis columns from features
            feature_cols = [col for col in feature_cols if not col.endswith('_mh')]

            # Add previous window predictions if available
            if previous_predictions is not None:
                for prev_window, preds in previous_predictions.items():
                    if patient_id in preds:
                        pre_tbi[f'pred_window_{prev_window}'] = np.full(len(pre_tbi), preds[patient_id])
                    else:
                        pre_tbi[f'pred_window_{prev_window}'] = 0
                    feature_cols = feature_cols + [f'pred_window_{prev_window}']

            all_feature_cols.update(feature_cols)

            sequence = pre_tbi[feature_cols].values
            sequences.append(sequence)
            temporal_weights.append(weights)

            # Check for MH diagnosis in the specific window
            post_tbi = patient_data[(patient_data['days_from_tbi'] > 0) & 
                                  (patient_data['days_from_tbi'] <= window_days)]
            
            # Look at all diagnosis columns that end with '_mh'
            mh_columns = [col for col in post_tbi.columns if col.endswith('_mh')]
            got_mh = post_tbi[mh_columns].any().any()
            
            labels.append(got_mh)
            patient_ids.append(patient_id)

    # Ensure all sequences have the same number of features
    all_feature_cols = list(all_feature_cols)
    for i in range(len(sequences)):
        df_seq = pd.DataFrame(sequences[i], columns=all_feature_cols).fillna(0)
        sequences[i] = df_seq.values

    return pad_sequences(sequences), pad_sequences(temporal_weights), np.array(labels), np.array(patient_ids)

def pad_sequences(sequences):
    """Consistent padding for sequences"""
    max_len = max(len(seq) for seq in sequences)
    
    if isinstance(sequences[0], np.ndarray):
        n_features = sequences[0].shape[1]
        padded = np.zeros((len(sequences), max_len, n_features))
    else:
        padded = np.zeros((len(sequences), max_len))
    
    for i, seq in enumerate(sequences):
        padded[i, :len(seq)] = seq
    
    return padded

def train_windows(windows):
    """Train models for multiple windows using previous predictions."""
    # Load data once
    print("Loading data...")
    df = pd.read_csv('cleaned_temporal_data.csv')
    
    previous_predictions = {}
    results = {}

    for window in windows:
        print(f"\nProcessing {window}-day window")

        X, weights, y, patient_ids = prepare_temporal_data(
            df,
            window, 
            previous_predictions=previous_predictions
        )

        # Print class distribution
        positive_rate = y.mean()
        print(f"\nClass distribution for {window}-day window:")
        print(f"Positive (MH diagnosis): {positive_rate:.1%}")
        print(f"Negative (No MH): {1-positive_rate:.1%}")

        # Create class weights (weight positive cases higher)
        class_weight = {
            0: 1.0,
            1: 2.0  # Weight positive cases 2x more
        }

        unique_patients = np.unique(patient_ids)
        np.random.shuffle(unique_patients)
        split_idx = int(len(unique_patients) * 0.8)

        train_patients = unique_patients[:split_idx]
        test_patients = unique_patients[split_idx:]

        train_mask = np.isin(patient_ids, train_patients)
        test_mask = np.isin(patient_ids, test_patients)

        X_train, X_test = X[train_mask], X[test_mask]
        w_train, w_test = weights[train_mask], weights[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        checkpoint_path = f'2_temporal_lstm_{window}d_best.keras'

        if os.path.exists(checkpoint_path):
            print(f"Loading saved model for {window}-day window...")
            model = keras.models.load_model(checkpoint_path)
        else:
            print(f"Training new model for {window}-day window...")
            model = TemporalLSTM(input_shape=(X_train.shape[1], X_train.shape[2])).model

            callback_list = [
                keras.callbacks.ModelCheckpoint(
                    checkpoint_path, monitor='val_auc', save_best_only=True, mode='max', verbose=1
                ),
                keras.callbacks.EarlyStopping(
                    monitor='val_auc', patience=10, restore_best_weights=True, verbose=1
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_auc', factor=0.5, patience=5, min_lr=0.0001, verbose=1
                )
            ]

            sample_weights = w_train.mean(axis=1)
            sample_weights[y_train == 1] *= 2.0

            history = model.fit(
                X_train, y_train,
                sample_weight=sample_weights,
                validation_data=(X_test, y_test, w_test.mean(axis=1)),
                epochs=50,
                batch_size=32,
                callbacks=callback_list
            )

            model.save(f'2_temporal_lstm_{window}d_final.keras')

        # Get predictions
        y_pred_proba = model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int)

        # Store predictions for next window
        test_patient_ids = patient_ids[test_mask]
        previous_predictions[window] = dict(zip(test_patient_ids, y_pred_proba))

        # Calculate metrics
        results[window] = {
            'auc_roc': roc_auc_score(y_test, y_pred_proba),
            'auc_pr': average_precision_score(y_test, y_pred_proba),
            'classification_report': classification_report(y_test, y_pred),
        }

        print(f"\nResults for {window}-day window:")
        print(f"ROC AUC: {results[window]['auc_roc']:.3f}")
        print(f"PR AUC: {results[window]['auc_pr']:.3f}")
        print("\nClassification Report:")
        print(results[window]['classification_report'])

    return results

def main():
    # Train models for different windows
    windows = [180, 365]
    results = train_windows(windows)
    
    # Save results
    np.save('temporal_lstm_results.npy', results)

if __name__ == "__main__":
    main()