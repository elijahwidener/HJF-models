import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
import os
'''
instead of lowering batch size, use feature selection so there is less data to process,
also can train on less data period
MRMR feature selection
PCA, 
double check how features are scaled
Often scaling of the features decreases the time it takes to train the model
Information theory based feature selection
imbalance learn library - look into techniques to handle class imbalance
IMBM pipeline
SMOTE

Read about class imbalance, be careful not to add bias
from a use case pov, it is better to have a greater rate of false positive, than to let a patient go undiagnosed


 -WHITEPAPTER
 AMIA, anual symposium, student papers, submit by March 19.  
'''

class StaticWindowLSTM:
    """LSTM model for static window prediction"""

    def __init__(self, input_shape):
        self.model = self._create_model(input_shape)

    def _create_model(self, input_shape):
        """Create LSTM model"""
        inputs = keras.layers.Input(shape=input_shape)

        masked = keras.layers.Masking(mask_value=0.)(inputs)
        lstm = keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True))(masked)
        pooled = keras.layers.GlobalAveragePooling1D()(lstm)
        dense = keras.layers.Dense(32, activation='relu')(pooled)
        dropout = keras.layers.Dropout(0.3)(dense)
        output = keras.layers.Dense(1, activation='sigmoid')(dropout)

        model = keras.Model(inputs=inputs, outputs=output)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                      loss='binary_crossentropy',
                      metrics=['accuracy', keras.metrics.AUC(name='auc')])
        return model


def prepare_static_data(start_day, end_day):
    """
    Prepare data for static window prediction with temporal weighting.
    Uses all data before start_day for training, and labels are based on MH diagnosis in [start_day, end_day].
    """
    print(f"\nPreparing data for days {start_day}-{end_day}...")
    df = pd.read_csv('cleaned_temporal_data.csv')
    df = df.sort_values(['patient_id', 'days_from_tbi'])

    sequences = []
    temporal_weights = []
    labels = []
    patient_ids = []

    all_feature_cols = set()

    for patient_id, patient_data in df.groupby('patient_id'):
        pre_window_data = patient_data[patient_data['days_from_tbi'] < start_day].copy()
        if len(pre_window_data) > 0:
            # Calculate temporal weights
            days = abs(pre_window_data['days_from_tbi'])
            weights = np.exp(-0.01 * days)
            temporal_weights.append(weights)

            feature_cols = pre_window_data.select_dtypes(include=[np.number]).columns
            feature_cols = feature_cols.drop(['patient_id', 'days_from_tbi'])

            all_feature_cols.update(feature_cols)

            sequence = pre_window_data[feature_cols].values
            sequences.append(sequence)

            # Get label from specific window
            window_data = patient_data[
                (patient_data['days_from_tbi'] >= start_day) & 
                (patient_data['days_from_tbi'] <= end_day)
            ]
            has_mh = window_data['has_mh_diagnosis'].any()
            labels.append(has_mh)
            patient_ids.append(patient_id)

    all_feature_cols = list(all_feature_cols)
    for i in range(len(sequences)):
        df_seq = pd.DataFrame(sequences[i], columns=all_feature_cols).fillna(0)
        sequences[i] = df_seq.values

    # Compute and display class distribution
    pos_count = np.sum(labels)
    neg_count = len(labels) - pos_count
    pos_pct = (pos_count / len(labels)) * 100 if len(labels) > 0 else 0
    neg_pct = (neg_count / len(labels)) * 100 if len(labels) > 0 else 0

    print(f"Positive cases: {pos_count} ({pos_pct:.2f}%)")
    print(f"Negative cases: {neg_count} ({neg_pct:.2f}%)")

    return pad_sequences(sequences), pad_sequences(temporal_weights), np.array(labels), np.array(patient_ids)


def pad_sequences(sequences):
    """Consistent padding for sequences"""
    if not sequences:
        return np.array([])
    
    max_len = max(len(seq) for seq in sequences)
    if isinstance(sequences[0], (np.ndarray, list)):
        n_features = sequences[0].shape[1] if isinstance(sequences[0], np.ndarray) else len(sequences[0][0])
        padded = np.zeros((len(sequences), max_len, n_features))
    else:
        padded = np.zeros((len(sequences), max_len))
    
    for i, seq in enumerate(sequences):
        padded[i, :len(seq)] = seq
    
    return padded


def train_static_windows(windows):
    """Train models for static windows with temporal weighting"""
    results = {}

    for (start_day, end_day) in windows:
        print(f"\nProcessing window {start_day}-{end_day} days")

        X, weights, y, patient_ids = prepare_static_data(start_day, end_day)

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

        checkpoint_path = f'static_lstm_{start_day}-{end_day}d_best.keras'

        if os.path.exists(checkpoint_path):
            print(f"Loading saved model for days {start_day}-{end_day}...")
            model = keras.models.load_model(checkpoint_path)
        else:
            print(f"Training new model for days {start_day}-{end_day}...")
            model = StaticWindowLSTM(input_shape=(X_train.shape[1], X_train.shape[2])).model

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

            model.fit(
                X_train, y_train,
                sample_weight=w_train.mean(axis=1),  # Added temporal weighting
                validation_data=(X_test, y_test, w_test.mean(axis=1)),  # Added validation weights
                epochs=50,
                batch_size=32,
                callbacks=callback_list
            )

            model.save(f'static_lstm_{start_day}-{end_day}d_final.keras')

        y_pred_proba = model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int)

        results[(start_day, end_day)] = {
            'auc_roc': roc_auc_score(y_test, y_pred_proba),
            'auc_pr': average_precision_score(y_test, y_pred_proba),
            'classification_report': classification_report(y_test, y_pred),
        }

        print(f"\nResults for {start_day}-{end_day}-day window:")
        print(f"ROC AUC: {results[(start_day, end_day)]['auc_roc']:.3f}")
        print(f"PR AUC: {results[(start_day, end_day)]['auc_pr']:.3f}")
        print("\nClassification Report:")
        print(results[(start_day, end_day)]['classification_report'])

    return results


def main():
    static_windows = [(0, 30), (31, 60), (61, 90), (91, 120)]
    results = train_static_windows(static_windows)
    np.save('static_lstm_results.npy', results)


if __name__ == "__main__":
    main()
