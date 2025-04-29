import numpy as np
import os

# Create temporal model summary file
def create_temporal_summary():
    # Define windows
    windows = [30, 60, 180, 365]
    
    # Create results dictionary with metrics from the document
    results = {}
    
    for window in windows:
        # Define metrics for each window based on your document
        if window == 30:
            metrics = {
                'auc_roc': 0.709,
                'precision': 0.49,
                'recall': 0.59,
                'classification_report': "2_temporal_lstm_30d model report"
            }
        elif window == 60:
            metrics = {
                'auc_roc': 0.725,
                'precision': 0.51,
                'recall': 0.57,
                'classification_report': "2_temporal_lstm_60d model report"
            }
        elif window == 180:
            metrics = {
                'auc_roc': 0.692,
                'precision': 0.59,
                'recall': 0.59,
                'classification_report': "2_temporal_lstm_180d model report"
            }
        elif window == 365:
            metrics = {
                'auc_roc': 0.685,
                'precision': 0.64,
                'recall': 0.57,
                'classification_report': "2_temporal_lstm_365d model report"
            }
        
        results[window] = metrics
    
    # Save as .npy file like the other result files
    output_file = "2_temporal_lstm_results.npy"
    np.save(output_file, results)
    print(f"Created summary file: {output_file}")
    
    # Display the contents for verification
    print("\nSummary contents:")
    for window, metrics in results.items():
        print(f"{window}-day window: AUC ROC = {metrics['auc_roc']}, "
              f"Precision = {metrics['precision']}, Recall = {metrics['recall']}")

if __name__ == "__main__":
    create_temporal_summary()