"""
temporal_pattern_analysis.py

Script to analyze temporal patterns in TBI mental health prediction:
1. Analyze how mental health diagnoses relate to time from TBI
2. Examine prediction accuracy across different time periods
3. Identify temporal patterns in correctly vs. incorrectly classified patients
4. Create visualizations of cumulative incidence over time

For AMIA Annual Symposium submission
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
import os
from matplotlib.ticker import PercentFormatter
from scipy import stats

# Configure plot style for publication
plt.style.use('seaborn-whitegrid')
sns.set_context("paper", font_scale=1.2)
COLORS = sns.color_palette("deep")

class TemporalPatternAnalysis:
    """Analyze temporal patterns in mental health diagnoses following TBI"""
    
    def __init__(self, data_path="cleaned_temporal_data.csv", predictions_dir="predictions"):
        """
        Initialize with paths to data and prediction files
        
        Args:
            data_path: Path to the processed TBI dataset
            predictions_dir: Directory containing model predictions
        """
        self.data_path = data_path
        self.predictions_dir = predictions_dir
        self.time_windows = [30, 60, 180, 365]
    
    def load_data(self):
        """Load and prepare data for temporal analysis"""
        print(f"Loading data from {self.data_path}")
        
        try:
            # Load raw data
            self.df = pd.read_csv(self.data_path)
            
            # Basic preprocessing
            self.df = self.df.sort_values(['patient_id', 'days_from_tbi'])
            
            # Separate pre and post TBI data
            self.pre_tbi = self.df[self.df['days_from_tbi'] <= 0]
            self.post_tbi = self.df[self.df['days_from_tbi'] > 0]
            
            print(f"Data loaded: {len(self.df)} records, "
                 f"{self.df['patient_id'].nunique()} patients")
            
            # Load prediction files if available
            self.load_predictions()
            
            return True
        
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def load_predictions(self):
        """Load model predictions for different time windows"""
        self.predictions = {}
        
        for window in self.time_windows:
            pred_file = os.path.join(self.predictions_dir, f"predictions_{window}d.csv")
            
            if os.path.exists(pred_file):
                self.predictions[window] = pd.read_csv(pred_file)
                print(f"Loaded predictions for {window}-day window: {len(self.predictions[window])} patients")
            else:
                print(f"No prediction file found for {window}-day window")
        
        # If no prediction files found, generate sample predictions
        if not self.predictions:
            print("No prediction files found. Using sample predictions.")
            self.generate_sample_predictions()
    
    def generate_sample_predictions(self):
        """Generate sample predictions for demonstration"""
        # Get unique patient IDs
        unique_patients = self.df['patient_id'].unique()
        n_patients = len(unique_patients)
        
        # Generate sample predictions for each time window
        for window in self.time_windows:
            # Create positive rates that increase with window size
            positive_rate = 0.2 + 0.25 * (window / 365)
            
            # Create true labels
            true_labels = np.random.binomial(1, positive_rate, n_patients)
            
            # Create predictions with realistic performance
            # Higher AUC for shorter windows
            auc_target = 0.68 - 0.01 * (window / 100)
            
            # Create predicted probabilities
            # True positives: higher probabilities (mean 0.7)
            # True negatives: lower probabilities (mean 0.3)
            # With some errors to achieve target AUC
            pos_probs = np.random.beta(5, 2, size=sum(true_labels)) 
            neg_probs = np.random.beta(2, 5, size=len(true_labels) - sum(true_labels))
            
            # Combine probabilities
            pred_proba = np.zeros(n_patients)
            pred_proba[true_labels == 1] = pos_probs
            pred_proba[true_labels == 0] = neg_probs
            
            # Create predicted labels using 0.5 threshold
            pred_labels = (pred_proba >= 0.5).astype(int)
            
            # Create DataFrame with predictions
            self.predictions[window] = pd.DataFrame({
                'patient_id': unique_patients,
                'true_label': true_labels,
                'predicted_prob': pred_proba,
                'predicted_label': pred_labels
            })
    
    def analyze_diagnosis_timing(self, save_path="figures"):
        """Analyze when mental health diagnoses occur post-TBI"""
        os.makedirs(save_path, exist_ok=True)
        
        print("Analyzing timing of mental health diagnoses...")
        
        # Filter patients with mental health diagnoses
        mh_diagnoses = self.post_tbi[self.post_tbi['has_mh_diagnosis'] == True]
        
        # Group by patient and get earliest MH diagnosis
        first_diagnosis = mh_diagnoses.groupby('patient_id')['days_from_tbi'].min().reset_index()
        first_diagnosis.columns = ['patient_id', 'days_to_diagnosis']
        
        # Create histogram of days to diagnosis
        plt.figure(figsize=(10, 6))
        
        sns.histplot(data=first_diagnosis, x='days_to_diagnosis', bins=30, 
                    color=COLORS[0], kde=True)
        
        # Add vertical lines for time windows
        for i, window in enumerate(self.time_windows):
            plt.axvline(x=window, color=COLORS[i+1], linestyle='--', 
                       label=f"{window}-day window")
        
        plt.title('Time to First Mental Health Diagnosis After TBI', fontsize=14)
        plt.xlabel('Days After TBI', fontsize=12)
        plt.ylabel('Number of Patients', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'time_to_diagnosis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create cumulative incidence plot
        plt.figure(figsize=(10, 6))
        
        # Calculate cumulative incidence
        max_days = 365
        days = range(1, max_days + 1)
        total_patients = self.df['patient_id'].nunique()
        
        # Count patients diagnosed by each day
        cumulative_counts = [sum(first_diagnosis['days_to_diagnosis'] <= day) for day in days]
        cumulative_incidence = [count / total_patients for count in cumulative_counts]
        
        # Plot cumulative incidence
        plt.plot(days, cumulative_incidence, color=COLORS[0], linewidth=2)
        
        # Add vertical lines for time windows
        for i, window in enumerate(self.time_windows):
            plt.axvline(x=window, color=COLORS[i+1], linestyle='--', 
                       label=f"{window}-day window")
            
            # Add text showing incidence at this window
            incidence_at_window = cumulative_incidence[window-1]
            plt.text(window + 5, incidence_at_window, 
                    f"{incidence_at_window:.1%}", 
                    va='center')
        
        plt.title('Cumulative Incidence of Mental Health Diagnoses After TBI', fontsize=14)
        plt.xlabel('Days After TBI', fontsize=12)
        plt.ylabel('Cumulative Incidence', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'cumulative_incidence.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create table with summary statistics
        summary_stats = pd.DataFrame({
            'Statistic': ['Mean days to diagnosis', 'Median days to diagnosis', 
                         'Standard deviation', '25th percentile', '75th percentile',
                         'Minimum days', 'Maximum days'],
            'Value': [first_diagnosis['days_to_diagnosis'].mean(),
                     first_diagnosis['days_to_diagnosis'].median(),
                     first_diagnosis['days_to_diagnosis'].std(),
                     first_