"""
risk_stratification.py

Script to develop a risk stratification approach based on model predictions:
1. Define risk tiers (low, medium, high) based on predicted probabilities
2. Calculate performance metrics within each tier
3. Analyze patient characteristics across risk tiers
4. Create visualizations of risk distribution
5. Generate "number needed to screen" metrics

For AMIA Annual Symposium submission
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import roc_curve, precision_recall_curve
from matplotlib.ticker import PercentFormatter

# Configure plot style for publication
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)
COLORS = sns.color_palette("deep")

class RiskStratification:
    """Develop risk stratification from model predictions"""
    
    def __init__(self, predictions_dir="predictions", data_path="cleaned_temporal_data.csv"):
        """
        Initialize with prediction files and original data
        
        Args:
            predictions_dir: Directory containing model predictions
            data_path: Path to the original data file
        """
        self.predictions_dir = predictions_dir
        self.data_path = data_path
        self.windows = [30, 60, 180, 365]
        
        # Risk tier thresholds
        self.tier_thresholds = {
            'low_high': [0.3, 0.7],  # [low_threshold, high_threshold]
            'tercile': None,  # Will be calculated from data
            'quartile': None  # Will be calculated from data
        }
    
    def load_data(self):
        """Load prediction data and original data"""
        print("Loading prediction data...")
        
        # Dictionary to store predictions for each window
        self.predictions = {}
        
        # Load prediction files
        for window in self.windows:
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
        
        # Try to load original data if path exists
        if os.path.exists(self.data_path):
            print(f"Loading original data from {self.data_path}")
            self.original_data = pd.read_csv(self.data_path)
        else:
            print(f"Original data file not found at {self.data_path}")
            self.original_data = None
        
        return len(self.predictions) > 0
    
    def generate_sample_predictions(self):
        """Generate sample predictions for demonstration"""
        # Generate sample patient IDs
        n_patients = 1000
        patient_ids = np.arange(1, n_patients + 1)
        
        # Generate sample predictions for each window
        for window in self.windows:
            # Create positive rates that increase with window size
            positive_rate = 0.2 + 0.25 * (window / 365)
            
            # Create true labels
            np.random.seed(42)  # For reproducibility
            true_labels = np.random.binomial(1, positive_rate, n_patients)
            
            # Create predicted probabilities with some correlation to true labels
            # True positives: higher probabilities (mean 0.7)
            # True negatives: lower probabilities (mean 0.3)
            pos_probs = np.random.beta(7, 3, size=sum(true_labels))
            neg_probs = np.random.beta(3, 7, size=n_patients - sum(true_labels))
            
            pred_proba = np.zeros(n_patients)
            pred_proba[true_labels == 1] = pos_probs
            pred_proba[true_labels == 0] = neg_probs
            
            # Add some noise to make it realistic
            noise = np.random.normal(0, 0.1, n_patients)
            pred_proba = np.clip(pred_proba + noise, 0, 1)
            
            # Create predicted labels using 0.5 threshold
            pred_labels = (pred_proba >= 0.5).astype(int)
            
            # Create DataFrame with predictions
            self.predictions[window] = pd.DataFrame({
                'patient_id': patient_ids,
                'true_label': true_labels,
                'predicted_prob': pred_proba,
                'predicted_label': pred_labels
            })
    
    def define_risk_tiers(self, window=60, method='fixed'):
        """
        Define risk tiers based on predicted probabilities
        
        Args:
            window: Time window to use (30, 60, 180, 365 days)
            method: Method to define tiers ('fixed', 'tercile', 'quartile')
        
        Returns:
            DataFrame with risk tiers added
        """
        print(f"Defining risk tiers for {window}-day window using {method} method...")
        
        # Get predictions for specified window
        if window not in self.predictions:
            print(f"No predictions available for {window}-day window")
            return None
        
        pred_df = self.predictions[window].copy()
        
        if method == 'fixed':
            # Use fixed thresholds
            low_threshold, high_threshold = self.tier_thresholds['low_high']
            
            # Assign risk tiers
            conditions = [
                (pred_df['predicted_prob'] < low_threshold),
                (pred_df['predicted_prob'] >= low_threshold) & (pred_df['predicted_prob'] < high_threshold),
                (pred_df['predicted_prob'] >= high_threshold)
            ]
            choices = ['low', 'medium', 'high']
            
        elif method == 'tercile':
            # Divide into terciles
            terciles = pred_df['predicted_prob'].quantile([1/3, 2/3]).values
            self.tier_thresholds['tercile'] = terciles
            
            # Assign risk tiers
            conditions = [
                (pred_df['predicted_prob'] < terciles[0]),
                (pred_df['predicted_prob'] >= terciles[0]) & (pred_df['predicted_prob'] < terciles[1]),
                (pred_df['predicted_prob'] >= terciles[1])
            ]
            choices = ['low', 'medium', 'high']
            
        elif method == 'quartile':
            # Divide into quartiles
            quartiles = pred_df['predicted_prob'].quantile([0.25, 0.5, 0.75]).values
            self.tier_thresholds['quartile'] = quartiles
            
            # Assign risk tiers
            conditions = [
                (pred_df['predicted_prob'] < quartiles[0]),
                (pred_df['predicted_prob'] >= quartiles[0]) & (pred_df['predicted_prob'] < quartiles[1]),
                (pred_df['predicted_prob'] >= quartiles[1]) & (pred_df['predicted_prob'] < quartiles[2]),
                (pred_df['predicted_prob'] >= quartiles[2])
            ]
            choices = ['very low', 'low', 'medium', 'high']
            
        else:
            print(f"Unknown method: {method}")
            return None
        
        # Create risk tier column
        pred_df['risk_tier'] = np.select(conditions, choices, default='unknown')
        
        # Calculate tier statistics
        tier_stats = []
        
        for tier in np.unique(pred_df['risk_tier']):
            tier_df = pred_df[pred_df['risk_tier'] == tier]
            
            # Calculate metrics
            true_positives = sum((tier_df['true_label'] == 1) & (tier_df['predicted_label'] == 1))
            false_positives = sum((tier_df['true_label'] == 0) & (tier_df['predicted_label'] == 1))
            true_negatives = sum((tier_df['true_label'] == 0) & (tier_df['predicted_label'] == 0))
            false_negatives = sum((tier_df['true_label'] == 1) & (tier_df['predicted_label'] == 0))
            
            # Precision and recall
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            
            # F1 score
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Actual risk (percent of positive cases)
            actual_risk = sum(tier_df['true_label']) / len(tier_df) if len(tier_df) > 0 else 0
            
            # Number needed to screen to find one case
            nns = 1 / actual_risk if actual_risk > 0 else float('inf')
            
            tier_stats.append({
                'risk_tier': tier,
                'count': len(tier_df),
                'percentage': len(tier_df) / len(pred_df) * 100,
                'positive_count': sum(tier_df['true_label']),
                'actual_risk': actual_risk * 100,  # Convert to percentage
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'number_needed_to_screen': nns
            })
        
        # Convert to DataFrame
        self.tier_stats = pd.DataFrame(tier_stats)
        
        # Sort by risk tier (assuming 'low', 'medium', 'high')
        tier_order = {'very low': 0, 'low': 1, 'medium': 2, 'high': 3}
        self.tier_stats['tier_order'] = self.tier_stats['risk_tier'].map(tier_order)
        self.tier_stats = self.tier_stats.sort_values('tier_order').drop('tier_order', axis=1)
        
        print("\nRisk tier statistics:")
        print(self.tier_stats.to_string(index=False))
        
        return pred_df
    
    def plot_risk_distribution(self, pred_df, save_path="figures"):
        """Plot distribution of risk scores and actual outcomes"""
        os.makedirs(save_path, exist_ok=True)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        
        # Plot histograms separated by true label
        sns.histplot(data=pred_df, x='predicted_prob', hue='true_label', 
                    bins=20, element='step', common_norm=False,
                    palette=[COLORS[1], COLORS[0]])
        
        # Add vertical lines for risk tier thresholds if available
        method = 'fixed'  # Default to fixed thresholds
        if hasattr(self, 'tier_thresholds'):
            if self.tier_thresholds['tercile'] is not None:
                method = 'tercile'
                for threshold in self.tier_thresholds['tercile']:
                    plt.axvline(x=threshold, color='black', linestyle='--')
            elif self.tier_thresholds['quartile'] is not None:
                method = 'quartile'
                for threshold in self.tier_thresholds['quartile']:
                    plt.axvline(x=threshold, color='black', linestyle='--')
            else:
                for threshold in self.tier_thresholds['low_high']:
                    plt.axvline(x=threshold, color='black', linestyle='--')
        
        plt.title(f'Distribution of Risk Scores by Actual Outcome ({method.capitalize()} Thresholds)', fontsize=14)
        plt.xlabel('Predicted Probability', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.legend(title='Mental Health Diagnosis', labels=['Negative', 'Positive'])
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'risk_distribution_{method}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create risk vs. actual outcome plot
        plt.figure(figsize=(10, 6))
        
        # Plot actual risk by predicted probability decile
        pred_df['prob_decile'] = pd.qcut(pred_df['predicted_prob'], 10, labels=False)
        
        # Calculate actual risk per decile
        decile_stats = pred_df.groupby('prob_decile').agg(
            actual_risk=('true_label', 'mean'),
            count=('true_label', 'count')
        ).reset_index()
        
        decile_stats['actual_risk_pct'] = decile_stats['actual_risk'] * 100
        decile_stats['prob_decile'] = decile_stats['prob_decile'] + 1  # Make 1-based
        
        # Plot as bar chart
        plt.bar(decile_stats['prob_decile'], decile_stats['actual_risk_pct'], color=COLORS[0])
        
        # Add text labels
        for i, row in decile_stats.iterrows():
            plt.text(row['prob_decile'], row['actual_risk_pct'] + 2, 
                    f"{row['actual_risk_pct']:.1f}%\n(n={row['count']})", 
                    ha='center', va='bottom', fontsize=8)
        
        plt.title('Actual Risk by Predicted Probability Decile', fontsize=14)
        plt.xlabel('Predicted Probability Decile', fontsize=12)
        plt.ylabel('Actual Risk (%)', fontsize=12)
        plt.xticks(range(1, 11))
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'risk_calibration.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved risk distribution plots to {save_path}/")
    
    def plot_risk_tier_metrics(self, save_path="figures"):
        """Plot performance metrics by risk tier"""
        os.makedirs(save_path, exist_ok=True)
        
        if not hasattr(self, 'tier_stats'):
            print("No tier statistics available. Run define_risk_tiers() first.")
            return
        
        # Create plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot actual risk by tier
        sns.barplot(x='risk_tier', y='actual_risk', data=self.tier_stats, ax=axes[0], color=COLORS[0])
        axes[0].set_title('Actual Risk by Tier', fontsize=12)
        axes[0].set_xlabel('Risk Tier', fontsize=10)
        axes[0].set_ylabel('Actual Risk (%)', fontsize=10)
        
        # Add text labels
        for i, row in self.tier_stats.iterrows():
            axes[0].text(i, row['actual_risk'] + 2, 
                       f"{row['actual_risk']:.1f}%", 
                       ha='center', va='bottom')
        
        # Plot precision and recall by tier
        tier_metrics = self.tier_stats.melt(
            id_vars='risk_tier', 
            value_vars=['precision', 'recall', 'f1_score'],
            var_name='metric', value_name='value'
        )
        
        sns.barplot(x='risk_tier', y='value', hue='metric', data=tier_metrics, ax=axes[1], palette=COLORS[0:3])
        axes[1].set_title('Performance Metrics by Tier', fontsize=12)
        axes[1].set_xlabel('Risk Tier', fontsize=10)
        axes[1].set_ylabel('Value', fontsize=10)
        axes[1].legend(title='Metric')
        
        # Plot number needed to screen by tier
        sns.barplot(x='risk_tier', y='number_needed_to_screen', data=self.tier_stats, ax=axes[2], color=COLORS[3])
        axes[2].set_title('Number Needed to Screen by Tier', fontsize=12)
        axes[2].set_xlabel('Risk Tier', fontsize=10)
        axes[2].set_ylabel('Number Needed to Screen', fontsize=10)
        
        # Add text labels
        for i, row in self.tier_stats.iterrows():
            axes[2].text(i, min(row['number_needed_to_screen'] + 0.5, axes[2].get_ylim()[1] * 0.9), 
                       f"{row['number_needed_to_screen']:.1f}", 
                       ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'risk_tier_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved risk tier metrics plot to {save_path}/risk_tier_metrics.png")
    
    def create_clinical_decision_chart(self, window=60, save_path="figures"):
        """Create a clinical decision flowchart showing risk tiers and recommended actions"""
        os.makedirs(save_path, exist_ok=True)
        
        if not hasattr(self, 'tier_stats'):
            print("No tier statistics available. Run define_risk_tiers() first.")
            return
        
        # Get predictions for specified window
        if window not in self.predictions:
            print(f"No predictions available for {window}-day window")
            return
        
        pred_df = self.predictions[window]
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Upper section: Risk tier distribution
        plt.subplot(2, 1, 1)
        
        # Create stacked bar charts showing proportion of true positives and negatives in each tier
        tiers = self.tier_stats['risk_tier'].tolist()
        tier_counts = self.tier_stats['count'].tolist()
        tier_positives = self.tier_stats['positive_count'].tolist()
        tier_negatives = [count - pos for count, pos in zip(tier_counts, tier_positives)]
        
        # Plot as percentage
        tier_total = sum(tier_counts)
        tier_pcts = [count / tier_total * 100 for count in tier_counts]
        
        # Plot stacked bars
        bars = plt.bar(tiers, tier_pcts, color=COLORS[0])
        
        # Add text labels
        for i, (pct, count) in enumerate(zip(tier_pcts, tier_counts)):
            plt.text(i, pct / 2, f"{count}\n({pct:.1f}%)", 
                    ha='center', va='center', color='white', fontweight='bold')
        
        plt.title(f'Risk Tier Distribution ({window}-Day Window)', fontsize=14)
        plt.ylabel('Percentage of Patients', fontsize=12)
        plt.xticks(range(len(tiers)), tiers)
        plt.ylim(0, max(tier_pcts) * 1.1)
        
        # Lower section: Clinical recommendations
        plt.subplot(2, 1, 2)
        
        # Define recommendations by tier
        recommendations = {
            'very low': "• Standard follow-up care\n• Provide educational materials\n• Self-monitoring guidance",
            'low': "• Standard follow-up care\n• Provide educational materials\n• Brief mental health screening at regular visits",
            'medium': "• Enhanced monitoring\n• Consider mental health screening\n• Follow-up within 30 days\n• Provide resources for support",
            'high': "• Priority mental health screening\n• Proactive follow-up within 14 days\n• Consider preventive intervention\n• Detailed assessment of risk factors"
        }
        
        # Create text boxes with recommendations
        n_tiers = len(tiers)
        for i, tier in enumerate(tiers):
            # Calculate position and size
            x_start = i / n_tiers
            x_end = (i + 1) / n_tiers
            width = 0.9 / n_tiers
            
            # Create box
            rect = plt.Rectangle((x_start + 0.05 / n_tiers, 0.1), width, 0.8, 
                               facecolor=COLORS[i], alpha=0.2)
            plt.gca().add_patch(rect)
            
            # Add tier title
            plt.text(x_start + 0.5 / n_tiers, 0.9, f"{tier.upper()} RISK", 
                    ha='center', va='top', fontweight='bold')
            
            # Add risk statistics
            actual_risk = self.tier_stats[self.tier_stats['risk_tier'] == tier]['actual_risk'].values[0]
            nns = self.tier_stats[self.tier_stats['risk_tier'] == tier]['number_needed_to_screen'].values[0]
            
            plt.text(x_start + 0.5 / n_tiers, 0.8, 
                    f"Actual Risk: {actual_risk:.1f}%\nNNS: {nns:.1f}", 
                    ha='center', va='top')
            
            # Add recommendations
            if tier in recommendations:
                plt.text(x_start + 0.5 / n_tiers, 0.65, recommendations[tier], 
                        ha='center', va='top')
        
        plt.title('Clinical Recommendations by Risk Tier', fontsize=14)
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'clinical_decision_chart_{window}d.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved clinical decision chart to {save_path}/clinical_decision_chart_{window}d.png")
    
    def create_risk_stratification_table(self, save_path="tables"):
        """Create a detailed table of risk stratification metrics"""
        os.makedirs(save_path, exist_ok=True)
        
        if not hasattr(self, 'tier_stats'):
            print("No tier statistics available. Run define_risk_tiers() first.")
            return
        
        # Create a formatted table
        table_data = self.tier_stats.copy()
        
        # Format columns
        table_data['percentage'] = table_data['percentage'].apply(lambda x: f"{x:.1f}%")
        table_data['actual_risk'] = table_data['actual_risk'].apply(lambda x: f"{x:.1f}%")
        table_data['precision'] = table_data['precision'].apply(lambda x: f"{x:.3f}")
        table_data['recall'] = table_data['recall'].apply(lambda x: f"{x:.3f}")
        table_data['f1_score'] = table_data['f1_score'].apply(lambda x: f"{x:.3f}")
        table_data['number_needed_to_screen'] = table_data['number_needed_to_screen'].apply(lambda x: f"{x:.1f}")
        
        # Save as CSV
        table_data.to_csv(os.path.join(save_path, 'risk_stratification_metrics.csv'), index=False)
        
        # Create HTML version with styling
        html = """
        <html>
        <head>
            <style>
                body {font-family: Arial, sans-serif;}
                table {border-collapse: collapse; margin: 20px 0;}
                th, td {border: 1px solid #ddd; padding: 8px; text-align: center;}
                th {background-color: #f2f2f2;}
                tr:nth-child(even) {background-color: #f9f9f9;}
                .high {background-color: #FADBD8;}
                .medium {background-color: #FCF3CF;}
                .low {background-color: #D5F5E3;}
                .very-low {background-color: #D6EAF8;}
            </style>
        </head>
        <body>
            <h2>Risk Stratification Metrics</h2>
            <table>
                <tr>
                    <th>Risk Tier</th>
                    <th>Count</th>
                    <th>Percentage</th>
                    <th>Positive Count</th>
                    <th>Actual Risk</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1 Score</th>
                    <th>Number Needed to Screen</th>
                </tr>
        """
        
        for _, row in table_data.iterrows():
            tier_class = row['risk_tier'].lower().replace(' ', '-')
            html += f"""
                <tr class="{tier_class}">
                    <td>{row['risk_tier'].upper()}</td>
                    <td>{row['count']}</td>
                    <td>{row['percentage']}</td>
                    <td>{row['positive_count']}</td>
                    <td>{row['actual_risk']}</td>
                    <td>{row['precision']}</td>
                    <td>{row['recall']}</td>
                    <td>{row['f1_score']}</td>
                    <td>{row['number_needed_to_screen']}</td>
                </tr>
            """
        
        html += """
            </table>
        </body>
        </html>
        """
        
        with open(os.path.join(save_path, 'risk_stratification_metrics.html'), 'w') as f:
            f.write(html)
        
        print(f"Saved risk stratification table to {save_path}/")
    
    def run_analysis(self, window=60, methods=None):
        """Run the full risk stratification analysis"""
        # Set default methods if not provided
        if methods is None:
            methods = ['fixed', 'tercile', 'quartile']
        
        # Load data
        self.load_data()
        
        # Create output directories
        os.makedirs("figures", exist_ok=True)
        os.makedirs("tables", exist_ok=True)
        
        # Analyze each stratification method
        for method in methods:
            # Define risk tiers
            pred_df = self.define_risk_tiers(window=window, method=method)
            
            if pred_df is not None:
                # Plot risk distribution
                self.plot_risk_distribution(pred_df)
                
                # Plot risk tier metrics
                self.plot_risk_tier_metrics()
                
                # Create clinical decision chart
                self.create_clinical_decision_chart(window=window)
                
                # Create risk stratification table
                self.create_risk_stratification_table()
        
        print("\nRisk stratification analysis complete!")


if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Develop risk stratification for TBI prediction')
    parser.add_argument('--window', type=int, default=60, choices=[30, 60, 180, 365],
                        help='Prediction window in days (30, 60, 180, or 365)')
    parser.add_argument('--method', type=str, default='all', 
                        choices=['fixed', 'tercile', 'quartile', 'all'],
                        help='Risk tier definition method')
    parser.add_argument('--predictions_dir', type=str, default='predictions',
                        help='Directory containing prediction files')
    parser.add_argument('--data_path', type=str, default='cleaned_temporal_data.csv',
                        help='Path to original data file')
    
    args = parser.parse_args()
    
    # Setup risk stratification analysis
    analysis = RiskStratification(
        predictions_dir=args.predictions_dir,
        data_path=args.data_path
    )
    
    # Run analysis
    if args.method == 'all':
        analysis.run_analysis(window=args.window)
    else:
        analysis.run_analysis(window=args.window, methods=[args.method])