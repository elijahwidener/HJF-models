import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import Counter

def validate_data_distribution(processed_data, title="Dataset Distribution"):
    """
    Validate that the dataset contains both positive and negative examples
    for each time window.
    
    Args:
        processed_data: List of processed patient data
        title: Title for the analysis output
    """
    windows = [30, 60, 180, 365]
    total_examples = len(processed_data)
    
    if total_examples == 0:
        print(f"\n{title}: No examples found.")
        return {}
    
    results = {}
    
    print(f"\n{title} (Total: {total_examples} examples):")
    print("-" * 60)
    
    for window in windows:
        positives = sum(1 for patient in processed_data if patient['outcomes'][window] == 1)
        negatives = total_examples - positives
        pos_percent = (positives / total_examples) * 100 if total_examples > 0 else 0
        
        print(f"{window}-day window:")
        print(f"  Positive cases: {positives} ({pos_percent:.1f}%)")
        print(f"  Negative cases: {negatives} ({100-pos_percent:.1f}%)")
        
        results[window] = {
            'positive': positives,
            'negative': negatives,
            'pos_percent': pos_percent
        }
    
    print("-" * 60)
    return results

def stratified_dataset_split(processed_data, test_size=0.2, val_size=0.1, stratify_window=60):
    """
    Split the dataset into train, validation, and test sets using stratified sampling
    
    Args:
        processed_data: List of processed patient data
        test_size: Fraction of data to use for testing
        val_size: Fraction of data to use for validation
        stratify_window: Which time window to use for stratification
        
    Returns:
        train_data, val_data, test_data: Split datasets
    """
    # Extract labels for stratification
    stratify_labels = [patient['outcomes'][stratify_window] for patient in processed_data]
    
    # Check if we have both classes
    unique_labels = set(stratify_labels)
    if len(unique_labels) < 2:
        print(f"Warning: Only one class present in the {stratify_window}-day window.")
        print("Falling back to random (non-stratified) split.")
        stratify_labels = None
    
    # First split into train+val and test
    train_val, test = train_test_split(
        processed_data, 
        test_size=test_size, 
        random_state=42,
        stratify=stratify_labels
    )
    
    # Extract stratification labels for the train_val split
    if stratify_labels is not None:
        train_val_labels = [patient['outcomes'][stratify_window] for patient in train_val]
        
        # Verify we still have both classes
        unique_train_val_labels = set(train_val_labels)
        if len(unique_train_val_labels) < 2:
            print(f"Warning: Only one class present in train+val for {stratify_window}-day window.")
            print("Falling back to random split for train/val.")
            train_val_labels = None
    else:
        train_val_labels = None
    
    # Then split train+val into train and val
    val_size_adjusted = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, 
        test_size=val_size_adjusted, 
        random_state=42,
        stratify=train_val_labels
    )
    
    print(f"Dataset split: {len(train)} train, {len(val)} validation, {len(test)} test")
    
    return train, val, test

def plot_class_distribution(distribution_results, output_dir="analysis"):
    """
    Plot the class distribution for different time windows
    
    Args:
        distribution_results: Results from validate_data_distribution
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    windows = list(distribution_results.keys())
    positives = [distribution_results[w]['positive'] for w in windows]
    negatives = [distribution_results[w]['negative'] for w in windows]
    
    # Create stacked bar chart
    plt.figure(figsize=(10, 6))
    
    bar_width = 0.6
    r = range(len(windows))
    
    # Plot bars
    plt.bar(r, negatives, color='#3498db', width=bar_width, label='Negative')
    plt.bar(r, positives, bottom=negatives, color='#e74c3c', width=bar_width, label='Positive')
    
    # Add percentage labels
    total = np.array(positives) + np.array(negatives)
    for i, (pos, neg) in enumerate(zip(positives, negatives)):
        # Label for positives (on top of the stack)
        plt.text(i, neg + pos/2, f"{pos}\n({100*pos/total[i]:.1f}%)", 
                 ha='center', va='center', color='white', fontweight='bold')
        
        # Label for negatives (at bottom of stack)
        plt.text(i, neg/2, f"{neg}\n({100*neg/total[i]:.1f}%)", 
                 ha='center', va='center', color='white', fontweight='bold')
    
    # Add labels and title
    plt.xlabel('Time Window (days)')
    plt.ylabel('Number of Patients')
    plt.title('Class Distribution by Time Window')
    plt.xticks(r, [f"{w} days" for w in windows])
    plt.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/class_distribution.png", dpi=300)
    plt.close()
    
    # Plot positive percentage trend
    plt.figure(figsize=(10, 5))
    
    pos_percent = [distribution_results[w]['pos_percent'] for w in windows]
    
    plt.plot(windows, pos_percent, marker='o', linestyle='-', linewidth=2)
    
    # Add data labels
    for i, pct in enumerate(pos_percent):
        plt.text(windows[i], pct+1, f"{pct:.1f}%", ha='center')
    
    plt.title('Positive Rate by Time Window')
    plt.xlabel('Time Window (days)')
    plt.ylabel('Positive Rate (%)')
    plt.grid(alpha=0.3)
    plt.ylim(0, max(pos_percent) * 1.2)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/positive_rate_trend.png", dpi=300)
    plt.close()

def validate_encounter_counts(processed_data, output_dir="analysis"):
    """
    Analyze the distribution of encounter counts per patient
    
    Args:
        processed_data: List of processed patient data
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Count encounters per patient
    encounter_counts = [len(patient['encounters']) for patient in processed_data]
    
    # Basic statistics
    avg_encounters = np.mean(encounter_counts)
    median_encounters = np.median(encounter_counts)
    max_encounters = max(encounter_counts)
    min_encounters = min(encounter_counts)
    
    print("\nEncounter Distribution:")
    print(f"  Average encounters per patient: {avg_encounters:.1f}")
    print(f"  Median encounters per patient: {median_encounters:.1f}")
    print(f"  Range: {min_encounters} to {max_encounters} encounters")
    
    # Create histogram
    plt.figure(figsize=(10, 6))
    
    bins = min(30, max(10, int(max_encounters/5)))
    plt.hist(encounter_counts, bins=bins, color='#2980b9', alpha=0.8, edgecolor='black')
    
    plt.axvline(avg_encounters, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {avg_encounters:.1f}')
    plt.axvline(median_encounters, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_encounters:.1f}')
    
    plt.title('Distribution of Encounter Counts per Patient')
    plt.xlabel('Number of Encounters')
    plt.ylabel('Number of Patients')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/encounter_distribution.png", dpi=300)
    plt.close()
    
    return {
        'avg_encounters': avg_encounters,
        'median_encounters': median_encounters,
        'max_encounters': max_encounters,
        'min_encounters': min_encounters
    }

def validate_days_from_tbi(processed_data, output_dir="analysis"):
    """
    Analyze the distribution of days from TBI
    
    Args:
        processed_data: List of processed patient data
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect all days from TBI values
    all_days = []
    for patient in processed_data:
        all_days.extend(patient['days_from_tbi'])
    
    # Basic statistics
    avg_days = np.mean(all_days)
    median_days = np.median(all_days)
    max_days = max(all_days)
    min_days = min(all_days)
    
    print("\nDays from TBI Distribution:")
    print(f"  Average days from TBI: {avg_days:.1f}")
    print(f"  Median days from TBI: {median_days:.1f}")
    print(f"  Range: {min_days} to {max_days} days")
    
    # Create histogram
    plt.figure(figsize=(10, 6))
    
    bins = min(30, max(10, int(abs(min_days)/10)))
    plt.hist(all_days, bins=bins, color='#8e44ad', alpha=0.8, edgecolor='black')
    
    plt.axvline(0, color='black', linestyle='solid', linewidth=2, label='TBI Day')
    plt.axvline(avg_days, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {avg_days:.1f}')
    plt.axvline(median_days, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_days:.1f}')
    
    plt.title('Distribution of Days from TBI')
    plt.xlabel('Days from TBI (negative = before TBI)')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/days_from_tbi.png", dpi=300)
    plt.close()
    
    return {
        'avg_days': avg_days,
        'median_days': median_days,
        'max_days': max_days,
        'min_days': min_days
    }

def analyze_dataset(processed_data, output_dir="analysis"):
    """
    Run a comprehensive analysis of the dataset to validate its quality and characteristics.
    
    This function performs multiple analytical checks on the prepared dataset:
    1. Class distribution analysis across different time windows
    2. Encounter count distribution analysis
    3. Days-from-TBI distribution analysis
    4. Stratified dataset splitting with validation of split quality
    5. Generation of visualizations for key distributions
    
    The analysis results are saved both as JSON statistics and as visualization plots,
    providing a comprehensive data quality report that can help identify potential
    issues before model training.
    
    Args:
        processed_data (list): List of processed patient data dictionaries
        output_dir (str): Directory to save analysis results and visualizations
        
    Returns:
        tuple: (train_data, val_data, test_data)
            - train_data: Training data subset
            - val_data: Validation data subset
            - test_data: Test data subset
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Check overall data size
    total_patients = len(processed_data)
    print(f"\nAnalyzing dataset with {total_patients} patients")
    
    # Validate class distribution
    dist_results = validate_data_distribution(processed_data)
    
    # Analyze encounter counts
    encounter_stats = validate_encounter_counts(processed_data, output_dir)
    
    # Analyze days from TBI
    days_stats = validate_days_from_tbi(processed_data, output_dir)
    
    # Plot distributions
    if dist_results:
        plot_class_distribution(dist_results, output_dir)
    
    # Prepare split
    print("\nPreparing stratified split:")
    train_data, val_data, test_data = stratified_dataset_split(processed_data)
    
    # Validate splits
    print("\nTraining set:")
    train_dist = validate_data_distribution(train_data, "Training Set Distribution")
    
    print("\nValidation set:")
    val_dist = validate_data_distribution(val_data, "Validation Set Distribution")
    
    print("\nTest set:")
    test_dist = validate_data_distribution(test_data, "Test Set Distribution")
    
    # Save split sizes
    split_info = {
        'total': total_patients,
        'train': len(train_data),
        'validation': len(val_data),
        'test': len(test_data)
    }
    
    # Combine all stats
    all_stats = {
        'split_info': split_info,
        'encounter_stats': encounter_stats,
        'days_stats': days_stats,
        'class_distribution': {
            'overall': dist_results,
            'train': train_dist,
            'validation': val_dist,
            'test': test_dist
        }
    }
    
    # Save stats to JSON
    with open(f"{output_dir}/dataset_stats.json", 'w') as f:
        # Convert numpy values to Python types for JSON serialization
        json_stats = json.dumps(all_stats, default=lambda o: float(o) if isinstance(o, (np.float32, np.float64)) else o)
        f.write(json_stats)
    
    print(f"\nAnalysis complete. Results saved to {output_dir}/")
    
    return train_data, val_data, test_data

if __name__ == "__main__":
    # Example usage
    from data_utils import load_patient_data, prepare_encounter_data
    
    print("Loading patient data...")
    patient_data, outcomes = load_patient_data()
    
    print("Preparing encounter data...")
    processed_data = prepare_encounter_data(patient_data, outcomes)
    
    print("Analyzing dataset...")
    train_data, val_data, test_data = analyze_dataset(processed_data)
    
    print("Analysis complete!")