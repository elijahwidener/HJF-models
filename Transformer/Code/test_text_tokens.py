import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import json
from encounter_to_tokens import encounter_to_text, process_patient_cohort_incremental, analyze_token_vocabulary

def test_text_tokenization():
    """
    Test the text-based tokenization of patient encounters
    """
    # Path to the data
    data_path = r"C:\Users\elija\Desktop\DoD SAFE-n4zvtrvnkUMaN767\Caban Model\code\enc_db.csv"
    
    # Create output directory
    os.makedirs("text_token_analysis", exist_ok=True)
    
    # Load a subset of the data
    print("Loading subset of encounter data...")
    start_time = time.time()
    
    # Read a small subset for initial testing
    df = pd.read_csv(data_path)

    
    print(f"Loaded {len(df)} encounters from {df['pseudo_personid'].nunique()} patients")
    print(f"Time taken: {time.time() - start_time:.2f} seconds\n")
    
    # Process the patient cohort
    print("\nProcessing patient cohort...")
    start_time = time.time()
    patient_data, outcomes = process_patient_cohort_incremental(df, output_dir="text_token_analysis")
    print(f"Processing complete. Time taken: {time.time() - start_time:.2f} seconds")
    
    # Analyze the token vocabulary
    print("\nAnalyzing token vocabulary...")
    all_tokens, token_freq = analyze_token_vocabulary(patient_data, output_dir="text_token_analysis")
    
    # Visualize token distribution
    visualize_token_statistics(patient_data, all_tokens, token_freq)
    
    # Visualize sequence lengths
    visualize_sequence_lengths(patient_data)
    
    # Print sample sequences
    print_sample_sequences(patient_data)
    
    return patient_data, all_tokens, token_freq

def visualize_token_statistics(patient_data, all_tokens, token_freq):
    """
    Create visualizations of token statistics
    """
    # Token frequency distribution (top 20)
    sorted_tokens = sorted(token_freq.items(), key=lambda x: x[1], reverse=True)
    top_tokens = sorted_tokens[:20]
    
    plt.figure(figsize=(12, 8))
    plt.bar([t[0] for t in top_tokens], [t[1] for t in top_tokens])
    plt.title('Top 20 Most Common Tokens')
    plt.xlabel('Token')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('text_token_analysis/top_tokens.png')
    plt.close()
    
    # Token type distribution
    token_types = {}
    for token in all_tokens:
        prefix = token.split('_')[0]
        token_types[prefix] = token_types.get(prefix, 0) + 1
    
    plt.figure(figsize=(10, 6))
    plt.bar(token_types.keys(), token_types.values())
    plt.title('Token Type Distribution')
    plt.xlabel('Token Type')
    plt.ylabel('Number of Unique Tokens')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('text_token_analysis/token_types.png')
    plt.close()
    
    # Token frequency by type
    type_freq = {}
    for token, freq in token_freq.items():
        prefix = token.split('_')[0]
        type_freq[prefix] = type_freq.get(prefix, 0) + freq
    
    plt.figure(figsize=(10, 6))
    plt.bar(type_freq.keys(), type_freq.values())
    plt.title('Token Frequency by Type')
    plt.xlabel('Token Type')
    plt.ylabel('Total Frequency')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('text_token_analysis/token_type_frequency.png')
    plt.close()

def visualize_sequence_lengths(patient_data):
    """
    Visualize the distribution of sequence lengths
    """
    sequence_lengths = [len(patient['pre_tbi_tokens']) for patient in patient_data.values()]
    tokens_per_encounter = []
    
    for patient in patient_data.values():
        for encounter in patient['pre_tbi_tokens']:
            tokens_per_encounter.append(len(encounter.split()))
    
    # Plot sequence lengths
    plt.figure(figsize=(10, 6))
    plt.hist(sequence_lengths, bins=max(10, min(10, max(sequence_lengths))), edgecolor='black')
    plt.title('Distribution of Sequence Lengths (Encounters per Patient)')
    plt.xlabel('Number of Encounters')
    plt.ylabel('Number of Patients')
    plt.grid(True, alpha=0.3)
    plt.savefig('text_token_analysis/sequence_lengths.png')
    plt.close()
    
    # Plot tokens per encounter
    plt.figure(figsize=(10, 6))
    plt.hist(tokens_per_encounter, bins=10, edgecolor='black')
    plt.title('Distribution of Tokens per Encounter')
    plt.xlabel('Number of Tokens')
    plt.ylabel('Number of Encounters')
    plt.grid(True, alpha=0.3)
    plt.savefig('text_token_analysis/tokens_per_encounter.png')
    plt.close()

def print_sample_sequences(patient_data):
    """
    Print sample token sequences for a few patients
    """
    # Save sample sequences to file
    with open('text_token_analysis/sample_sequences.txt', 'w') as f:
        # Take 3 random patients
        sample_patients = list(patient_data.keys())[:3]
        
        for i, patient_id in enumerate(sample_patients):
            patient = patient_data[patient_id]
            f.write(f"Patient {i+1} (ID: {patient_id}):\n")
            f.write(f"Number of encounters: {len(patient['pre_tbi_tokens'])}\n\n")
            
            # Print first 5 encounters (or all if less than 5)
            f.write("First encounters:\n")
            for j, encounter in enumerate(patient['pre_tbi_tokens'][:5]):
                f.write(f"  Encounter {j+1}: {encounter}\n\n")
            
            # Print last encounter
            if len(patient['pre_tbi_tokens']) > 5:
                f.write("Last encounter:\n")
                f.write(f"  Encounter {len(patient['pre_tbi_tokens'])}: {patient['pre_tbi_tokens'][-1]}\n\n")
            
            f.write("\n" + "-"*80 + "\n\n")
    
    print("\nSample sequences saved to text_token_analysis/sample_sequences.txt")

if __name__ == "__main__":
    patient_data, all_tokens, token_freq = test_text_tokenization()