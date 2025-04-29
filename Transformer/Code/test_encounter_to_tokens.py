import pandas as pd
from encounter_to_tokens import encounter_to_text

def print_sample_tokenized_encounters(csv_path, num_samples=5):
    # Load the data
    df = pd.read_csv(csv_path)
    
    print(f"Loaded {len(df)} encounters from {df['pseudo_personid'].nunique()} patients.\n")

    # Print tokenized sample encounters
    for i in range(min(num_samples, len(df))):
        encounter = df.iloc[i]
        tokenized = encounter_to_text(encounter)
        print(f"Encounter {i+1} (Patient ID: {encounter['pseudo_personid']}):")
        print(tokenized)
        print("-" * 60)

if __name__ == "__main__":
    csv_path = r"C:\Users\elija\Desktop\DoD SAFE-n4zvtrvnkUMaN767\Caban Model\code\enc_db.csv"
    print_sample_tokenized_encounters(csv_path, num_samples=5)
