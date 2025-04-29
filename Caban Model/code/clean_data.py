import pandas as pd
import pickle
from cleaning_functions import *

def main():
    # Configuration
    input_file = r'C:\Users\elija\Desktop\DoD SAFE-n4zvtrvnkUMaN767\Caban Model\enc_db.csv'
    output_file = 'cleaned_temporal_data.csv'
    encoders_file = 'temporal_encoders.pkl'
    
    # Load data
    print("Loading data...")
    df = pd.read_csv(input_file)
    print(f"Initial shape: {df.shape}")
    
    # Store encoders and scalers
    encoders = {}
    
    # Apply cleaning steps
    print("\nApplying cleaning steps...")
    
    print("Cleaning IDs...")
    df = clean_id_fields(df)
    
    print("Cleaning severity...")
    df, severity_encoder = clean_severity(df)
    encoders['severity'] = severity_encoder
    
    print("Cleaning gender...")
    df = clean_gender(df)
    
    print("Cleaning MEPRS codes...")
    df, meprs_encoder = clean_meprs(df)
    encoders['meprs'] = meprs_encoder
    
    print("Cleaning diagnoses...")
    df, diag_encoder = clean_diagnoses(df)
    encoders['diagnoses'] = diag_encoder
    
    print("Cleaning procedure codes...")
    df = clean_procedure_codes(df)
    
    print("Cleaning appointment info...")
    df, appt_encoders = clean_appointment_info(df)
    encoders['appointments'] = appt_encoders
    
    print("Cleaning costs...")
    df, cost_scaler = clean_costs(df)
    encoders['costs'] = cost_scaler
    
    print("Cleaning study periods...")
    df = clean_study_periods(df)
    
    print("Cleaning categorical columns...")
    df, cat_encoders = clean_categorical_columns(df)
    encoders['categorical'] = cat_encoders
    
    print("Cleaning patient category...")
    df = clean_patient_category(df)
    
    print("Cleaning provider skill type...")
    df = clean_provider_skill(df)
    
    print("Adding temporal features...")
    df = add_temporal_features(df)
    
    # Remove unused columns
    print("Removing unused columns...")
    cols_to_drop = ['record_id', 'treatment_id', 'bencatcom', 'fmp']
    df = df.drop(cols_to_drop, axis=1)
    
    # Save cleaned data
    print(f"\nSaving cleaned data to {output_file}")
    df.to_csv(output_file, index=False)
    print(f"Final shape: {df.shape}")
    
    # Save encoders
    print(f"Saving encoders to {encoders_file}")
    with open(encoders_file, 'wb') as f:
        pickle.dump(encoders, f)
    
    # Print summary
    print("\nSummary:")
    print(f"Total patients: {df['patient_id'].nunique()}")
    print(f"Total encounters: {len(df)}")
    print(f"Mental health diagnoses: {df['has_mh_diagnosis'].sum()} ({df['has_mh_diagnosis'].mean():.1%} of encounters)")
    
    # Verify no string columns remain
    str_cols = df.select_dtypes(include=['object']).columns
    if len(str_cols) > 0:
        print("\nWarning: String columns remain:")
        print(str_cols)
    else:
        print("\nAll columns converted to numeric!")

if __name__ == "__main__":
    main()