import pandas as pd
import numpy as np
from datetime import datetime
import json
import os

# Load mappings once
with open("C:/Users/elija/Desktop/DoD SAFE-n4zvtrvnkUMaN767/Transformer/IDC_codes/diagnosis_map.json", "r") as f:
    diagnosis_map = json.load(f)

with open("C:/Users/elija/Desktop/DoD SAFE-n4zvtrvnkUMaN767/Transformer/IDC_codes/category_map.json", "r") as f:
    category_map = json.load(f)

with open("C:/Users/elija/Desktop/DoD SAFE-n4zvtrvnkUMaN767/Transformer/provider_spec_codes/provider_spec_map.json", "r") as f:
    provider_spec_map = json.load(f)

# Load and prepare DMIS dict
dmis_mapping = pd.read_excel("C:/Users/elija/Desktop/DoD SAFE-n4zvtrvnkUMaN767/Transformer/202505_dmisid.xlsx", usecols=[
    'DMIS ID', 
    'Facility Type Code (6-char maximum)', 
    'DMIS Facility Name (30-character maximum)'
])
dmis_dict = dmis_mapping.set_index('DMIS ID')[
    ['Facility Type Code (6-char maximum)', 'DMIS Facility Name (30-character maximum)']
].to_dict(orient='index')

def encounter_to_text(encounter, diagnosis_map, category_map, provider_spec_map, dmis_dict):
    """
    Convert a single medical encounter record to a text representation suitable for transformer models.
    
    This function transforms structured encounter data into a space-separated sequence of tokens that 
    preserves critical clinical information while standardizing the format for NLP processing.
    Each piece of information is prefixed with its category (e.g., "DIAG1_DESC_" for primary diagnosis).
    
    Args:
        encounter (dict): Dictionary containing the encounter data fields
        diagnosis_map (dict): Mapping from diagnosis codes to human-readable descriptions
        category_map (dict): Mapping from diagnosis code first character to diagnosis category
        provider_spec_map (dict): Mapping from provider specialty codes to specialty descriptions
        dmis_dict (dict): Mapping from DMIS IDs to facility information
        
    Returns:
        str: Space-separated sequence of tokens representing the encounter
    
    Example output:
        "DATE_2023-05-15 AGE_45 GENDER_Male FACILITY_NAME_Walter_Reed DIAG1_CAT_Mental_Disorders 
         DIAG1_DESC_Anxiety_Disorder PROVIDER_SPEC_Psychiatrist"
    """
    tokens = []
    
    # Patient ID not needed, will create noise
    
    # Encounter date - tokenize both calendar features and date itself
    if 'servicedate' in encounter and pd.notna(encounter['servicedate']):
        date = pd.to_datetime(encounter['servicedate'])
        tokens.append(f"DATE_{date.strftime('%Y-%m-%d')}")

    # TBI Index Date
    if 'tbi_index_date' in encounter and pd.notna(encounter['tbi_index_date']):
        tbi_date = pd.to_datetime(encounter['tbi_index_date'])
        tokens.append(f"TBI_DATE_{tbi_date.strftime('%Y-%m-%d')}")
       
    # Basic demographic tokens
    if 'age' in encounter and pd.notna(encounter['age']):
        tokens.append(f"AGE_{int(encounter['age'])}")
    
    if 'gender' in encounter and pd.notna(encounter['gender']):
        gender_mapping = {
            'M': 'Male',
            'F': 'Female'
        }
        gender = gender_mapping.get(encounter['gender'], 'Other')
        tokens.append(f"GENDER_{gender}")
    
    if 'race' in encounter and pd.notna(encounter['race']):
        race_mapping = {
            'C': 'White',
            'M': 'Asian_or_Pacific_Islander',
            'N': 'Black',
            'R': 'American_Indian_or_Alaskan_Native',
            'X': 'Other',
            'Z': 'Unknown'
        }
        race = race_mapping.get(encounter['race'], 'Unknown')
        tokens.append(f"RACE_{race}")
    
    if 'severity' in encounter and pd.notna(encounter['severity']):
        tokens.append(f"SEVERITY_{encounter['severity']}")

   
    # # Temporal information
    # if 'days_from_tbi' in encounter and pd.notna(encounter['days_from_tbi']):
    #     # Bin days to reduce token space
    #     days = int(encounter['days_from_tbi'])
    #     if days <= -365:
    #         time_token = "TIME_OVER_1YR_BEFORE_TBI"
    #     elif days <= -180:
    #         time_token = "TIME_6-MO_BEFORE_TBI"
    #     elif days <= -90:
    #         time_token = "TIME_3-MO_BEFORE_TBI"
    #     elif days <= -30:
    #         time_token = "TIME_1-MO_BEFORE_TBI"
    #     elif days <= -7:
    #         time_token = "TIME_1-WK_BEFORE_TBI"
    #     else:
    #         time_token = "TIME_LAST_WEEK_BEFORE_TBI"
    #     tokens.append(time_token)
        
    #     # Also add exact days for precision
    #     tokens.append(f"DAYS_{days}")
    
    # Study period indicators
    for period in [90, 180, 270, 365]:
        study_key = f'study{period}day'
        if study_key in encounter and pd.notna(encounter[study_key]):
            tokens.append(f"STUDY_{period}DAY_{encounter[study_key]}")
            
        encounter_key = f'study{period}day_encounter'
        if encounter_key in encounter and pd.notna(encounter[encounter_key]):
            tokens.append(f"STUDY_{period}DAY_ENC_{encounter[encounter_key]}")
    
    # Defense Medical Information System Identifier       TODO: look at this more closely
    if 'tmt_dmisid' in encounter and pd.notna(encounter['tmt_dmisid']):
        try:
                # Find the matching DMIS ID
                dmis_id = encounter['tmt_dmisid']
                match = dmis_dict.get(dmis_id, {})
                
                if match:
                    facility_name = match.get('DMIS Facility Name (30-character maximum)', 'Unknown')
                    facility_type_code = match.get('Facility Type Code (6-char maximum)', 'Unknown')
                    tokens.append(f"FACILITY_NAME_{facility_name[:30]}")
                    tokens.append(f"FACILITY_TYPE_{facility_type_code}")
                else:
                    # No match found, just use the ID
                    tokens.append(f"FACILITY_ID_{dmis_id}")
        except Exception as e:
            # If any error occurs during lookup, just use the ID
            tokens.append(f"FACILITY_ID_{encounter['tmt_dmisid']}")
            print(f"Warning: Error processing DMIS ID {encounter['tmt_dmisid']}: {e}")
   

    for i in range(1, 6):
        diag_key = f'diag{i}'
        if diag_key in encounter and pd.notna(encounter[diag_key]) and encounter[diag_key] != "":
            # For diagnoses, add both the exact code and the category (first letter)
            code = encounter[diag_key]
            if len(code) > 0:
                category = category_map.get(code[0], "Unknown_Category")
                tokens.append(f"DIAG{i}_CAT_{category}")
                # Use the mapping for actual diagnosis, default to 'Unknown'
                diagnosis_description = diagnosis_map.get(code, "Unknown")
                tokens.append(f"DIAG{i}_DESC_{diagnosis_description}")
    
    # Procedures
    for i in range(1, 6):
        proc_key = f'proc{i}'
        if proc_key in encounter and pd.notna(encounter[proc_key]) and encounter[proc_key] != "":
            tokens.append(f"PROC{i}_{encounter[proc_key]}")
    
    # EM code (different from procedures)
    if 'em1' in encounter and pd.notna(encounter['em1']) and encounter['em1'] != "":
        tokens.append(f"EM_{encounter['em1']}")
    
    # Cost information - combine cost category and value in the same token
    cost_features = {
        'fullcost': 'TOTAL_COST',
        'fullcostclinsal': 'CLINICAL_SALARY_COST',
        'fullcostlab': 'LAB_COST',
        'fullcostother': 'OTHER_COST',
        'fullcostotheranc': 'ANC_COST',
        'fullcostpharm': 'PHARMACY_COST',
        'fullcostprofsal': 'PROF_SALARY_COST',
        'fullcostrad': 'RAD_COST',
        'fullcostsupport': 'SUPPORT_COST'
    }
    
    for feature, prefix in cost_features.items():
        if feature in encounter and pd.notna(encounter[feature]):
            value = float(encounter[feature])
            # Bin costs into categories
            if value == 0:
                cost_bin = "ZERO"
            elif value < 10:
                cost_bin = "VERY_LOW"
            elif value < 50:
                cost_bin = "LOW"
            elif value < 200:
                cost_bin = "MEDIUM"
            elif value < 1000:
                cost_bin = "HIGH"
            else:
                cost_bin = "VERY_HIGH"
            tokens.append(f"{prefix}: {cost_bin}")
    
    # Military-specific data
    if 'sponrankgrp' in encounter and pd.notna(encounter['sponrankgrp']):
        rank_mapping = {
            'EJ': 'Enlisted Junior (E1–E4)',
            'ES': 'Enlisted Senior (E5–E9)',
            'OJ': 'Officer Junior (O1–O3)',
            'OS': 'Officer Senior (O4–O10)',
            'WO': 'Warrant Officer (W1–W5)',
            'CD': 'Cadet (Academy students, ROTC)',
            'XX': 'Unknown or Not Reported'
        }
        rank = rank_mapping.get(encounter['sponrankgrp'], 'Unknown')
        tokens.append(f"RANK_{rank}")
    
    if 'sponservice' in encounter and pd.notna(encounter['sponservice']):
        service_mapping = {
            'A': 'Army',
            'C': 'Coast Guard',
            'E': 'Military Entrance Processing',
            'F': 'Air Force',
            'J': 'Joint Military Organization',
            'M': 'Managed Care Support Contractor',
            'N': 'Navy',
            'O': 'Other Government',
            'P': 'Defense Health Agency',
            'R': 'Pharmacy Operations Division',
            'S': 'Noncatchment Area or Navy Afloat Area',
            'T': 'Uniformed Services Family Health Plan',
            'V': 'Veterans Administration',
            'X': 'Not Applicable'
        }
        service = service_mapping.get(encounter['sponservice'], 'Unknown')
        tokens.append(f"SERVICE_{service}")
      
    if 'prodline' in encounter and pd.notna(encounter['prodline']):
        prodline_mapping = {
            'PC': 'Primary_Care',
            'ORTHO': 'Orthopedics',
            'MH': 'Mental_Health',
            'OBGYN': 'Obstetrics_Gynecology',
            'OPTOM': 'Optometry',
            'IMSUB': 'Internal_Medicine_Subspecialty',
            'ER': 'Emergency_Room',
            'SURG': 'General_Surgery',
            'SURGSUB': 'Surgical_Subspecialty',
            'ENT': 'Otolaryngology',
            'DERM': 'Dermatology',
            'OTHER': 'Other'
        }
        prodline_category = prodline_mapping.get(encounter['prodline'], 'Unknown')
        tokens.append(f"PRODLINE_{prodline_category}")
    
    if 'appttype' in encounter and pd.notna(encounter['appttype']):
        appttype_mapping = {
            '24HR': '24-Hour Care',
            'SPEC': 'Specialty',
            'PROC': 'Procedure',
            'FTR': 'Follow-Up',
            'ACUT': 'Acute Care',
            'WELL': 'Wellness',
            'T-CON*': 'Telephone Consultation',
            'ROUT$': 'Routine (Billing Modifier)',
            'APV': 'Appointment Visit',
            'ROUT': 'Routine',
            'EROOM': 'Emergency Room',
            'GRP': 'Group',
            'ACUT$': 'Acute Care (Billing Modifier)',
            'VIRT': 'Virtual',
            'FTR$': 'Follow-Up (Billing Modifier)',
            'RNDS*': 'Rounds',
            'SPEC$': 'Specialty (Billing Modifier)',
            'GRP$': 'Group (Billing Modifier)',
            'WELL$': 'Wellness (Billing Modifier)',
            'PROC$': 'Procedure (Billing Modifier)',
            'PCM': 'Primary Care Management',
            '24HR$': '24-Hour Care (Billing Modifier)',
            'NOMAP': 'No Mapping',
            'PCM$': 'Primary Care Management (Billing Modifier)',
            'N-MTF': 'Non-Military Treatment Facility',
            'RAD*': 'Radiology',
            'DROUT': 'Discharge/Outpatient'
        }
        appttype_category = appttype_mapping.get(encounter['appttype'], 'Unknown')
        tokens.append(f"APPT_TYPE_{appttype_category}")
    
    # Provider information
    if 'apptprovspec' in encounter and pd.notna(encounter['apptprovspec']):
        # Map the provider specialization code to its description
        provider_spec_code = encounter['apptprovspec']
        provider_spec_desc = provider_spec_map.get(provider_spec_code, "Unknown")
        tokens.append(f"PROVIDER_SPEC_{provider_spec_desc}")
           
    # Provider work RVU
    if 'provwrkrvuapptprov' in encounter and pd.notna(encounter['provwrkrvuapptprov']):
        value = float(encounter['provwrkrvuapptprov'])
        tokens.append(f"PROVIDER_RVU_{value:.2f}")
    
    # Provider skill type
    if 'provskilltype_appt' in encounter and pd.notna(encounter['provskilltype_appt']):
        skill_mapping = {
            '1': 'Clinician',
            '2': 'Direct_Care_Professional',
            '3': 'Registered_Nurse',
            '4': 'Direct_Care_Para_Prof',
            '5': 'Admin_Clerical',
            'N': 'Not_FTE_Expense'
        }
        skill_type = skill_mapping.get(encounter['provskilltype_appt'], 'Unknown')
        tokens.append(f"PROVIDER_SKILL_TYPE_{skill_type}")
    
    # Appointment status
    if 'apptstatcode' in encounter and pd.notna(encounter['apptstatcode']):
        appt_status_mapping = {
            '1': 'Pending',
            '2': 'Kept',
            '3': 'Cancellation',
            '4': 'No-Show',
            '5': 'Walk-in',
            '6': 'Sick Call',
            '7': 'Telephone Consult',
            '8': 'Left Without Being Seen',
            '9': 'Admin',
            '10': 'Occasion of Service',
            '12': 'Booked'
        }
        appt_status = appt_status_mapping.get(str(encounter['apptstatcode']), 'Unknown')
        tokens.append(f"APPT_STATUS_{appt_status}")
    
    # Patient category
    if 'patientcat' in encounter and pd.notna(encounter['patientcat']):
        tokens.append(f"PAT_CAT_{encounter['patientcat']}")
    
    return " ".join(tokens)

def convert_patient_encounters(df, patient_id):
    """
    Convert all encounters for a specific patient to token sequences.
    
    Args:
        df: DataFrame containing encounters
        patient_id: ID of the patient to convert
        
    Returns:
        Tuple of (pre-TBI tokens, post-TBI tokens by window)
    """
    # Convert dates
    df['servicedate'] = pd.to_datetime(df['servicedate'])
    df['tbi_index_date'] = pd.to_datetime(df['tbi_index_date'])
    
    # Calculate days from TBI
    df['days_from_tbi'] = (df['servicedate'] - df['tbi_index_date']).dt.days
    
    # Filter to the specific patient
    patient_df = df[df['pseudo_personid'] == patient_id].copy()
    
    # Sort by date
    patient_df = patient_df.sort_values('days_from_tbi')
    
    # Split into pre and post TBI
    pre_tbi = patient_df[patient_df['days_from_tbi'] <= 0]
    post_tbi = patient_df[patient_df['days_from_tbi'] > 0]
    
    # Convert pre-TBI encounters to tokens
    pre_tbi_tokens = []
    for _, encounter in pre_tbi.iterrows():
        tokens = encounter_to_text(encounter, diagnosis_map, category_map, provider_spec_map, dmis_dict)
        pre_tbi_tokens.append(tokens)
    
    # Convert post-TBI encounters by window
    post_tbi_windows = {}
    for window in [30, 60, 180, 365]:
        window_df = post_tbi[post_tbi['days_from_tbi'] <= window]
        window_tokens = []
        for _, encounter in window_df.iterrows():
            tokens = encounter_to_text(encounter, diagnosis_map, category_map, provider_spec_map, dmis_dict)
            window_tokens.append(tokens)
        post_tbi_windows[window] = window_tokens
    
    return pre_tbi_tokens, post_tbi_windows

def process_patient_cohort_incremental(df, output_dir="token_data", checkpoint_every=100):
    """
    Process a cohort of patients incrementally with checkpointing to handle large datasets.
    
    This function converts raw encounter data to tokenized format for all patients in a cohort,
    with built-in checkpointing to allow for processing very large datasets without memory issues
    or having to restart from scratch in case of interruption.
    
    The function also determines mental health outcomes for different time windows post-TBI
    by looking for diagnosis codes starting with 'F' (mental health codes in ICD-10).
    
    Key steps:
    1. Loads existing progress if available to resume processing
    2. Processes each patient's encounters to generate tokenized sequences
    3. Determines mental health outcomes for 30, 60, 180, and 365-day windows
    4. Periodically saves checkpoints with processed data
    5. Generates summary statistics upon completion
    
    Args:
        df (DataFrame): DataFrame containing encounter records with columns:
            - pseudo_personid: Patient identifier
            - servicedate: Date of the encounter
            - tbi_index_date: Date of the TBI event
            - diag1-diag5: Diagnosis codes
        output_dir (str): Directory to save the processed data
        checkpoint_every (int): Save checkpoint after processing this many patients
        
    Returns:
        tuple: (patient_data, outcomes)
            - patient_data: Dictionary mapping patient IDs to tokenized encounters
            - outcomes: Dictionary of outcomes by window with (patient_id, has_mh) tuples
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check for existing progress file
    progress_file = f"{output_dir}/processing_progress.json"
    
    processed_patient_ids = set()
    patient_data = {}
    outcomes = {30: [], 60: [], 180: [], 365: []}
    
    # If progress file exists, load it
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            progress = json.load(f)
            processed_patient_ids = set(progress['processed_patients'])
            
        # Load existing patient data
        patient_data_file = f"{output_dir}/patient_tokens_partial.json"
        if os.path.exists(patient_data_file):
            with open(patient_data_file, 'r') as f:
                patient_data = json.load(f)
                
        # Load existing outcomes
        for window in [30, 60, 180, 365]:
            outcome_file = f"{output_dir}/outcomes_{window}day_partial.csv"
            if os.path.exists(outcome_file):
                outcomes_df = pd.read_csv(outcome_file)
                outcomes[window] = [(row['patient_id'], row['has_mh']) 
                                   for _, row in outcomes_df.iterrows()]
                
        print(f"Resuming from {len(processed_patient_ids)} already processed patients")
    
    # Get unique patients that haven't been processed yet
    all_patient_ids = df['pseudo_personid'].unique()
    patients_to_process = [pid for pid in all_patient_ids if str(pid) not in processed_patient_ids]
    
    print(f"Processing {len(patients_to_process)} patients...")
    
    # Process each patient
    for i, patient_id in enumerate(patients_to_process):
        
        # Convert patient encounters to tokens
        pre_tbi_tokens, post_tbi_windows = convert_patient_encounters(df, patient_id)
        
        # Skip patients with no pre-TBI data
        if len(pre_tbi_tokens) == 0:
            continue
        
        # Store patient data
        patient_data[str(patient_id)] = {
            'pre_tbi_tokens': pre_tbi_tokens,
            'post_tbi_tokens': post_tbi_windows
        }
        
        # Determine mental health outcomes
        for window in [30, 60, 180, 365]:
            # Check if any encounters in this window have F diagnoses
            has_mh = False
            for _, encounter in df[(df['pseudo_personid'] == patient_id) & 
                                 (df['days_from_tbi'] > 0) & 
                                 (df['days_from_tbi'] <= window)].iterrows():
                
                for j in range(1, 6):
                    diag = encounter.get(f'diag{j}', '')
                    if pd.notna(diag) and isinstance(diag, str) and diag.startswith('F'):
                        has_mh = True
                        break
                if has_mh:
                    break
            
            outcomes[window].append((str(patient_id), has_mh))
        
        # Add to processed set
        processed_patient_ids.add(str(patient_id))
        print (f"Processing patient {i+1}/{len(patients_to_process)}: {patient_id}")

        
        # Save checkpoint periodically
        if (i+1) % checkpoint_every == 0 or (i+1) == len(patients_to_process):
            # Save progress
            with open(progress_file, 'w') as f:
                json.dump({
                    'processed_patients': list(processed_patient_ids),
                    'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'total_patients': len(all_patient_ids)
                }, f)
            
            # Save patient data
            with open(f"{output_dir}/patient_tokens_partial.json", 'w') as f:
                json.dump(patient_data, f)
            
            # Save outcomes
            for window in [30, 60, 180, 365]:
                outcomes_df = pd.DataFrame(outcomes[window], columns=['patient_id', 'has_mh'])
                outcomes_df.to_csv(f"{output_dir}/outcomes_{window}day_partial.csv", index=False)
            
            print(f"Checkpoint saved: {len(processed_patient_ids)}/{len(all_patient_ids)} patients")
    
    # After completing all patients, save final versions
    with open(f"{output_dir}/patient_tokens.json", 'w') as f:
        json.dump(patient_data, f)
    
    for window in [30, 60, 180, 365]:
        outcomes_df = pd.DataFrame(outcomes[window], columns=['patient_id', 'has_mh'])
        outcomes_df.to_csv(f"{output_dir}/outcomes_{window}day.csv", index=False)
    
    # Print summary statistics
    print("\nProcessing complete.")
    print(f"Processed {len(patient_data)} patients with valid pre-TBI data.")
    
    # Calculate outcome stats
    for window in [30, 60, 180, 365]:
        positive_cases = sum(has_mh for _, has_mh in outcomes[window])
        total_cases = len(outcomes[window])
        print(f"{window}-day window: {positive_cases}/{total_cases} positive cases ({positive_cases/total_cases:.1%})")
    
    return patient_data, outcomes

def analyze_token_vocabulary(patient_data, output_dir="token_data"):
    """
    Analyze the token vocabulary created from patient encounters.
    
    Args:
        patient_data: Dictionary of patient token data
        output_dir: Directory to save analysis results
    """
    # Extract all unique tokens
    all_tokens = set()
    token_freq = {}
    
    for patient in patient_data.values():
        for encounter_tokens in patient['pre_tbi_tokens']:
            tokens = encounter_tokens.split()
            for token in tokens:
                all_tokens.add(token)
                token_freq[token] = token_freq.get(token, 0) + 1
    
    # Analyze token types
    token_types = {}
    for token in all_tokens:
        prefix = token.split('_')[0]
        token_types[prefix] = token_types.get(prefix, 0) + 1
    
    # Print statistics
    print("\nToken Vocabulary Analysis:")
    print(f"Total unique tokens: {len(all_tokens)}")
    print("\nToken types:")
    for prefix, count in sorted(token_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  {prefix}: {count} unique tokens")
    
    # Save token vocabulary
    with open(f"{output_dir}/token_vocabulary.json", 'w') as f:
        json.dump(list(all_tokens), f)
    
    # Save token frequency
    token_freq_python = {k: int(v) for k, v in token_freq.items()}
    token_freq_sorted = sorted(token_freq_python.items(), key=lambda x: x[1], reverse=True)
    with open(f"{output_dir}/token_frequency.json", 'w') as f:
        json.dump(token_freq_sorted, f)
    
    # Create CSV of most common tokens by type
    common_tokens = []
    
    for prefix in token_types.keys():
        prefix_tokens = {t: f for t, f in token_freq.items() if t.startswith(prefix+"_")}
        top_tokens = sorted(prefix_tokens.items(), key=lambda x: x[1], reverse=True)[:10]
        
        for token, freq in top_tokens:
            common_tokens.append({
                'type': prefix,
                'token': token,
                'frequency': freq
            })
    
    common_df = pd.DataFrame(common_tokens)
    common_df.to_csv(f"{output_dir}/common_tokens.csv", index=False)
    
    return all_tokens, token_freq

if __name__ == "__main__":
    # Example usage
    print("This module provides functions to convert encounter data to tokens.")
    print("Import and use in your main script or test scripts.")