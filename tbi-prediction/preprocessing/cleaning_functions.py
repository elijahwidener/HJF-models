import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder

def clean_id_fields(df):
    """
    Standardize ID column names and remove unnecessary ID columns
    """
    df = df.rename(columns={
        'pseudo_personid': 'patient_id',
        'recordid': 'record_id',
        'tmt_dmisid': 'treatment_id' 
    })
    return df

def clean_severity(df):
    """
    Convert severity to numeric
    """
    severity_encoder = LabelEncoder()
    df['severity'] = severity_encoder.fit_transform(df['severity'])
    return df, severity_encoder

def clean_gender(df):
    """
    Convert gender to binary
    """
    df['gender'] = df['gender'].map({'M': 1, 'F': 0})
    return df

def clean_meprs(df):
    """
    Convert MEPRS4 codes to numeric using first three characters
    """
    encoder = LabelEncoder()
    # Extract first three characters
    df['meprs_code'] = df['meprs4'].str[:3]
    df['meprs_code'] = encoder.fit_transform(df['meprs_code'])
    df = df.drop('meprs4', axis=1)
    return df, encoder

def clean_diagnoses(df):
    """
    Convert diagnosis codes to numeric and identify mental health diagnoses
    """
    diag_cols = ['diag1', 'diag2', 'diag3', 'diag4', 'diag5']
    encoder = LabelEncoder()
    
    # Fill missing values first
    for col in diag_cols:
        df[col] = df[col].fillna('MISSING')
    
    # Combine all diagnoses for encoder fitting (including 'MISSING')
    all_diagnoses = pd.concat([df[col] for col in diag_cols])
    encoder.fit(all_diagnoses.unique())
    
    # Transform each column
    for col in diag_cols:
        df[f'{col}_encoded'] = encoder.transform(df[col])
        df[f'{col}_mh'] = df[col].str.startswith('F', na=False)
    
    # Drop original string columns
    df = df.drop(diag_cols, axis=1)
    
    # Create overall mental health flag
    df['has_mh_diagnosis'] = df[[f'{col}_mh' for col in diag_cols]].any(axis=1)
    
    return df, encoder

def clean_procedure_codes(df):
    """
    Clean and standardize procedure codes
    """
    proc_cols = ['em1', 'proc1', 'proc2', 'proc3', 'proc4', 'proc5']
    
    df[proc_cols] = df[proc_cols].fillna('0')
    df[proc_cols] = df[proc_cols].replace('', '0')
    
    def clean_code(code):
        if code == '0' or code == 'XXXXX':
            return 0
        elif code.isdigit():
            return int(code)
        elif code.startswith('G'):
            return 100000
        elif code.startswith('S'):
            return 100001
        elif code.startswith('T'):
            return 100002
        elif code.startswith('L'):
            return 100003
        else:
            return 0
    
    for col in proc_cols:
        df[col] = df[col].apply(clean_code)
    
    return df

def clean_appointment_info(df):
    """
    Clean appointment-related columns
    """
    # Clean appointment type
    appt_encoder = LabelEncoder()
    df['appttype'] = appt_encoder.fit_transform(df['appttype'])
    
    # Clean provider specialty
    spec_encoder = LabelEncoder()
    df['apptprovspec'] = df['apptprovspec'].fillna(-1)
    df['apptprovspec'] = spec_encoder.fit_transform(df['apptprovspec'].astype(str))
    
    # Clean HIPAA specialty
    hipaa_encoder = LabelEncoder()
    df['apptprovspechipaa'] = hipaa_encoder.fit_transform(df['apptprovspechipaa'])
    
    # Clean appointment status
    df['apptstatcode'] = df['apptstatcode'].fillna(-1)
    
    encoders = {
        'appttype': appt_encoder,
        'provspec': spec_encoder,
        'hipaa': hipaa_encoder
    }
    
    return df, encoders

def clean_costs(df):
    """
    Normalize cost columns
    """
    cost_columns = {
        'fullcost': 'total_cost',
        'fullcostclinsal': 'clinical_salary_cost',
        'fullcostlab': 'laboratory_cost',
        'fullcostother': 'other_cost',
        'fullcostotheranc': 'ancillary_cost',
        'fullcostpharm': 'pharmacy_cost',
        'fullcostprofsal': 'professional_salary_cost',
        'fullcostrad': 'radiology_cost',
        'fullcostsupport': 'support_cost'
    }
    
    df = df.rename(columns=cost_columns)
    new_cols = list(cost_columns.values())
    df[new_cols] = df[new_cols].fillna(0)
    
    scaler = StandardScaler()
    df[new_cols] = scaler.fit_transform(df[new_cols])
    
    return df, scaler

def clean_study_periods(df):
    """
    Convert study period indicators to binary
    """
    period_cols = ['study90day', 'study180day', 'study270day', 'study365day',
                  'study90day_encounter', 'study180day_encounter', 
                  'study270day_encounter', 'study365day_encounter']
    
    for col in period_cols:
        df[col] = df[col].astype(int)
    
    return df

def clean_categorical_columns(df):
    """
    Encode remaining categorical columns
    """
    categorical_cols = ['prodline', 'sponservice', 'sponrankgrp', 'race']
    encoders = {}
    
    for col in categorical_cols:
        # Fill missing values first
        df[col] = df[col].fillna('MISSING')
        
        # Then fit and transform
        encoder = LabelEncoder()
        encoder.fit(df[col])
        df[col] = encoder.transform(df[col])
        encoders[col] = encoder
    
    return df, encoders

def add_temporal_features(df):
    """
    Add temporal features relative to TBI date
    """
    # Convert dates to datetime
    df['tbi_index_date'] = pd.to_datetime(df['tbi_index_date'])
    df['servicedate'] = pd.to_datetime(df['servicedate'])
    
    # Calculate days from TBI
    df['days_from_tbi'] = (df['servicedate'] - df['tbi_index_date']).dt.total_seconds() / (24 * 60 * 60)
    
    # Create exponential time weights
    lambda_param = 0.01  # Can be tuned
    df['time_weight'] = np.exp(-lambda_param * np.abs(df['days_from_tbi']))
    
    # Drop original date columns
    df = df.drop(['tbi_index_date', 'servicedate'], axis=1)
    
    return df

def clean_provider_skill(df):
    """
    Clean provider skill type
    """
    df['provskilltype_appt'] = pd.to_numeric(df['provskilltype_appt'], errors='coerce')
    df['provskilltype_appt'] = df['provskilltype_appt'].fillna(-1)
    return df

def clean_patient_category(df):
    """
    Convert patient category codes from letter+number format to pure numbers.
    First digit comes from letter (A=1, B=2, etc.)
    Last two digits are kept as is.
    
    Examples:
    A11 -> 111  (1 for A, keeps 11)
    B11 -> 211  (2 for B, keeps 11)
    F13 -> 613  (6 for F, keeps 13)
    Z99 -> 2699 (26 for Z, keeps 99)
    """
    def convert_category(code):
        if pd.isna(code) or code == '':
            return 0
        
        # Get number value of letter (A=1, B=2, etc)
        letter_val = ord(code[0].upper()) - ord('A') + 1
        
        # Get the numbers from the original code
        number_part = code[1:]
        
        # Combine them (letter value * 100 + original numbers)
        return letter_val * 100 + int(number_part)
    
    df['patientcat'] = df['patientcat'].apply(convert_category)
    return df