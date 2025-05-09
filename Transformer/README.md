# TBI Mental Health Prediction using Transformer Models

This repository contains code for predicting mental health outcomes after Traumatic Brain Injury (TBI) using transformer-based models. The implementation uses Bio_ClinicalBERT as a base model, with temporal attention mechanisms to capture the progression of patient encounters over time.

## Project Structure

```
Transformer/
├── Code/                           # Main code for the transformer model
│   ├── Bio_ClinicalBERT.py         # Implementation of Bio_ClinicalBERT for TBI prediction
│   ├── config.py                   # Configuration parameters
│   ├── data_utils.py               # Utilities for data loading and preparation
│   ├── data_validation.py          # Functions for validating and analyzing data
│   ├── dataset.py                  # PyTorch dataset implementation
│   ├── encounter_to_tokens.py      # Functions to convert medical encounters to tokens
│   ├── main.py                     # Main script to train and evaluate the model
│   ├── models.py                   # Model architecture definitions
│   ├── predict.py                  # Functions for making predictions and risk stratification
│   ├── requirements.txt            # Required Python packages
│   ├── test_encounter_to_tokens.py # Test script for token conversion
│   ├── test_text_tokens.py         # Test script for analyzing token distribution
│   ├── train_utils.py              # Training and evaluation utilities
│   └── transformer_tokenizer.py    # Custom tokenizer for clinical encounters
├── provider_spec_codes/            # Maps for provider specialization codes
│   ├── create_map.py               # Script to create the provider specialization map
│   ├── provider_spec_map.json      # Provider specialization code mapping
└── README.md                       # This file
```

## Setup in Google Colab

Follow these steps to set up and run the model in Google Colab:

1. **Create a new Colab notebook**
   - Go to [Google Colab](https://colab.research.google.com/)
   - Create a new notebook

2. **Mount Google Drive** (to store model artifacts and data)
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

3. **Clone or upload the code**
   - Option 1: Clone from GitHub (if the repository is public)
     ```python
     !git clone [REPOSITORY_URL] /content/tbi-transformer
     ```
   - Option 2: Upload the code manually
     - Upload code to your Google Drive
     - Copy to Colab:
       ```python
       !cp -r /content/drive/MyDrive/path/to/Transformer /content/tbi-transformer
       ```

4. **Install required packages**
   ```python
   !pip install -r /content/tbi-transformer/Code/requirements.txt
   ```

5. **Configure data paths**
   - You need to adjust paths in the code to point to your data locations
   - Update paths in:
     - `encounter_to_tokens.py` for diagnosis map, category map, provider map, and DMIS data
     - `test_encounter_to_tokens.py` and `test_text_tokens.py` for input data paths

## Running the Model

### Data Preprocessing

1. **Convert encounters to tokens** (if not already done)
   ```python
   import sys
   sys.path.append('/content/tbi-transformer/Code')
   
   from encounter_to_tokens import process_patient_cohort_incremental
   import pandas as pd
   
   # Load your encounter data
   df = pd.read_csv('/path/to/encounter_data.csv')
   
   # Process the patient cohort
   patient_data, outcomes = process_patient_cohort_incremental(
       df, 
       output_dir="/content/drive/MyDrive/tbi_model/token_data"
   )
   ```

2. **Analyze token vocabulary** (optional)
   ```python
   from encounter_to_tokens import analyze_token_vocabulary
   
   all_tokens, token_freq = analyze_token_vocabulary(
       patient_data, 
       output_dir="/content/drive/MyDrive/tbi_model/token_data"
   )
   ```

### Training and Evaluation

1. **Prepare data and train the model**
   ```python
   import os
   import sys
   sys.path.append('/content/tbi-transformer/Code')
   
   # Set output directory
   os.environ['OUTPUT_DIR'] = '/content/drive/MyDrive/tbi_model/output'
   
   # Run the main script
   !python /content/tbi-transformer/Code/main.py
   ```

2. **Monitor training in Colab**
   - You can visualize training progress by adding TensorBoard integration:
   ```python
   %load_ext tensorboard
   %tensorboard --logdir /content/drive/MyDrive/tbi_model/output/logs
   ```

### Making Predictions

```python
import sys
sys.path.append('/content/tbi-transformer/Code')
import torch
from transformers import AutoTokenizer
from models import MultiTaskTBIPredictor
from predict import predict_for_new_patient

# Load the trained model
model_path = "/content/drive/MyDrive/tbi_model/output/tbi_bert_model/model.pt"
tokenizer = AutoTokenizer.from_pretrained("/content/drive/MyDrive/tbi_model/output/tbi_bert_model")

model = MultiTaskTBIPredictor("emilyalsentzer/Bio_ClinicalBERT")
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Example patient encounters (tokenized)
patient_encounters = [
    "AGE_45 GENDER_Male FACILITY_NAME_WALTER_REED DIAG1_CAT_Mental_Disorders DIAG1_DESC_Post-traumatic stress disorder",
    "AGE_45 GENDER_Male FACILITY_NAME_WALTER_REED DIAG1_CAT_Injury_and_Poisoning DIAG1_DESC_Traumatic brain injury"
]

# Make predictions
risk_scores = predict_for_new_patient(model, tokenizer, patient_encounters, torch.device('cpu'))
print("Risk scores by window:")
for window, score in risk_scores.items():
    print(f"{window}-day window: {score:.4f}")
```

## Model Details

This implementation uses a transformer-based architecture with the following components:

1. **Base Model**: Bio_ClinicalBERT, a BERT model fine-tuned on clinical text
2. **Temporal Attention**: A custom attention mechanism that considers the temporal progression of patient encounters
3. **Multi-Task Learning**: Predicts mental health outcomes at different time windows (30, 60, 180, and 365 days post-TBI)
4. **Risk Stratification**: Categorizes patients into low, medium, and high risk groups

## Data Requirements

The model requires patient encounter data with the following information:
- Patient demographics (age, gender, etc.)
- Encounter details (facility, date, etc.)
- Diagnosis codes (preferably ICD-10)
- Provider specialization
- TBI index date (date of TBI diagnosis)

The data should be preprocessed into a format compatible with the `encounter_to_tokens.py` script, which converts each encounter into a standardized token representation.

## Notes

- The code assumes the availability of mapping files for diagnosis codes, provider specialization codes, and DMIS facility codes.
- GPU acceleration is recommended for training the model. Google Colab provides free GPU access, which can be enabled via Runtime > Change runtime type > Hardware accelerator > GPU.
- For large datasets, consider using Colab Pro or other cloud computing resources.