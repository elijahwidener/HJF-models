# TBI Mental Health Prediction Project

This repository contains research and code for predicting mental health outcomes following Traumatic Brain Injury (TBI) using various machine learning and deep learning approaches.

## Project Overview

Traumatic Brain Injury (TBI) patients are at elevated risk for developing mental health conditions following their injury. This project focuses on developing predictive models that can identify patients at high risk for mental health conditions based on their pre-TBI medical records. The models aim to enable earlier intervention and potentially improve patient outcomes.

## Key Features

- **Multiple Modeling Approaches**: Traditional machine learning and transformer-based deep learning
- **Temporal Analysis**: Consideration of the temporal nature of medical encounters
- **Multi-Window Prediction**: Risk predictions at 30, 60, 180, and 365 days post-TBI
- **Clinical Interpretability**: Feature importance analysis to identify key risk factors
- **Risk Stratification**: Classification of patients into risk tiers for clinical decision support

## Repository Structure

```
TBI-Project/
├── Transformer/                    # Transformer-based deep learning models
│   ├── Code/                       # Implementation code for transformer models
│   ├── provider_spec_codes/        # Provider specialization code mappings
│   └── README.md                   # Transformer-specific documentation
├── TBI-Predictions/                # Traditional machine learning models
│   ├── Code/                       # Implementation code for ML models
│   ├── Analysis/                   # Analysis notebooks
│   ├── Models/                     # Trained model files
│   └── README.md                   # ML-specific documentation
├── Documentation/                  # Project documentation
│   ├── Whitepaper.pdf              # Technical whitepaper describing the approach
│   ├── Model_Card.md               # Model card with performance characteristics
│   └── Data_Dictionary.xlsx        # Dictionary of data fields and descriptions
└── README.md                       # This file
```

## Important Documents

### Whitepaper

The technical whitepaper (`Documentation/Whitepaper.pdf`) provides a comprehensive description of:
- The clinical problem and motivation
- Literature review on TBI and mental health
- Data collection and cohort definition
- Feature engineering approach
- Modeling methodologies
- Experimental results
- Clinical implications
- Limitations and future work

### Model Cards

Model cards (`Documentation/Model_Card.md`) document the performance characteristics, intended use cases, and limitations of the deployed models.

## Implementation Details

### Transformer-Based Approach

The `Transformer/` directory contains deep learning models based on Bio_ClinicalBERT that incorporate temporal attention mechanisms to process sequential medical encounters. These models are implemented in PyTorch and can be run in Google Colab for GPU acceleration.

See the [Transformer README](Transformer/README.md) for detailed setup and usage instructions.

### Traditional Machine Learning Approach

The `TBI-Predictions/` directory contains traditional machine learning models (Random Forest, Gradient Boosting, etc.) with extensive feature engineering. These models serve as both baselines and complementary approaches.

See the [TBI-Predictions README](TBI-Predictions/README.md) for detailed setup and usage instructions.

## Data Requirements

This repository contains only code; data is not included due to privacy concerns. The models expect the following data:

1. **Encounter data**: Medical encounters with diagnosis codes, procedures, provider information, etc.
2. **Patient demographics**: Age, gender, military rank, service branch, etc.
3. **TBI index dates**: Dates of TBI diagnosis for cohort identification
4. **Mental health outcomes**: Mental health diagnoses post-TBI for training labels

Data should be preprocessed into the format expected by the respective models. See the README files in each directory for specifics.


## Usage Notes

- These models were developed for research purposes and should be validated in a clinical setting before deployment
- The repository provides code for both model development and inference
- For deployment in a clinical setting, additional work would be needed to integrate with existing EHR systems
- The transformer-based approach requires GPU resources for efficient training

