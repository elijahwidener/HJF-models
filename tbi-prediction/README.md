# TBI Mental Health Prediction Models

This folder contains machine learning models and analysis scripts for predicting mental health outcomes following Traumatic Brain Injury (TBI) using traditional machine learning approaches.

## Project Structure

```
TBI-Predictions/
├── Code/                         # Main code for machine learning models
│   ├── feature_engineering.py    # Feature extraction and engineering
│   ├── model_training.py         # Model training scripts
│   ├── evaluation.py             # Model evaluation utilities
│   ├── visualization.py          # Visualization functions
│   ├── data_preprocessing.py     # Data cleaning and preprocessing
│   └── requirements.txt          # Required Python packages
├── Analysis/                     # Analysis scripts and notebooks
│   ├── feature_importance.ipynb  # Analysis of feature importance
│   ├── model_comparison.ipynb    # Comparison of different ML models
│   └── cohort_analysis.ipynb     # Analysis of the TBI patient cohort
└── Models/                       # Saved model files
    ├── rf_model.pkl              # Random Forest model
    ├── gb_model.pkl              # Gradient Boosting model
    └── lr_model.pkl              # Logistic Regression model
```

## Setup and Installation

1. **Install required packages**
   ```bash
   pip install -r Code/requirements.txt
   ```

2. **Configure data paths**
   - The code expects data to be available at specific locations
   - Update path variables in the code files to match your data locations

## Data Processing Pipeline

### 1. Data Preprocessing

The `data_preprocessing.py` script handles:
- Cleaning of encounter data
- Handling of missing values
- Date normalization
- Patient cohort selection based on inclusion/exclusion criteria

```bash
python Code/data_preprocessing.py --input /path/to/raw_data.csv --output /path/to/processed_data.csv
```

### 2. Feature Engineering

The `feature_engineering.py` script extracts clinically relevant features:
- Demographics (age, gender, etc.)
- Encounter frequency
- Diagnosis history
- Medication usage
- Provider specializations
- Facility types

```bash
python Code/feature_engineering.py --input /path/to/processed_data.csv --output /path/to/features.csv
```

### 3. Model Training

The `model_training.py` script trains multiple models:
- Random Forest
- Gradient Boosting
- Logistic Regression
- Support Vector Machines
- Neural Networks

```bash
python Code/model_training.py --features /path/to/features.csv --output /path/to/Models/
```

### 4. Model Evaluation

The `evaluation.py` script provides:
- Cross-validation
- Performance metrics (AUC, precision, recall, F1)
- Confusion matrices
- ROC curves

```bash
python Code/evaluation.py --model /path/to/Models/gb_model.pkl --test /path/to/test_data.csv
```

## Analysis Notebooks

The `Analysis/` folder contains Jupyter notebooks for in-depth exploration:

1. **feature_importance.ipynb**
   - Identifies the most predictive features
   - Visualizes feature importance by model type
   - Analyzes correlations between features

2. **model_comparison.ipynb**
   - Compares performance of different model architectures
   - Examines trade-offs between different metrics
   - Parameter sensitivity analysis

3. **cohort_analysis.ipynb**
   - Demographics of the TBI patient cohort
   - Temporal patterns in encounters
   - Mental health diagnosis distribution

## Usage Notes

- The models were developed for research purposes and should be validated in a clinical setting before deployment
- Performance varies by prediction window, with better results for longer windows due to more complete data
- Feature importance analysis can provide clinical insights into risk factors