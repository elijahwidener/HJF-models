# Configuration parameters
import torch

# Model settings
PRETRAINED_MODEL = "emilyalsentzer/Bio_ClinicalBERT"
MAX_SEQ_LENGTH = 512
MAX_ENCOUNTERS = 30

# Training settings
BATCH_SIZE = 8
EVAL_BATCH_SIZE = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 5
PATIENCE = 2

# Data settings
WINDOWS = [30, 60, 180, 365]
RANDOM_SEED = 42

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Risk thresholds
RISK_THRESHOLDS = {
    'low': 0.3,
    'high': 0.7
}