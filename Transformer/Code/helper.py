import pandas as pd

# Path to your file
file_path = r"C:\Users\elija\Desktop\DoD SAFE-n4zvtrvnkUMaN767\Caban Model\code\enc_db.csv"

# Read the CSV
df = pd.read_csv(file_path)

# Print unique values in the 'sponrankgrp' column
unique_values = df['fmp'].dropna().unique()
print("Unique appttype values:")
for val in unique_values:
    print(val)
