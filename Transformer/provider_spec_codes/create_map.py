import pandas as pd
import json

# Path to the Excel file (use raw string literal or double backslashes)
file_path = r"C:\Users\elija\Desktop\DoD SAFE-n4zvtrvnkUMaN767\Transformer\provider_spec_codes\codes.xlsx"

# Read the Excel file
df = pd.read_excel(file_path)

# Create a dictionary map from the data (code as key and description as value)
map_dict = {}

# Assuming the first column is the code and the second is the description
for index, row in df.iterrows():
    code = str(row[0])  # Ensure code is treated as a string
    description = row[1]  # Description is in the second column
    map_dict[code] = description

# Optionally, save the map to a JSON file
with open("map_output.json", "w") as json_file:
    json.dump(map_dict, json_file, indent=4)
