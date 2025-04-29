import pandas as pd
import json
from collections import defaultdict

def find_fuzzy_icd10_match(code, icd10_map):
    candidates = [k for k in icd10_map if k.startswith(code)]
    if len(candidates) == 1:
        return candidates[0], icd10_map[candidates[0]], False
    elif len(candidates) > 1:
        # Choose the first candidate as the match but flag the conflict
        chosen = candidates[0]
        return chosen, f"Conflict for {code}: multiple matches {candidates}, chose {chosen}", True
    else:
        return None, "No match found in ICD-10", True



# === FILE PATHS ===
icd10_txt_path = r"C:\Users\elija\Desktop\DoD SAFE-n4zvtrvnkUMaN767\Transformer\IDC_codes\icd10cm-codes-April-2025.txt"
icd9_path = r"C:/Users/elija/Downloads/section111validicd9-jan2025_0.xlsx"
enc_db_path = r"C:/Users/elija/Desktop/DoD SAFE-n4zvtrvnkUMaN767/Caban Model/code/enc_db.csv"

# === LOAD FILES ===
df_icd9 = pd.read_excel(icd9_path)
df_enc = pd.read_csv(enc_db_path)

# === BUILD ICD MAPS ===
icd9_map = dict(zip(df_icd9.iloc[:, 0].astype(str), df_icd9.iloc[:, 1]))
icd10_map = {}

with open(icd10_txt_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)  # split at first group of whitespace
        if len(parts) == 2:
            code, desc = parts
            icd10_map[code.strip()] = desc.strip()

# === GET UNIQUE DIAG CODES ===
diag_cols = ['diag1', 'diag2', 'diag3', 'diag4', 'diag5']
all_diags = pd.unique(df_enc[diag_cols].values.ravel())
all_diags = [d for d in all_diags if isinstance(d, str) and d.strip()]

# === MATCHING FUNCTION ===
final_map = {}
conflicts = []
bypassed_conflicts = []

for code in all_diags:
    if code.startswith("DOD"):
        raw_code = code[3:]
        # Exact match
        if raw_code in icd9_map:
            final_map[code] = icd9_map[raw_code]
        else:
            # Fuzzy match: find codes that start with this
            candidates = [desc for k, desc in icd9_map.items() if k.startswith(raw_code)]
            if len(candidates) == 1:
                final_map[code] = candidates[0]
            elif len(candidates) > 1:
                conflicts.append((code, f"Multiple matches: {[k for k in icd9_map if k.startswith(raw_code)]}"))
            else:
                conflicts.append((code, "No match found in ICD-9"))
    else:
        if code in icd10_map:
            final_map[code] = icd10_map[code]
        else:
            match, result, flagged = find_fuzzy_icd10_match(code, icd10_map)
            if match and not flagged:
                final_map[code] = match
            if match and flagged:
                final_map[code] = match
                bypassed_conflicts.append((code, result))
            else:
                conflicts.append((code, result))

# === SAVE OUTPUTS ===
with open("diagnosis_map.json", "w") as f:
    json.dump(final_map, f, indent=2)

conflict_df = pd.DataFrame(conflicts, columns=["Code", "Issue"])
conflict_df.to_csv("conflicts.csv", index=False)

bypassed_conflict_df = pd.DataFrame(bypassed_conflicts, columns=["Code", "Potential Matches"])
bypassed_conflict_df.to_csv("bypassed_conflicts.csv", index=False)

print(f"✅ Mapping complete. {len(final_map)} codes mapped.")
print(f"⚠️ {len(conflicts)} conflicts saved to 'conflicts.csv'.")
print(f"⚠️ {len(bypassed_conflicts)} potential conflicts saved to 'bypassed_conflicts.csv'.")
