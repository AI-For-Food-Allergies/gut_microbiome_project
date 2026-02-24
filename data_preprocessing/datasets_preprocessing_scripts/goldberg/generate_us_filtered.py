# generate_us_filtered.py
# Extracted from run_evaluation() in main_mod_us_filtered.py
# Applies outlier filtering + RandomUnderSampler to all_groups.csv
# and saves the resulting sid/label pairs to all_groups_us_filtered.csv

import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────

INPUT_CSV  = Path("./data_preprocessing/datasets/goldberg/all_groups.csv")           
OUTPUT_CSV = Path("./outputs/all_groups_us_filtered.csv")  

OUTLIER_SAMPLES = [
    "ERS4518583",
    "ERS4518584",
    "ERS4518585",
    "ERS4518586",
    "ERS4518587",
    "ERS4518588",
]

# ── 1. Load ────────────────────────────────────────────────────────────────────

dataset_df = pd.read_csv(INPUT_CSV)
print(f"Loaded: {len(dataset_df)} samples")
print(f"Label distribution: {dataset_df['label'].value_counts().to_dict()}")

# ── 2. Filter outliers   ────────────────────

id_col = "sid"
initial_count = len(dataset_df)
dataset_df = dataset_df[~dataset_df[id_col].isin(OUTLIER_SAMPLES)].copy().reset_index(drop=True)
removed = initial_count - len(dataset_df)

if removed > 0:
    print(f"\n🔍 Outlier Filtering:")
    print(f"   - Initial samples  : {initial_count}")
    print(f"   - Removed outliers : {removed}")
    print(f"   - Remaining samples: {len(dataset_df)}")
    print(f"   - Excluded IDs     : {', '.join(OUTLIER_SAMPLES)}\n")

# ── 3. Prepare X / y  (mirrors prepare_data()) ────────────────────────────────

X = np.arange(len(dataset_df)).reshape(-1, 1)
y = dataset_df["label"].values
print(f"Dataset before undersampling: {X.shape[0]} samples")

# ── 4. Undersample  (mirrors the RandomUnderSampler block in run_evaluation) ──

undersampler = RandomUnderSampler(random_state=42)
undersampler.fit_resample(X, y)          
print(f"Dataset after  undersampling: {len(undersampler.sample_indices_)} samples")

# ── 5. Retrieve sid + label for the selected indices ──────────────────────────

undersampled_data = dataset_df.iloc[undersampler.sample_indices_][["sid", "label"]].reset_index(drop=True)
print(f"Label distribution after undersampling: {undersampled_data['label'].value_counts().to_dict()}")

# ── 6. Save ───────────────────────────────────────────────────────────────────

undersampled_data.to_csv(OUTPUT_CSV, index=False)
print(f"\n✓ Saved undersampled data: {OUTPUT_CSV}")
