# ========================
# HEART DISEASE DATASET LOADER
# ========================

from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Fetch dataset
print("Fetching UCI Heart Disease dataset (id=45)...")
heart_disease = fetch_ucirepo(id=45)

# 2. Extract features and targets
X = heart_disease.data.features
y = heart_disease.data.targets

# 3. Combine into dataframe
df = pd.concat([X, y], axis=1)

# 4. Check what we got
print(f"\nDataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# 5. Handle target variable
if 'num' in df.columns:
    df['target'] = (df['num'] > 0).astype(int)
elif 'target' in df.columns:
    df['target'] = df['target']
else:
    # Assume last column is target
    last_col = df.columns[-1]
    if df[last_col].nunique() <= 5:  # Likely a target
        df['target'] = df[last_col]

# 6. Basic info
print("\n=== DATASET OVERVIEW ===")
print(f"Total samples: {len(df)}")
print(f"Features: {len(df.columns)-1}")
print(f"Target distribution:\n{df['target'].value_counts()}")

# 7. Check for missing values
print(f"\nMissing values:\n{df.isnull().sum()}")

# 8. Save for later
df.to_csv('heart_disease_uci.csv', index=False)
print("\n✅ Dataset saved as 'heart_disease_uci.csv'")

# 9. Show first few rows
print("\nFirst 5 rows:")
print(df.head())

# 10. Summary stats
print("\nSummary statistics for numerical columns:")
print(df.describe())

print("\n✅ Ready to analyze!")