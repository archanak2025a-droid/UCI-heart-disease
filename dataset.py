# ================================================================
# HEART DISEASE EXPLORATORY DATA ANALYSIS
# BAMAT207 - Probability and Statistics Project
# Track 7: Data Visualization & Exploratory Statistical Analysis
# ================================================================

# ================================================================
# 1. IMPORT LIBRARIES
# ================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set plot style for publication quality
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Set figure DPI for high-quality images
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# ================================================================
# 2. LOAD THE DATASET
# ================================================================

from ucimlrepo import fetch_ucirepo

print("=" * 60)
print("LOADING UCI HEART DISEASE DATASET (ID: 45)")
print("=" * 60)

# Fetch dataset
heart_disease = fetch_ucirepo(id=45)

# Extract features and targets
X = heart_disease.data.features
y = heart_disease.data.targets

# Combine into single dataframe
df = pd.concat([X, y], axis=1)

# Check if target is 'num' or 'target'
if 'num' in df.columns:
    df['target'] = (df['num'] > 0).astype(int)
    df = df.drop('num', axis=1)
elif 'target' not in df.columns:
    # If target is the last column
    last_col = df.columns[-1]
    df['target'] = df[last_col]
    df = df.drop(last_col, axis=1)

# ================================================================
# 3. DATA OVERVIEW
# ================================================================

print("\n" + "=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)

print(f"\nDataset shape: {df.shape}")
print(f"Number of samples: {len(df)}")
print(f"Number of features: {len(df.columns) - 1}")

print("\nColumn names and data types:")
print(df.dtypes)

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values per column:")
print(df.isnull().sum())

print("\nSummary statistics:")
print(df.describe())

print("\nTarget distribution:")
print(df['target'].value_counts())
print(f"Percentage with heart disease: {df['target'].mean() * 100:.2f}%")

# ================================================================
# 4. DEFINE VARIABLE GROUPS
# ================================================================

numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

print("\n" + "=" * 60)
print("VARIABLE CLASSIFICATION")
print("=" * 60)
print(f"\nNumerical Random Variables: {numerical_cols}")
print(f"Categorical Random Variables: {categorical_cols}")
print(f"Target Variable: target")

# ================================================================
# 5. DESCRIPTIVE STATISTICS
# ================================================================

print("\n" + "=" * 60)
print("DESCRIPTIVE STATISTICS (Expectation, Variance, Standard Deviation)")
print("=" * 60)

stats_df = pd.DataFrame({
    'Variable': numerical_cols,
    'Mean (Expectation)': [df[col].mean() for col in numerical_cols],
    'Median': [df[col].median() for col in numerical_cols],
    'Variance': [df[col].var() for col in numerical_cols],
    'Std Dev': [df[col].std() for col in numerical_cols],
    'Min': [df[col].min() for col in numerical_cols],
    'Max': [df[col].max() for col in numerical_cols]
})

print("\n", stats_df.round(2))

# ================================================================
# 6. SPLIT BY TARGET FOR GROUP COMPARISON
# ================================================================

df_healthy = df[df['target'] == 0]
df_disease = df[df['target'] == 1]

print("\n" + "=" * 60)
print("STATISTICS BY TARGET GROUP")
print("=" * 60)

for col in numerical_cols:
    print(f"\n{col}:")
    print(f"  Healthy (n={len(df_healthy)}): Mean = {df_healthy[col].mean():.2f}, Std = {df_healthy[col].std():.2f}")
    print(f"  Disease (n={len(df_disease)}): Mean = {df_disease[col].mean():.2f}, Std = {df_disease[col].std():.2f}")

# ================================================================
# 7. COVARIANCE AND CORRELATION
# ================================================================

print("\n" + "=" * 60)
print("COVARIANCE & CORRELATION MATRIX")
print("=" * 60)

# Covariance matrix
cov_matrix = df[numerical_cols + ['target']].cov()
print("\nCovariance Matrix:")
print(cov_matrix.round(4))

# Correlation matrix
corr_matrix = df[numerical_cols + ['target']].corr()
print("\nCorrelation Matrix:")
print(corr_matrix.round(4))

# Correlation with target
print("\nCorrelation with Target (Heart Disease):")
corr_with_target = corr_matrix['target'].drop('target').sort_values(ascending=False)
print(corr_with_target.round(4))

# ================================================================
# 8. HYPOTHESIS TESTING (T-TEST)
# ================================================================

print("\n" + "=" * 60)
print("HYPOTHESIS TESTING (Independent T-Tests)")
print("=" * 60)

for col in numerical_cols:
    healthy_group = df_healthy[col].dropna()
    disease_group = df_disease[col].dropna()
    
    t_stat, p_value = stats.ttest_ind(healthy_group, disease_group)
    
    print(f"\n{col}:")
    print(f"  T-statistic: {t_stat:.4f}")
    print(f"  P-value: {p_value:.4f}")
    
    if p_value < 0.05:
        print(f"  ✓ Statistically significant difference (p < 0.05)")
    else:
        print(f"  ✗ No significant difference (p >= 0.05)")

# ================================================================
# 9. SAVE DATASET FOR REPORT
# ================================================================

df.to_csv('heart_disease_analysis.csv', index=False)
print("\nDataset saved as 'heart_disease_analysis.csv'")

# ================================================================
# 10. VISUALIZATIONS (FOR YOUR REPORT)
# ================================================================

print("\n" + "=" * 60)
print("GENERATING VISUALIZATIONS")
print("=" * 60)

# Create a directory for plots
import os
os.makedirs('plots', exist_ok=True)

# ------- FIGURE 1: Distribution of Patient Age (Histogram) -------
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(df['age'], bins=20, kde=True, ax=ax)
ax.axvline(df['age'].mean(), color='red', linestyle='--', 
           label=f'Mean: {df["age"].mean():.1f}')
ax.axvline(df['age'].median(), color='green', linestyle='--', 
           label=f'Median: {df["age"].median():.1f}')
ax.set_title('Fig. 1: Distribution of Patient Age', fontsize=12)
ax.set_xlabel('Age (years)')
ax.set_ylabel('Number of Patients')
ax.legend()
plt.tight_layout()
plt.savefig('plots/fig1_age_distribution.png', dpi=300, bbox_inches='tight')
plt.show()


# ------- FIGURE 2: Distribution of Sex (Corrected) -------

fig, ax = plt.subplots(figsize=(8, 5))

sex_counts = (
    df['sex']
    .map({0: 'Female', 1: 'Male'})
    .value_counts()
    .reindex(['Female', 'Male'], fill_value=0)
)

sex_counts.plot(kind='bar', ax=ax, color=['#ff7f0e', '#1f77b4'])

ax.set_title('Fig. 2: Distribution of Sex', fontsize=12)
ax.set_xlabel('Sex')
ax.set_ylabel('Count')
ax.set_xticklabels(['Female', 'Male'], rotation=0)

for i, v in enumerate(sex_counts.values):
    ax.text(i, v + 2, str(v), ha='center')

plt.tight_layout()
plt.savefig('plots/fig2_sex_distribution.png',
            dpi=300, bbox_inches='tight')
plt.show()


# ------- FIGURE 3: Cholesterol by Target (Box Plot) -------
fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x='target', y='chol', data=df, ax=ax)
ax.set_title('Fig. 3: Cholesterol Levels by Heart Disease Status', fontsize=12)
ax.set_xlabel('Heart Disease (0 = No, 1 = Yes)')
ax.set_ylabel('Cholesterol (mg/dl)')
plt.tight_layout()
plt.savefig('plots/fig3_cholesterol_by_target.png', dpi=300, bbox_inches='tight')
plt.show()

# ------- FIGURE 4: Max Heart Rate by Target (Violin Plot) -------
fig, ax = plt.subplots(figsize=(10, 6))
sns.violinplot(x='target', y='thalach', data=df, ax=ax)
ax.set_title('Fig. 4: Distribution of Maximum Heart Rate by Heart Disease Status', fontsize=12)
ax.set_xlabel('Heart Disease (0 = No, 1 = Yes)')
ax.set_ylabel('Maximum Heart Rate (bpm)')
plt.tight_layout()
plt.savefig('plots/fig4_thalach_by_target.png', dpi=300, bbox_inches='tight')
plt.show()

# ------- FIGURE 5: Age vs Max Heart Rate (Scatter Plot) -------
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x='age', y='thalach', data=df, hue='target', alpha=0.7, ax=ax)
ax.set_title('Fig. 5: Age vs Maximum Heart Rate by Heart Disease Status', fontsize=12)
ax.set_xlabel('Age (years)')
ax.set_ylabel('Maximum Heart Rate (bpm)')
ax.legend(title='Heart Disease', labels=['No', 'Yes'])
plt.tight_layout()
plt.savefig('plots/fig5_age_vs_thalach.png', dpi=300, bbox_inches='tight')
plt.show()

# ------- FIGURE 6: Correlation Heatmap -------
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', 
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title('Fig. 6: Correlation Matrix of Key Variables', fontsize=12)
plt.tight_layout()
plt.savefig('plots/fig6_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# ------- FIGURE 7: Box Plots for All Numerical Variables -------
fig, axes = plt.subplots(1, 5, figsize=(20, 5))
for i, col in enumerate(numerical_cols):
    sns.boxplot(x='target', y=col, data=df, ax=axes[i])
    axes[i].set_title(f'{col} by Target')
    axes[i].set_xlabel('Heart Disease (0=No, 1=Yes)')
    axes[i].set_ylabel(col)
plt.suptitle('Fig. 7: Distribution of Clinical Variables by Heart Disease Status', fontsize=14)
plt.tight_layout()
plt.savefig('plots/fig7_all_boxplots.png', dpi=300, bbox_inches='tight')
plt.show()

# ------- FIGURE 8: Pairplot (The Grand Finale) -------
sns.pairplot(df[numerical_cols + ['target']], hue='target', 
             diag_kind='kde', corner=True)
plt.suptitle('Fig. 8: Pairwise Relationships of Key Variables', y=1.02, fontsize=14)
plt.savefig('plots/fig8_pairplot.png', dpi=300, bbox_inches='tight')
plt.show()

# ================================================================
# 11. SUMMARY OF KEY FINDINGS
# ================================================================

print("\n" + "=" * 60)
print("SUMMARY OF KEY FINDINGS")
print("=" * 60)

print("""

1. SAMPLE CHARACTERISTICS:
   - Total patients: 303
   - Heart disease prevalence: 45.9% (139 patients)
   - Sex distribution: 68.3% male (207 patients)

2. STRONGEST CORRELATIONS WITH HEART DISEASE:
   - Maximum Heart Rate (thalach): r = -0.42 (negative correlation)
   - ST Depression (oldpeak): r = 0.35 (positive correlation)
   - Age (age): r = 0.23 (positive correlation)

3. STATISTICALLY SIGNIFICANT DIFFERENCES (p < 0.05):
   - Four out of five numerical variables show statistically significant differences (p < 0.05). Cholesterol did not show a significant difference (p = 0.1391).

4. MOST DISCRIMINATING VARIABLES:
   - Maximum Heart Rate: Disease group has lower values
   - ST Depression: Disease group has higher values
   - Age: Disease group tends to be older

""")

print("\n" + "=" * 60)
print("✅ ANALYSIS COMPLETE!")
print("📁 All plots saved in 'plots/' directory")
print("📄 Dataset saved as 'heart_disease_analysis.csv'")
print("=" * 60)
