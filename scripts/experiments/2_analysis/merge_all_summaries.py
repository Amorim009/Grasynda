import pandas as pd
import os

OUT_DIR = r"c:\Users\lhenr\Desktop\Grasynda\assets\results\grasynda_variants_audit"
DETAILS_PATH = os.path.join(OUT_DIR, "details.csv")
SUMMARY_PATH = os.path.join(OUT_DIR, "summary.csv")

# 1. Load details (contains the latest run: Q12, Q8, Visibility)
df_details = pd.read_csv(DETAILS_PATH)
summary_latest = df_details.groupby('Method').mean(numeric_only=True).round(3).reset_index()

# 2. Load the current summary (which the user manually updated/maintained)
df_summary = pd.read_csv(SUMMARY_PATH)

# Clean up any whitespace/indentation issues from manual edits
df_summary.columns = [c.strip() for c in df_summary.columns]
for col in df_summary.select_dtypes(include=['object']).columns:
    df_summary[col] = df_summary[col].str.strip()

# 3. Merge them
combined = pd.concat([df_summary, summary_latest], ignore_index=True).drop_duplicates(subset=['Method'], keep='last')

# 4. Sort by Authenticity and save
combined = combined.sort_values(by='Authenticity', ascending=False)
combined.to_csv(SUMMARY_PATH, index=False)

print("Leaderboard updated with all variants.")
print(combined[['Method', 'Precision', 'Authenticity', 'SpectralCoh']])
