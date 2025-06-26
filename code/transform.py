import pandas as pd
import os
import re
from sklearn.preprocessing import MultiLabelBinarizer

# Load raw CSV
input_path = "data/gsearch_jobs.csv"
output_path = "data/processed/cleaned_gsearch_jobs.csv"

print("📂 Loading raw data...")
df = pd.read_csv(input_path)

# ---------------------------
# SCHEDULE TYPE CLEANING
# ---------------------------

# Fill nulls with empty strings
df['schedule_type'] = df['schedule_type'].fillna('').astype(str).str.lower()

# Standardize text
df['schedule_type'] = (
    df['schedule_type']
    .str.replace(' and ', ',', regex=False)
    .str.replace('/', ',', regex=False)
    .str.replace(';', ',', regex=False)
    .str.strip()
)

# Convert string to list of schedule types
df['schedule_type'] = df['schedule_type'].apply(lambda x: [s.strip() for s in x.split(',') if s.strip()])

# Rename variations for consistency
schedule_renames = {
    'full time': 'full-time',
    'fulltime': 'full-time',
    'part time': 'part-time',
    'parttime': 'part-time',
    'contractor': 'contract',
    'contract position': 'contract',
    'temporary': 'temp',
    'seasonal': 'seasonal'
}

df['schedule_type'] = df['schedule_type'].apply(
    lambda lst: [schedule_renames.get(s, s) for s in lst]
)

# One-hot encode using MultiLabelBinarizer
mlb = MultiLabelBinarizer()
schedule_dummies = pd.DataFrame(mlb.fit_transform(df['schedule_type']), 
                                columns=[f'is_{s}' for s in mlb.classes_],
                                index=df.index)

df = pd.concat([df.drop('schedule_type', axis=1), schedule_dummies], axis=1)

# ---------------------------
# SALARY PAY CLEANING
# ---------------------------

def clean_salary(s):
    if pd.isnull(s):
        return None

    s = str(s).lower()
    s = s.replace('$', '').replace(',', '').replace('usd', '').strip()

    # Convert K to thousand
    s = re.sub(r'(\d+)k', lambda m: str(int(m.group(1)) * 1000), s)

    # If it's a range like "50000-70000", take average
    if '-' in s:
        parts = s.split('-')
        try:
            low = float(parts[0])
            high = float(parts[1])
            return (low + high) / 2
        except:
            return None
    else:
        try:
            return float(s)
        except:
            return None

# Clean the salary column
df['salary_cleaned'] = df['salary'].apply(clean_salary)

# Fill missing values with the mean salary
mean_salary = df['salary_cleaned'].mean()
df['salary_cleaned'] = df['salary_cleaned'].fillna(mean_salary)

# Save cleaned file


df.to_csv(output_path, index=False)
print(f" Cleaned data saved to: {output_path}")
