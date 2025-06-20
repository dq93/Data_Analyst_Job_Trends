import pandas as pd
import numpy as np
import os


# Define file paths
raw_data_path = "data/raw/gsearch_jobs.csv"
cleaned_data_path = "data/cleaned/cleaned_jobs.csv"

def feature_engineer_schedule_type(df):
    all_types = ['Full-time', 'Part-time', 'Contractor', 'Internship', 'Temp work', 'Per diem', 'Volunteer']
    df = df.copy()
    df['schedule_type'] = df['schedule_type'].fillna('')
    for t in all_types:
        df[t] = df['schedule_type'].apply(lambda x: int(t in x))
    return df

# Load raw data
df = pd.read_csv(raw_data_path)

# - - - Transform: Data cleaning and feature engineering - - -

# 1. Check missing data 
missing_data = df.isnull().sum().sort_values(ascending=False)
missing_values_percentage = (df.isnull().sum() / len(df)) * 100
missing_values_percentage = missing_values_percentage[missing_values_percentage > 0]
# print(missing_data)
# print(missing_values_percentage.sort_values(ascending=False))

# 2. Convert salary_avg to numeric for entire DF
df['salary_avg'] = pd.to_numeric(df['salary_avg'], errors='coerce')

# 3. Separate yearly vs hourly salaries for analysis
yearly_salaries = df[df['salary_avg'] >= 10000]['salary_avg'].dropna()
hourly_salaries = df[df['salary_avg'] < 200]['salary_avg'].dropna()

#print(yearly_salaries.describe())
#print(hourly_salaries.describe())

# 4. Prepare filtered dataset for commute and salary analysis
filtered = df[['salary_avg', 'commute_time']].copy()
filtered['salary_avg'] = pd.to_numeric(filtered['salary_avg'], errors='coerce')
filtered['commute_time'] = pd.to_numeric(filtered['commute_time'], errors='coerce')
filtered = filtered.dropna()

# 5. Clean 'work_from_home' column to lowercase, map to bool, include NaNs
df['work_from_home'] = (
    df['work_from_home']
    .astype(str)
    .str.lower()
    .map({'true': True, 'false': False})
    .fillna(pd.NA)
)

# 6. Create 'has_pay_range' by checking for salary in text columns
text_cols = [col for col in ['description', 'extensions'] if col in df.columns]
df['has_pay_range'] = df[text_cols].apply(
    lambda row: row.astype(str).str.contains(r'\$\d+', case=False, na=False).any(),
    axis=1
)

# 7. Extract state from 'location' and handle exceptions
df['state'] = df['location'].str.extract(r',\s*([A-Z]{2})')
df['state_clean'] = np.where(
    df['location'].isin(['United States', 'Anywhere']),
    df['location'],
    df['state']
)

# 8. Calculate job description word count
df['desc_wordcount'] = df['description'].astype(str).apply(lambda x: len(x.split()))

# 9. Feature engineer schedule_type into binary flags
df = feature_engineer_schedule_type(df)

# - - - Save cleaned data - - - 
os.makedirs(os.path.dirname(cleaned_data_path), exist_ok=True)
df.to_csv(cleaned_data_path, index=False)
