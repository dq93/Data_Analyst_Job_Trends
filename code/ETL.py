import os
import zipfile
import pandas as pd
import re
from collections import Counter
import numpy as np

"""
Download a Kaggle dataset of data analyst job postings, extracts the ZIP file, 
and load the job data into a pandas DataFrame.
"""

os.system("kaggle datasets download -d lukebarousse/data-analyst-job-postings-google-search")

zip_path = "data-analyst-job-postings-google-search.zip"
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall("data/raw")
os.remove(zip_path)    

df = pd.read_csv("data/raw/gsearch_jobs.csv")

# ===Data transformation===

df[df.duplicated("job_id")]

# Remove duplicates from "Job_id" and keep those that were posted first
df = df.drop_duplicates(subset="job_id", keep="first")

# Dropping columns
columns_to_drop = [
    'Unnamed: 0', 'index', 'search_term', 'search_location', 'commute_time',
    'thumbnail', 'salary', 'salary_pay', 'salary_yearly', 'salary_hourly', 'salary_avg'
]
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

# Remove any strange charater from "title"
df["title"] = df["title"].str.replace(r"[^a-zA-Z0-9\s,./\-&()]", "", regex=True)

# Capitalize the first letter of each word in the job titles
df["title"]= df["title"].str.title()

# Removes the prefix via from the via column
df['via'] = df['via'].str.replace(r'^via\s+', '', regex=True).str.strip()

# Removes whitespace around locations in the location column
df['location'] = df['location'].str.strip()

df["Date"] = pd.to_datetime(df["date_time"]).dt.date

# Function to standardize schedule_type into binary Columns
def feature_engineer_schedule_type(df):
    all_types = ['Full-time', 'Part-time', 'Contractor', 'Internship', 'Temp work', 'Per diem', 'Volunteer']
    df = df.copy()

    for t in all_types:
        df[t] = df['schedule_type'].apply(lambda x: int(t in x))
    return df

# Find Min and Max experience required for role
exp_pattern = r"(?:(?:at least|min(?:imum)? of)\s*\d+\s*years?)|(?:\d+\+?\s*[-–]?\s*\d*\s*years?)"

# Find Min and Max Degree required for role
degree_pattern = r"(?:Bachelor(?:'s)?|BA|BS|BSc|Master(?:'s)?|MS|MSc|MBA|PhD|Doctorate|degree in [A-Za-z ]+)"

# Function for standardizing job titles
def normalize_title(title):
    title = title.lower() # makes lowercase
    title = re.sub(r'(sr\.?|senior)', 'senior', title) # Standardize senior
    title = re.sub(r'(jr\.?|junior)', 'junior', title) # Standardize junior
    title = re.sub(r'\s*-\s*.*$', '', title) # Removes suffices like '-contract to hire'
    title = re.sub(r'[^\w\s]', '', title) # Removes punctiation
    title = title.strip()
    return title

df['title'] = df['title'].apply(normalize_title)

# Create Boolean columns to indicate whether a job description mentions
# Experience or degree requirements based on regex pattern matching
df["Has_experience_requirement"] = df["description"].str.contains(exp_pattern, flags=re.IGNORECASE,regex=True).astype(int)
df["Has_degree_requirement"] = df["description"].str.contains(degree_pattern, flags=re.IGNORECASE, regex=True).astype(int)

# List of desired technical skills
skills = [
    "Python", "R", "SQL", "Java", "Scala", "Excel", "Microsoft Excel", "Tableau", "Power BI", 
    "Looker", "Google Sheets", "Matplotlib", "Seaborn", "Apache Airflow", "dbt", "Apache NiFi", 
    "SSIS", "Informatica", "Talend", "MySQL", "PostgreSQL", "Oracle", "Redshift", "Snowflake", 
    "BigQuery", "MongoDB", "AWS", "Azure", "GCP", "Google Cloud Platform", "Apache Spark", 
    "Hadoop", "Kafka", "Hive", "Presto", "Docker", "Kubernetes", "Terraform", "Git", "GitHub", 
    "Scikit-learn", "TensorFlow", "Keras", "XGBoost", "Pandas", "NumPy"
]
# Make sure description has no missing values
df["description"] = df["description"].fillna("")

print('Creating features, this should take a few minutes')

# Create one column for each skill (True if the skill is mentioned)
for skill in skills:
    df[skill] = df["description"].str.contains(rf"\b{re.escape(skill)}\b", case=False, regex=True).astype(int)

# Create a list of skills found in each row
def find_skills(text):
    return [skill for skill in skills if re.search(rf"\b{re.escape(skill)}\b", text, re.IGNORECASE)]

df["skills_found"] = df["description"].apply(find_skills)

# Count how often each skill appears
all_skills = [skill for row in df["skills_found"] for skill in row]
skill_counts = Counter(all_skills)

# Put into a DataFrame
skill_df = pd.DataFrame(skill_counts.most_common(20), columns=["Skill", "Frequency"])

text_cols = [col for col in ['description', 'extensions'] if col in df.columns]
df['has_pay_range'] = df[text_cols].apply(
    lambda row: row.astype(str).str.contains(r'\$\d+', case=False, na=False).any(),
    axis=1
)

# List of visa keywords to check for
visa_keywords = ['h-1b', 'h1b', 'h-2b', 'l-1a', 'l-1b', 'o-1', 'eb-2', 'eb2', 'eb-3', 'eb3', 'visa sponsorship']

# Function to return 1 if any visa keyword is found in the description, else 0
def binary_visa_flag(description):
    desc = str(description).lower()  # ensure description is a string
    return int(any(kw in desc for kw in visa_keywords))

# Add the binary column to your existing DataFrame
df["visa_sponsorship_flag"] = df["description"].apply(binary_visa_flag)

# Using regions to find and filter states
df['description'] = df['description'].fillna('').astype(str)
def city_state(text):
    matches = re.findall(r'\b([A-Za-z\s]+),\s*([A-Z]{2})\b', text)
    return matches
df['location_df'] = df['description'].apply(city_state)
df['location_city'] = df['location_df'].apply(lambda x: x[0][0] if x else None)
df['location_state'] = df['location_df'].apply(lambda x: x[0][1] if x else None)
location_keywords = {
    "bay area": "CA",
    "silicon valley": "CA",
    "new york city": "NY",
    "nyc": "NY",
    "tri-state": "NY",
    "los angeles": "CA",
    "seattle area": "WA",
    "greater seattle": "WA",
    "dfw": "TX",
    "chicago area": "IL",
    "atlanta metro": "GA",
    "boston area": "MA",
    "san francisco": "CA",
    "washington dc": "DC"
}
def find_region_keywords(text):
    text = text.lower()
    for keyword, state in location_keywords.items():
        if keyword in text:
            return keyword, state
        return None, None

df[['region_keyword', 'region_state']] = df['description'].apply(lambda x: pd.Series(find_region_keywords(x)))
def choose_state(row):
    return row['location_state'] if pd.notnull(row['location_state']) else row['region_state']

df['state_result'] = df.apply(choose_state, axis=1)

df['location_found'] = df['state_result'].apply(lambda x: 'Found' if pd.notnull(x) else 'Unknown')

us_states = {
    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
    'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
    'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
    'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
    'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY',
    'DC'
}
valid_states = df[df['state_result'].isin(us_states)]

valid_states['location_state']

df['State'] = valid_states['location_state']

# Droppping columns and features no longer needed
df = df.drop(columns=['location_df', 'location_city', 'state_result', 'region_keyword', 'region_state', 'location_state'])

# Adding work type feature
def classify_work_type(text):
    text_lower = text.lower() if isinstance(text, str) else ''
    remote_keywords = ["remote", "work from home", "telecommute", "virtual", "distributed team"]
    hybrid_keywords = ["hybrid", "partial remote", "partially remote", "flexible location", "mix of remote and office"]
    in_office_keywords = ["on-site", "onsite", "in-office", "office-based", "must be onsite", "must be on site"]
    # Check remote first
    if any(kw in text_lower for kw in remote_keywords):
        if any(kw in text_lower for kw in hybrid_keywords):
            return "Hybrid"
        else:
            return "Remote"
    # Then check hybrid if remote not found
    if any(kw in text_lower for kw in hybrid_keywords):
        return "Hybrid"
    # Then check in-office
    if any(kw in text_lower for kw in in_office_keywords):
        return "In-Office"
    # If none matched
    return "Unknown"
# Add the new feature column to your DataFrame
df['Work_type'] = df['description'].apply(classify_work_type)

# Creating feature for AI related keywords
keywords = [
    "AI", "LLM", "NLP", "machine learning", "deep learning",
    "artificial intelligence", "chatbot", "transformer model", "generative"
]

# Build regex pattern with smart word boundaries, case-insensitive
pattern = re.compile(
    r'(?i)\b(?:' + '|'.join([
        "AI", "LLMs?", "NLP", "machine learning", "deep learning",
        "artificial intelligence", "chatbots?", "transformer models?", "generative"
    ]) + r')\b'
)

def extract_keywords(text):
    if pd.isna(text):
        return []
    matches = pattern.findall(text)
    return list(set(match.upper() for match in matches))  # deduplicate, normalize case

# Apply the function to the DataFrame
df['AI_keywords'] = df['description'].apply(extract_keywords)

# Filling nulls to load SQL tables
df.replace(['', 'nan', 'NaN', 'None', None], np.nan, inplace=True)

df['salary_max'] = df['salary_max'].fillna(df['salary_max'].median())
df['salary_min'] = df['salary_min'].fillna(df['salary_min'].median())
df['salary_standardized'] = df['salary_standardized'].fillna(df['salary_standardized'].median())
df['salary_rate'] = df['salary_rate'].fillna(df['salary_rate'].mode()[0])
df['work_from_home'] = df['work_from_home'].fillna(df['work_from_home'].mode()[0])
df['schedule_type'] = df['schedule_type'].fillna(df['schedule_type'].mode()[0])

df['State'] = df['State'].fillna(df['State'].mode()[0])

df['title'] = df['title'].fillna(df['title'].mode()[0])
df['via'] = df['via'].fillna(df['via'].mode()[0])
df['posted_at'] = df['posted_at'].fillna(df['posted_at'].mode()[0])
df['location'] = df['location'].fillna('Unknown')

df = feature_engineer_schedule_type(df)

# Rename only columns that start with lowercase letters
df = df.rename(
    columns={col: col[0].upper() + col[1:] for col in df.columns if re.match(r'^[a-z]', col)},
)

df.to_csv("data/processed/cleaned_gsearch_jobs.csv", index=False, encoding='utf-8')