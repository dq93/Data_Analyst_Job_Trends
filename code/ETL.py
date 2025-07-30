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

# ===Data transformations===

# Remove duplicates from "Job_id" and keep those that were posted first
df[df.duplicated("job_id")]
df = df.drop_duplicates(subset="job_id", keep="first")

# Dropping columns
columns_to_drop = [
    'Unnamed: 0', 'index', 'search_term', 'search_location', 'commute_time',
    'thumbnail', 'salary', 'salary_pay', 'salary_yearly', 'salary_hourly', 'salary_avg'
    'posted_at', 'job_id', 'salary_rate', 'salary_min', 'salary_max',
    'description_tokens', 'work_from_home'
]
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

# Define degree pattern
degree_pattern = r"(Bachelor(?:'s)?|BA|BS|BSc|Master(?:'s)?|MS|MSc|MBA|PhD|Doctorate|degree in [A-Za-z ]+)"

# Extract degree mentions
df["Degree_Requirement"] = df["description"].str.findall(degree_pattern, flags=re.IGNORECASE)

# Clean and join matches into one string
df["Degree_Requirement"] = df["Degree_Requirement"].apply(
    lambda x: ", ".join(set([i.strip().title() for i in x])) if x else None
)

# Create binary flags
degree_keywords = {
    "Bachelor": ["bachelor", "ba", "bs", "bsc"],
    "Master": ["master", "ms", "msc"],
    "MBA": ["mba"],
    "PhD": ["phd", "doctorate"]
}

for degree, keywords in degree_keywords.items():
    df[f"Has_{degree}"] = df["Degree_Requirement"].str.lower().apply(
        lambda text: int(any(kw in text for kw in keywords)) if isinstance(text, str) else 0
    )

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
    desc = str(description).lower()  # Ensure description is a string
    return int(any(kw in desc for kw in visa_keywords))

# Add the binary column to your existing DataFrame
df["visa_sponsorship_flag"] = df["description"].apply(binary_visa_flag)

# Using regions to find and filter states
valid_states = {
    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
    'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
    'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
    'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
    'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY',
    'DC'
}

# Define AR/VR-related keywords to avoid misclassifying "AR" as Arkansas
ar_keywords = [
    "ar/vr", "augmented reality", "virtual reality", "mixed reality", "xr",
    "spatial computing", "vr headset", "ar headset", "meta quest", "oculus"
]

# Extract (city, state) from text
def city_state(text):
    if not isinstance(text, str):
        return []
    text_lower = text.lower()
    is_ar_text = any(term in text_lower for term in ar_keywords)

    matches = re.findall(r'\b([A-Za-z\s]+?),\s*([A-Z]{2})\b', text)

    excl_non_city_keywords = {
        "are", "we", "you", "the", "they", "join", "as", "with", "and", "be", "or", "to", "in", "for"
    }

    results = []
    for city, state in matches:
        city_clean = city.strip().lower()
        if state not in valid_states:
            continue
        if city_clean in excl_non_city_keywords:
            continue
        if state == "AR" and is_ar_text:
            continue
        results.append((city.title(), state))
    return results

# Apply city/state extraction to Description
df['location_data'] = df['description'].apply(city_state)
df['location_city'] = df['location_data'].apply(lambda x: x[0][0] if x else None)
df['location_state'] = df['location_data'].apply(lambda x: x[0][1] if x else None)

# Define keyword-based region lookup
location_keywords = {
    "bay area": "CA",
    "silicon valley": "CA",
    "new york city": "NY",
    "manhattan": "NY",
    "new york": "NY",
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
    "washington dc": "DC",
    "austin": "TX", 
    "houston": "TX", 
    "dallas": "TX", 
    "san antonio": "TX",
    "denver": "CO", 
    "boulder": "CO", 
    "miami": "FL", 
    "orlando": "FL",
    "phoenix": "AZ", 
    "scottsdale": "AZ", 
    "pittsburgh": "PA",
    "san diego": "CA", 
    "oakland": "CA", 
    "sacramento": "CA",
    "raleigh": "NC", 
    "charlotte": "NC", 
    "nashville": "TN",
    "columbus": "OH", 
    "minneapolis": "MN", 
    "st. paul": "MN",
    "detroit": "MI", 
    "indianapolis": "IN"
}

def find_region_keywords(text):
    text = text.lower()
    for keyword, state in location_keywords.items():
        if keyword in text:
            return keyword, state
    return None, None

# Apply keyword-based region match
df[['region_keyword', 'region_state']] = df['description'].apply(lambda x: pd.Series(find_region_keywords(x)))

# Choose the best state result: from regex match first, then keyword
def choose_state(row):
    return row['location_state'] if pd.notnull(row['location_state']) else row['region_state']

df['state_result'] = df.apply(choose_state, axis=1)

# Also parse the original Location field as backup
df['loc_data'] = df['location'].apply(city_state)
df['loc_city'] = df['loc_data'].apply(lambda x: x[0][0] if x else None)
df['loc_state'] = df['loc_data'].apply(lambda x: x[0][1] if x else None)

# Use loc_state if state_result is still missing
df['state_result'] = df['state_result'].fillna(df['loc_state'])

# Final location classification
def classify_us_locations(row):
    return row['state_result'] if pd.notnull(row['state_result']) else 'US'

df['State'] = df.apply(classify_us_locations, axis=1)

# Droppping state related columns no longer needed
df = df.drop(columns=['location_data', 'location_city', 'state_result', 'region_keyword', 'region_state', 'location_state',
                      'loc_data', 'loc_city', 'loc_state'])

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
# Add the new feature to your DataFrame
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
    return list(set(match.upper() for match in matches))  # Deduplicate, normalize case

# Apply the function to the DataFrame
df['AI_keywords'] = df['description'].apply(extract_keywords)

# Filling any remaining nulls in features
df.replace(['', 'nan', 'NaN', 'None', None], np.nan, inplace=True)

df['salary_standardized'] = df['salary_standardized'].fillna(df['salary_standardized'].median())
df['schedule_type'] = df['schedule_type'].fillna(df['schedule_type'].mode()[0])
df['State'] = df['State'].fillna(df['State'].mode()[0])
df['title'] = df['title'].fillna(df['title'].mode()[0])
df['via'] = df['via'].fillna(df['via'].mode()[0])
df['location'] = df['location'].fillna('Unknown')

df = feature_engineer_schedule_type(df)

# Rename columns that start with lowercase letters into uppercase
df = df.rename(
    columns={col: col[0].upper() + col[1:] for col in df.columns if re.match(r'^[a-z]', col)},
)

# Creating feature for minimum years of experience
yrs_exp_pattern = r'(?i)(?:at least|min(?:imum)? of)?\s*(\d+)\+?\s*(?:[-–to]{1,3}\s*(\d+))?\s+years?'

def extract_min_experience(text):
    matches = re.findall(yrs_exp_pattern, text)
    if matches:
        return int(matches[0][0])
    else:
        return np.nan

# Extract min years from description
df["Min_Years_Experience"] = df["Description"].apply(extract_min_experience)

# Handle missing (optional: fill with 0 or keep NaN)
df["Min_Years_Experience"] =  df["Min_Years_Experience"].fillna(0)

# Bin the experience years
def bin_experience(min_years):
    if pd.isna(min_years):
        return "Not Specified"
    elif min_years <= 3:
        return "0-3 years"
    elif 4 <= min_years <= 6:
        return "4-6 years"
    else:
        return "7+ years"

df["Experience_Bin"] = df["Min_Years_Experience"].apply(bin_experience)

# Map bins to seniority levels
def seniority_level(exp_bin):
    if exp_bin in ["0-3 years", "No Experience Required"]:
        return "Entry-Level"
    elif exp_bin == "4-6 years":
        return "Mid-Level"
    elif exp_bin == "7+ years":
        return "Senior-Level"
    else:
        return "Unknown"

df["Seniority_Level"] = df["Experience_Bin"].apply(seniority_level)

# Dropping columns that are no longer needed after info is extracted from them
df = df.drop(columns= [
    "Description", "Extensions", "Date_time", "Experience_Bin",
      "Skills_found", "Posted_at", "Salary_avg"
    ])

# ===Loading to CSV===
df.to_csv("data/processed/cleaned_gsearch_jobs.csv", index=False, encoding='utf-8')