import os
import zipfile
import pandas as pd
import re
from collections import Counter
import numpy as np
from dotenv import load_dotenv
from sqlalchemy import create_engine
from dotenv import load_dotenv
from sqlalchemy import create_engine
import os

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

"""
Download a Kaggle dataset of data analyst job postings, extracts the ZIP file, 
and load the job data into a pandas DataFrame.
"""

os.system("kaggle datasets download -d lukebarousse/data-analyst-job-postings-google-search")

with zipfile.ZipFile("data-analyst-job-postings-google-search.zip", 'r') as zip_ref:
    zip_ref.extractall("data")

df = pd.read_csv("data/gsearch_jobs.csv")

# Data transformation

df[df.duplicated("job_id")]

# Remove duplicates from "Job_id" and keep those that were posted first
df = df.drop_duplicates(subset="job_id", keep="first")

# dropping columns
columns_to_drop = [
    'Unnamed: 0', 'index', 'search_term', 'search_location', 'commute_time',
    'thumbnail', 'salary', 'salary_pay', 'salary_yearly', 'salary_hourly', 'salary_avg'
]
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

#Remove any strange charater from "title"
df["title"] = df["title"].str.replace(r"[^a-zA-Z0-9\s,./\-&()]", "", regex=True)

# Capitalize the first letter of each word in the job titles
df["title"]= df["title"].str.title()

# removes the prefix via from the via column
df['via'] = df['via'].str.replace(r'^via\s+', '', regex=True).str.strip()

# removes whitespace around locations in the location column
df['location'] = df['location'].str.strip()

df["date"] = pd.to_datetime(df["date_time"]).dt.date

# function to standardize schedule_type into binary Columns
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

# function for standardizing job titles
def normalize_title(title):
    title = title.lower() # makes lowercase
    title = re.sub(r'(sr\.?|senior)', 'senior', title) # standardize senior
    title = re.sub(r'(jr\.?|junior)', 'junior', title) # standardize junior
    title = re.sub(r'\s*-\s*.*$', '', title) # removes suffices like '-contract to hire'
    title = re.sub(r'[^\w\s]', '', title) # removes punctiation
    title = title.strip()
    return title

df['title'] = df['title'].apply(normalize_title)

# Create Boolean columns to indicate whether a job description mentions
# experience or degree requirements based on regex pattern matching
df["Has_experience_requirement"] = df["description"].str.contains(exp_pattern, flags=re.IGNORECASE)
df["Has_degree_requirement"] = df["description"].str.contains(degree_pattern, flags=re.IGNORECASE, regex=True)

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

print('creating features, this may take a couple of minutes')

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

# creates state and state_clean features
df['state'] = df['location'].str.extract(r',\s*([A-Z]{2})')
df['state_clean'] = np.where(
    df['location'].isin(['United States', 'Anywhere']),
    df['location'],
    df['state']
)

df = df.drop('state', axis=1)
df = df.rename(columns={"state_clean": "state"})

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

#filling nulls to load SQL tables
df.replace(['', 'nan', 'NaN', 'None', None], np.nan, inplace=True)

df['salary_max'] = df['salary_max'].fillna(df['salary_max'].median())
df['salary_min'] = df['salary_min'].fillna(df['salary_min'].median())
df['salary_standardized'] = df['salary_standardized'].fillna(df['salary_standardized'].median())
df['salary_rate'] = df['salary_rate'].fillna(df['salary_rate'].mode()[0])
df['work_from_home'] = df['work_from_home'].fillna(df['work_from_home'].mode()[0])
df['schedule_type'] = df['schedule_type'].fillna(df['schedule_type'].mode()[0])
df['state'] = df['state'].fillna(df['state'].mode()[0])
df['title'] = df['title'].fillna(df['title'].mode()[0])
df['via'] = df['via'].fillna(df['via'].mode()[0])
df['posted_at'] = df['posted_at'].fillna(df['posted_at'].mode()[0])
df['location'] = df['location'].fillna('Unknown')

df = feature_engineer_schedule_type(df)
from sqlalchemy.orm import sessionmaker



df.to_csv("/Users/sa21/Desktop/Data_Analyst_Job_Trends/data/cleaned_gsearch_jobs.csv", index=False)

# Load the cleaned CSV
load_dotenv()
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")

# Connect to PostgreSQL
engine = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

# Load the cleaned CSV
df = pd.read_csv("data/cleaned_gsearch_jobs.csv")

# === CREATE TABLES ===
create_tables_sql = """
DROP TABLE IF EXISTS job_skills, skills, jobs, locations, companies CASCADE;

CREATE TABLE companies (
    company_id SERIAL PRIMARY KEY,
    company_name TEXT UNIQUE
);

CREATE TABLE locations (
    location_id SERIAL PRIMARY KEY,
    location TEXT,
    state TEXT,
    state_clean TEXT
);

CREATE TABLE skills (
    skill_id SERIAL PRIMARY KEY,
    skill_name TEXT UNIQUE
);

CREATE TABLE jobs (
    job_id TEXT PRIMARY KEY,
    company_id INT,
    date_time TIMESTAMP,
    has_experience_requirement BOOLEAN,
    has_degree_requirement BOOLEAN,
    work_from_home BOOLEAN
);

CREATE TABLE job_skills (
    job_id TEXT,
    skill_id INT,
    PRIMARY KEY (job_id, skill_id),
    FOREIGN KEY (job_id) REFERENCES jobs(job_id),
    FOREIGN KEY (skill_id) REFERENCES skills(skill_id)
);
"""

with engine.connect() as conn:
    for statement in create_tables_sql.strip().split(";"):
        if statement.strip():
            conn.execute(text(statement.strip()))
    conn.commit()
print("Tables created successfully.")

# === Insert into companies ===
df['company_id'] = df.groupby('company_name').ngroup() + 1
companies_df = df[['company_id', 'company_name']].drop_duplicates()
companies_df.to_sql("companies", engine, if_exists="append", index=False)

# === Insert into locations ===
df['location_id'] = df.groupby(['location', 'state']).ngroup() + 1
locations_df = df[['location_id', 'location', 'state']].drop_duplicates()
locations_df['state_clean'] = locations_df['state']
locations_df.to_sql("locations", engine, if_exists="append", index=False)

# === Insert into jobs ===
jobs_df = df[['job_id', 'company_id', 'date_time', 'Has_experience_requirement',
              'Has_degree_requirement', 'work_from_home']].copy()
jobs_df = jobs_df.rename(columns={
    'Has_experience_requirement': 'has_experience_requirement',
    'Has_degree_requirement': 'has_degree_requirement'
})
jobs_df.to_sql("jobs", engine, if_exists="append", index=False)

# === Handle skills ===
skills = [
    "Python", "R", "SQL", "Java", "Scala", "Excel", "Microsoft Excel", "Tableau", "Power BI", 
    "Looker", "Google Sheets", "Matplotlib", "Seaborn", "Apache Airflow", "dbt", "Apache NiFi", 
    "SSIS", "Informatica", "Talend", "MySQL", "PostgreSQL", "Oracle", "Redshift", "Snowflake", 
    "BigQuery", "MongoDB", "AWS", "Azure", "GCP", "Google Cloud Platform", "Apache Spark", 
    "Hadoop", "Kafka", "Hive", "Presto", "Docker", "Kubernetes", "Terraform", "Git", "GitHub", 
    "Scikit-learn", "TensorFlow", "Keras", "XGBoost", "Pandas", "NumPy"
]

# 1. Insert into `skills` table
skill_df = pd.DataFrame({'skill_name': sorted(skills)})
skill_df.to_sql("skills", engine, if_exists="append", index=False)

# 2. Map skill_name to skill_id
with engine.connect() as conn:
    result = conn.execute(text("SELECT skill_id, skill_name FROM skills"))
    skill_map = {row.skill_name: row.skill_id for row in result}

# 3. Insert into `job_skills`
job_skills_records = []
for i, row in df.iterrows():
    for skill in row['skills_found'].strip("[]").replace("'", "").split(", "):
        if skill in skill_map:
            job_skills_records.append({
                'job_id': row['job_id'],
                'skill_id': skill_map[skill]
            })

job_skills_df = pd.DataFrame(job_skills_records)
job_skills_df.to_sql("job_skills", engine, if_exists="append", index=False)

print("Data loaded into PostgreSQL.")
