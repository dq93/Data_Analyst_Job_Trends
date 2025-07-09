import os
import re
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from collections import Counter
from kaggle.api.kaggle_api_extended import KaggleApi
from sqlalchemy import create_engine

def download_dataset() -> pd.DataFrame:
    api = KaggleApi()
    api.authenticate()

    dataset_slug = 'lukebarousse/data-analyst-job-postings-google-search'
    download_folder = 'data'
    csv_filename = 'gsearch_jobs.csv'
    csv_path = os.path.join(download_folder, csv_filename)

    os.makedirs(download_folder, exist_ok=True)

    if not os.path.exists(csv_path):
        print("Downloading from Kaggle...")
        api.dataset_download_files(dataset_slug, path=download_folder, unzip=True)
    else:
        print("File already exists locally. Skipping download.")
    
    return pd.read_csv(csv_path)

def normalize_title(title):
    title = title.lower()
    title = re.sub(r'(sr\\.?|senior)', 'senior', title)
    title = re.sub(r'(jr\\.?|junior)', 'junior', title)
    title = re.sub(r'\\s*-\\s*.*$', '', title)
    title = re.sub(r'[^\\w\\s]', '', title)
    return title.strip()

def feature_engineer_schedule_type(df: pd.DataFrame) -> pd.DataFrame:
    all_types = ['Full-time', 'Part-time', 'Contractor', 'Internship', 'Temp work', 'Per diem', 'Volunteer']
    df = df.copy()
    df['schedule_type'] = df['schedule_type'].fillna('')
    for t in all_types:
        df[t] = df['schedule_type'].apply(lambda x: int(t in x))
    return df

def find_skills(text, skills):
    return [skill for skill in skills if re.search(rf"\\b{re.escape(skill)}\\b", text, re.IGNORECASE)]

def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset="job_id", keep="first")
    columns_to_drop = [
        'Unnamed: 0', 'index', 'search_term', 'search_location', 'commute_time',
        'thumbnail', 'salary', 'salary_pay', 'salary_yearly', 'salary_hourly', 'salary_avg'
    ]
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

    df["title"] = df["title"].str.replace(r"[^a-zA-Z0-9\\s,./&()\\-]", "", regex=True)
    df["title"] = df["title"].str.title()
    df["title"] = df["title"].apply(normalize_title)

    df['via'] = df['via'].str.replace(r'^via\\s+', '', regex=True).str.strip()
    df['location'] = df['location'].str.strip()

    df = feature_engineer_schedule_type(df)

    exp_pattern = r"(?:(?:at least|min(?:imum)? of)\\s*\\d+\\s*years?)|(?:\\d+\\+?\\s*[-–]?\\s*\\d*\\s*years?)"
    degree_pattern = r"(?:Bachelor(?:'s)?|BA|BS|BSc|Master(?:'s)?|MS|MSc|MBA|PhD|Doctorate|degree in [A-Za-z ]+)"
    df["Has_experience_requirement"] = df["description"].str.contains(exp_pattern, flags=re.IGNORECASE, regex=True, na=False)
    df["Has_degree_requirement"] = df["description"].str.contains(degree_pattern, flags=re.IGNORECASE, regex=True, na=False)

    skills = [
        "Python", "R", "SQL", "Java", "Scala", "Excel", "Microsoft Excel", "Tableau", "Power BI", 
        "Looker", "Google Sheets", "Matplotlib", "Seaborn", "Apache Airflow", "dbt", "Apache NiFi", 
        "SSIS", "Informatica", "Talend", "MySQL", "PostgreSQL", "Oracle", "Redshift", "Snowflake", 
        "BigQuery", "MongoDB", "AWS", "Azure", "GCP", "Google Cloud Platform", "Apache Spark", 
        "Hadoop", "Kafka", "Hive", "Presto", "Docker", "Kubernetes", "Terraform", "Git", "GitHub", 
        "Scikit-learn", "TensorFlow", "Keras", "XGBoost", "Pandas", "NumPy"
    ]
    df["description"] = df["description"].fillna("")

    for skill in skills:
        df[skill] = df["description"].str.contains(rf"\\b{re.escape(skill)}\\b", case=False, regex=True)
    
    df["skills_found"] = df["description"].apply(lambda text: find_skills(text, skills))

    df['state'] = df['location'].str.extract(r',\\s*([A-Z]{2})')
    df['state_clean'] = np.where(
        df['location'].isin(['United States', 'Anywhere']),
        df['location'],
        df['state']
    )

    text_cols = [col for col in ['description', 'extensions'] if col in df.columns]
    df['has_pay_range'] = df[text_cols].apply(
        lambda row: row.astype(str).str.contains(r'\\$\\d+', case=False, na=False).any(),
        axis=1
    )
    return df

def load_to_postgres(df: pd.DataFrame, table_name: str = "job_postings2"):
    load_dotenv()
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "5432")
    db_name = os.getenv("DB_NAME")
    db_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    engine = create_engine(db_url)
    df.to_sql(table_name, engine, if_exists="replace", index=False)
    print(f"Data loaded into table `{table_name}` in database `{db_name}`")

def main():
    df = download_dataset()
    df = transform_data(df)
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/cleaned_gsearch_jobs.csv", index=False)
    print("Cleaned data saved to: data/processed/cleaned_gsearch_jobs.csv")
    load_to_postgres(df)

if __name__ == "__main__":
    main()

""" Additonal Transformations for ETL pipeline"""
#Transform datetime to just date
df["date"] = pd.to_datetime(df["date_time"]).dt.date

# Find Min and Max experience required for role
exp_pattern = r"((at least|min(?:imum)? of)\s*\d+\s*years?)|(\d+\+?\s*[-–]?\s*\d*\s*years?)"
# Find Min and Max Degree required for role
degree_pattern = r"(Bachelor(?:'s)?|BA|BS|BSc|Master(?:'s)?|MS|MSc|MBA|PhD|Doctorate|degree in [A-Za-z ]+)"

# Create Boolean columns
df["Has_experience_requirement"] = df["description"].str.contains(exp_pattern, flags=re.IGNORECASE, regex=True).astype(int)
df["Has_degree_requirement"] = df["description"].str.contains(degree_pattern, flags=re.IGNORECASE, regex=True).astype(int)
df["Has_degree_requirement"]

# List of visa keywords to check for
visa_keywords = ["h-1b', 'h1b', 'h-2b', 'l-1a', 'l-1b', 'o-1', 'eb-2', 'eb2', 'eb-3', 'eb3', 'visa sponsorship"]

# Function to return 1 if any visa keyword is found in the description, else 0
def binary_visa_flag(description):
    desc = str(description).lower()  # ensure description is a string
    return int(any(kw in desc for kw in visa_keywords))

# Add the binary column to your existing DataFrame
df["visa_sponsorship_flag"] = df["description"].apply(binary_visa_flag)
df


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
df

