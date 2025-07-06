
import os
import re
import numpy as np
import pandas as pd
import psycopg2
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

def normalize_title(title: str) -> str:
    title = title.lower()
    title = re.sub(r'(sr\.?|senior)', 'senior', title)
    title = re.sub(r'(jr\.?|junior)', 'junior', title)
    title = re.sub(r'\s*-\s*.*$', '', title)
    title = re.sub(r'[^\w\s]', '', title)
    return title.strip()

def feature_engineer_schedule_type(df: pd.DataFrame) -> pd.DataFrame:
    all_types = ['Full-time', 'Part-time', 'Contractor', 'Internship', 'Temp work', 'Per diem', 'Volunteer']
    df = df.copy()
    df['schedule_type'] = df['schedule_type'].fillna('')
    for t in all_types:
        df[t] = df['schedule_type'].apply(lambda x: int(t in x))
    return df

def find_skills(text: str, skills: list) -> list:
    return [skill for skill in skills if re.search(rf"\b{re.escape(skill)}\b", text, re.IGNORECASE)]

def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset="job_id", keep="first")
    columns_to_drop = [
        'Unnamed: 0', 'index', 'search_term', 'search_location', 'commute_time',
        'thumbnail', 'salary', 'salary_pay', 'salary_yearly', 'salary_hourly', 'salary_avg'
    ]
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

    df["title"] = df["title"].str.replace(r"[^a-zA-Z0-9\s,./&()\-]", "", regex=True)
    df["title"] = df["title"].str.title().apply(normalize_title)
    df['via'] = df['via'].str.replace(r'^via\s+', '', regex=True).str.strip()
    df['location'] = df['location'].str.strip()

    df = feature_engineer_schedule_type(df)

    exp_pattern = r"(?:(?:at least|min(?:imum)? of)\s*\d+\s*years?)|(?:\d+\+?\s*[-–]?\s*\d*\s*years?)"
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
        df[skill] = df["description"].str.contains(rf"\b{re.escape(skill)}\b", case=False, regex=True)
    df["skills_found"] = df["description"].apply(lambda text: find_skills(text, skills))

    df['state'] = df['location'].str.extract(r',\s*([A-Z]{2})')
    df['state_clean'] = np.where(
        df['location'].isin(['United States', 'Anywhere']),
        df['location'],
        df['state']
    )

    text_cols = [col for col in ['description', 'extensions'] if col in df.columns]
    df['has_pay_range'] = df[text_cols].apply(
        lambda row: row.astype(str).str.contains(r'\$\d+', case=False, na=False).any(),
        axis=1
    )
    return df

def create_database_schema():
    load_dotenv()
    try:
        conn = psycopg2.connect(
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT")
        )
        conn.autocommit = True
        cur = conn.cursor()

        schema_sql = """
        DROP TABLE IF EXISTS job_skills CASCADE;
        DROP TABLE IF EXISTS jobs CASCADE;
        DROP TABLE IF EXISTS companies CASCADE;
        DROP TABLE IF EXISTS locations CASCADE;
        DROP TABLE IF EXISTS skills CASCADE;

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
            skill_name TEXT UNIQUE NOT NULL
        );

        CREATE TABLE jobs (
            job_id TEXT PRIMARY KEY,
            title TEXT,
            company_id INT REFERENCES companies(company_id),
            location_id INT REFERENCES locations(location_id),
            via TEXT,
            description TEXT,
            posted_at TIMESTAMP,
            schedule_type TEXT,
            work_from_home BOOLEAN,
            date_time TIMESTAMP,
            salary_rate TEXT,
            salary_min NUMERIC,
            salary_max NUMERIC,
            salary_standardized NUMERIC,
            description_tokens TEXT[],
            has_experience_requirement BOOLEAN,
            has_degree_requirement BOOLEAN,
            has_pay_range BOOLEAN
        );

        CREATE TABLE job_skills (
            job_id TEXT REFERENCES jobs(job_id) ON DELETE CASCADE,
            skill_id INT REFERENCES skills(skill_id) ON DELETE CASCADE,
            PRIMARY KEY (job_id, skill_id)
        );
        """

        cur.execute(schema_sql)
        print("Database schema created successfully")

    except Exception as e:
        print(f"Error creating database schema: {e}")
        raise
    finally:
        if 'cur' in locals():
            cur.close()
        if 'conn' in locals():
            conn.close()

def main():
    df = download_dataset()
    df = transform_data(df)
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/cleaned_gsearch_jobs.csv", index=False)
    create_database_schema()
    print("ETL process complete.")

if __name__ == "__main__":
    main()
