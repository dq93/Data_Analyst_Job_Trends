import os
import re
import json
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from kaggle.api.kaggle_api_extended import KaggleApi

# Load environment and DB connection
load_dotenv()
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")
ENGINE = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")


# Download dataset
def download_dataset() -> pd.DataFrame:
    csv_path = 'data/gsearch_jobs.csv'
    if not os.path.exists(csv_path):
        os.makedirs("data", exist_ok=True)
        print("📥 Downloading dataset from Kaggle...")
        KaggleApi().authenticate()
        KaggleApi().dataset_download_files(
            'lukebarousse/data-analyst-job-postings-google-search',
            path='data', unzip=True
        )
    return pd.read_csv(csv_path)


# Normalize title
def normalize_title(t: str) -> str:
    if not isinstance(t, str): return ""
    t = t.lower()
    t = re.sub(r'(sr\.?|senior)', 'senior', t)
    t = re.sub(r'(jr\.?|junior)', 'junior', t)
    t = re.sub(r'\s*-\s*.*$', '', t)
    return re.sub(r'[^\w\s]', '', t).strip().title()


# Skill list and matcher
SKILLS = ["Python", "R", "SQL", "Java", "Scala", "Excel", "Tableau",
          "Power BI", "Looker", "AWS", "GCP", "Spark", "Docker",
          "Kubernetes", "Pandas", "NumPy", "TensorFlow", "PyTorch"]

def extract_skills(text: str) -> list[str]:
    return [s for s in SKILLS if re.search(rf"\b{re.escape(s)}\b", text, re.IGNORECASE)]


# Transform raw df
def transform(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset="job_id").copy()
    df['company_name'] = df['company_name'].fillna('Unknown').str.strip()
    df['title'] = df['title'].apply(normalize_title)
    df['location'] = df['location'].fillna('Unknown').str.strip()
    df['via'] = df['via'].fillna('').str.replace(r'^via\s+', '', regex=True).str.strip()
    df['description'] = df['description'].fillna('').astype(str)

    df['skills_found'] = df['description'].apply(extract_skills)
    df['has_experience_requirement'] = df['description'].str.contains(r'\d+\+\s*(years|yrs)', case=False, na=False)
    df['has_degree_requirement'] = df['description'].str.contains(r"(Bachelor|Master|Ph\.?D|degree)", case=False, na=False)
    df['has_pay_range'] = df['description'].str.contains(r'\$\d+', na=False)

    df['state'] = df['location'].str.extract(r',\s*([A-Z]{2})')
    df['state_clean'] = np.where(df['location'].isin(['United States', 'Anywhere']),
                                  df['location'], df['state'])
    # Safe salary parsing (adapt as needed if salary columns exist)
    df['salary_rate'] = None
    df['salary_min'] = None
    df['salary_max'] = None
    df['salary_standardized'] = None

    # Tokenize description
    df['description_tokens'] = df['description'].str.lower().str.split()
    return df


# Helper for unique mappings
def insert_and_map(df: pd.DataFrame, table: str, field: str, id_field: str, extras=None) -> dict:
    df = df.drop_duplicates(subset=field)
    cols = [field] + (extras or [])
    df[cols].to_sql(table, ENGINE, if_exists="append", index=False, method='multi')
    with ENGINE.connect() as conn:
        rows = conn.execute(text(f"SELECT {id_field}, {field} FROM {table}")).fetchall()
    return {val: key for key, val in rows}


# Populate all tables
def populate_all(df: pd.DataFrame):
    df = transform(df)

    company_map = insert_and_map(df[['company_name']], 'companies', 'company_name', 'company_id')
    loc_map = insert_and_map(df[['location', 'state', 'state_clean']].drop_duplicates(),
                              'locations', 'location', 'location_id', extras=['state', 'state_clean'])
    skill_map = insert_and_map(pd.DataFrame({'skill_name': sorted({s for sl in df['skills_found'] for s in sl})}),
                                'skills', 'skill_name', 'skill_id')

    jobs = []
    for idx, row in df.iterrows():
        company_id = company_map.get(row['company_name'].strip())
        location_id = loc_map.get(row['location'].strip())

        if not row['job_id']:
            print(f"❌ Missing job_id, skipping row")
            continue

        if company_id is None:
            print(f"❌ Company not found: '{row['company_name']}'")
            continue

        if location_id is None:
            print(f"❌ Location not found: '{row['location']}'")
            continue

        jobs.append({
            'job_id': row['job_id'],
            'title': row['title'],
            'company_id': company_map[row['company_name']],
            'location_id': loc_map[row['location']],
            'via': row['via'],
            'description': row['description'],
            'posted_at': None,
            'schedule_type': row.get('schedule_type', ''),
            'work_from_home': bool(row.get('Work from home', False)),
            'date_time': pd.Timestamp.now(),
            'salary_rate': row['salary_rate'],
            'salary_min': row['salary_min'],
            'salary_max': row['salary_max'],
            'salary_standardized': row['salary_standardized'],
            'description_tokens': row['description_tokens'],
            'has_experience_requirement': bool(row['has_experience_requirement']),
            'has_degree_requirement': bool(row['has_degree_requirement']),
            'has_pay_range': bool(row['has_pay_range']),
        })

    pd.DataFrame(jobs).to_sql('jobs', ENGINE, if_exists="append", index=False, method='multi')

    # Populate join table
    rows = []
    for _, row in df.iterrows():
        for skill in row['skills_found']:
            sid = skill_map.get(skill)
            if sid:
                rows.append({'job_id': row['job_id'], 'skill_id': sid})
    pd.DataFrame(rows).drop_duplicates().to_sql('job_skills', ENGINE, index=False,
                                               if_exists="append", method='multi')

    print("✅ Populated all tables successfully.")


def main():
    df = download_dataset()
    populate_all(df)

if __name__ == "__main__":
    main()