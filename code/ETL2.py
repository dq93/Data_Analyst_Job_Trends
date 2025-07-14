import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# Load environment and DB connection
load_dotenv()
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")
engine = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

df = pd.read_csv('/Users/sa22/Documents/code/Data_Analyst_Job_Trends/data/cleaned_gsearch_jobs.csv')
import ast
df['skills_found'] = df['skills_found'].apply(ast.literal_eval)


companies_df = df[['company_name']].drop_duplicates().dropna()
companies_df.to_sql('companies', engine, if_exists='append', index=False)

with engine.connect() as conn:
    company_ids = dict(conn.execute(text("SELECT company_name, company_id FROM companies")).fetchall())

locations_df = df[['location', 'state']].drop_duplicates().dropna()
locations_df.columns = ['location', 'state']
locations_df.to_sql('locations', engine, if_exists='append', index=False)

with engine.connect() as conn:
    location_ids = dict(conn.execute(text("SELECT location, location_id FROM locations")).fetchall())

skills_list = sorted({skill for row in df['skills_found'] for skill in row})
skills_df = pd.DataFrame({'skill_name': skills_list})
skills_df.to_sql('skills', engine, if_exists='append', index=False)

with engine.connect() as conn:
    skill_ids = dict(conn.execute(text("SELECT skill_name, skill_id FROM skills")).fetchall())

jobs_df = df.copy()

jobs_df['company_id'] = jobs_df['company_name'].map(company_ids)
jobs_df = jobs_df.dropna(subset=['company_id'])

jobs_insert = jobs_df[['job_id', 'title', 'company_id', 'has_experience_requirement',
                       'has_degree_requirement', 'work_from_home']].copy()
jobs_insert['date_time'] = pd.Timestamp.now()

jobs_insert.to_sql('jobs', engine, if_exists='append', index=False)

rows = []
for _, row in df.iterrows():
    for skill in row['skills_found']:
        skill_id = skill_ids.get(skill)
        if skill_id:
            rows.append({'job_id': row['job_id'], 'skill_id': skill_id})

job_skills_df = pd.DataFrame(rows).drop_duplicates()
job_skills_df.to_sql('job_skills', engine, if_exists='append', index=False)
