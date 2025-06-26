import pandas as pd
from sqlalchemy import create_engine
import os

# Load the cleaned CSV
csv_path = "data/processed/cleaned_gsearch_jobs.csv"
df = pd.read_csv(csv_path)

# PostgreSQL connection details
db_user = 'your_username'       # e.g., 'postgres'
db_password = 'your_password'   # e.g., 'admin'
db_host = 'localhost'
db_port = '5432'
db_name = 'job_data'
table_name = 'job_postings'

# SQLAlchemy engine
db_url = f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'
engine = create_engine(db_url)

# Load data to PostgreSQL
df.to_sql(table_name, engine, if_exists='replace', index=False)
print(f"Loaded data to table: {table_name} in database: {db_name}")
