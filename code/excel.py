import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os



def export_tables_to_excel():
    # --- Load credentials from .env file ---
    load_dotenv()

    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME")

    # --- Set up PostgreSQL connection ---
    engine = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

    # --- List of tables to export ---
    tables = ["jobs", "companies", "locations", "skills", "job_skills"]

    # --- Output directory ---
    output_dir = "/Users/sa21/Desktop/Data_Analyst_Job_Trends/tables"
    os.makedirs(output_dir, exist_ok=True)

    # --- Export each table as a separate Excel file ---
    for table in tables:
        df = pd.read_sql(f"SELECT * FROM {table}", engine)
        output_path = os.path.join(output_dir, f"{table}_table.xlsx")
        df.to_excel(output_path, index=False)
        print(f"Exported: {output_path}")

if __name__ == "__main__":
    export_tables_to_excel()

# Read the cleaned CSV file 
df1 = pd.read_csv("/Users/sa21/Desktop/Data_Analyst_Job_Trends/data/cleaned_gsearch_jobs.csv")

# Save as Excel to a subfolder
df1.to_excel('/Users/sa21/Desktop/Data_Analyst_Job_Trends/data/processed/cleaned_gsearch_jobs.xlsx', index=False)

