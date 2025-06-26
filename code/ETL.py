import os
import re
from typing import List, Dict, Optional

import pandas as pd
from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.preprocessing import MultiLabelBinarizer
from sqlalchemy import create_engine


def download_dataset() -> str:
    """Download dataset from Kaggle if it doesn't exist locally."""
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
    
    return csv_path


def clean_schedule_type(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and one-hot encode schedule_type column."""
    # Fill nulls with empty strings
    df['schedule_type'] = df['schedule_type'].fillna('').astype(str).str.lower()

    # Standardize text
    df['schedule_type'] = (
        df['schedule_type']
        .str.replace(' and ', ',', regex=False)
        .str.replace('/', ',', regex=False)
        .str.replace(';', ',', regex=False)
        .str.strip()
    )

    # Convert string to list of schedule types
    df['schedule_type'] = df['schedule_type'].apply(
        lambda x: [s.strip() for s in x.split(',') if s.strip()]
    )

    # Rename variations for consistency
    schedule_renames = {
        'full time': 'full-time',
        'fulltime': 'full-time',
        'part time': 'part-time',
        'parttime': 'part-time',
        'contractor': 'contract',
        'contract position': 'contract',
        'temporary': 'temp',
        'seasonal': 'seasonal'
    }

    df['schedule_type'] = df['schedule_type'].apply(
        lambda lst: [schedule_renames.get(s, s) for s in lst]
    )

    # One-hot encode using MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    schedule_dummies = pd.DataFrame(
        mlb.fit_transform(df['schedule_type']),
        columns=[f'is_{s}' for s in mlb.classes_],
        index=df.index
    )

    return pd.concat([df.drop('schedule_type', axis=1), schedule_dummies], axis=1)


def clean_salary(s: str) -> Optional[float]:
    """Clean and standardize salary values."""
    if pd.isnull(s):
        return None

    s = str(s).lower()
    s = s.replace('$', '').replace(',', '').replace('usd', '').strip()

    # Convert K to thousand
    s = re.sub(r'(\d+)k', lambda m: str(int(m.group(1)) * 1000), s)

    # If it's a range like "50000-70000", take average
    if '-' in s:
        parts = s.split('-')
        try:
            low = float(parts[0])
            high = float(parts[1])
            return (low + high) / 2
        except ValueError:
            return None
    else:
        try:
            return float(s)
        except ValueError:
            return None


def clean_salary_column(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the salary column and fill missing values."""
    df['salary_cleaned'] = df['salary'].apply(clean_salary)
    mean_salary = df['salary_cleaned'].mean()
    df['salary_cleaned'] = df['salary_cleaned'].fillna(mean_salary)
    return df


def load_to_postgres(df: pd.DataFrame, table_name: str = "job_postings") -> None:
    """Load DataFrame to PostgreSQL database."""
    # Load environment variables
    load_dotenv()
    
    # Get credentials from environment
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "5432")
    db_name = os.getenv("DB_NAME")

    # Build connection string and create engine
    db_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    engine = create_engine(db_url)

    # Upload to PostgreSQL
    df.to_sql(table_name, engine, if_exists="replace", index=False)
    print(f"Data loaded into table `{table_name}` in database `{db_name}`")


def main():
    # Download data if needed
    raw_csv_path = download_dataset()
    
    # Define paths
    input_path = "data/gsearch_jobs.csv"
    output_path = "data/processed/cleaned_gsearch_jobs.csv"
    
    # Create processed directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(" Loading raw data...")
    df = pd.read_csv(input_path)

    # Clean data
    df = clean_schedule_type(df)
    df = clean_salary_column(df)

    # Save cleaned data
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to: {output_path}")

    # Load to PostgreSQL
    load_to_postgres(df)


if __name__ == "__main__":
    main()