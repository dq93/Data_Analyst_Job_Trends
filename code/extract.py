import os
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi


api = KaggleApi()
api.authenticate()

dataset_slug = 'lukebarousse/data-analyst-job-postings-google-search'
download_folder = 'data'
csv_filename = 'gsearch_jobs.csv'
csv_path = os.path.join(download_folder, csv_filename)

os.makedirs(download_folder, exist_ok=True)


api.dataset_download_files(dataset_slug, path=download_folder, unzip=True)

if not os.path.exists(csv_path):
    print("Downloading from Kaggle...")
    api.dataset_download_files(dataset_slug, path=download_folder, unzip=True)
else:
    print("File already exists locally. Skipping download.")

df = pd.read_csv('/Users/sa21/Desktop/Data_Analyst_Job_Trends/data/gsearch_jobs.csv')  # adjust filename
df.head()




