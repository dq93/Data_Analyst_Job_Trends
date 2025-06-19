from kaggle.api.kaggle_api_extended import KaggleApi
import os
import zipfile

def download_csv(output_dir: str = "data"):
    api = KaggleApi()
    api.authenticate()

    dataset = "lukebarousse/data-analyst-job-postings-google-search"
    file_name = "gsearch_jobs.csv"

    os.makedirs(output_dir, exist_ok=True)
    zip_path = os.path.join(output_dir, file_name + ".zip")

    # Download .zip file
    api.dataset_download_file(dataset, file_name, path=output_dir, force=True)

    # Unzip
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
        print(f"Extracted to {output_dir}")

    # Optional: delete zip file
    os.remove(zip_path)
    print(f"Removed zip file: {zip_path}")

    return os.path.join(output_dir, file_name)

download_csv()