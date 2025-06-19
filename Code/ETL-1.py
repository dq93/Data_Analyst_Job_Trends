import os
import zipfile
import pandas as pd


"""
Download a Kaggle dataset of data analyst job postings, extracts the ZIP file, 
and load the job data into a pandas DataFrame.
"""

os.system("kaggle datasets download -d lukebarousse/data-analyst-job-postings-google-search")

with zipfile.ZipFile("data-analyst-job-postings-google-search.zip", 'r') as zip_ref:
    zip_ref.extractall("your_extract_folder")

df = pd.read_csv("/Users/sa19/Desktop/Data_Analyst_Job_Trends/Extracted_Data_File/gsearch_jobs.csv")



# Data transformation

df.head()

df.columns

df.describe()

df.dtypes

df[df.duplicated("job_id")]

# Remove duplicates from "Job_id" and keep those that were posted first
df = df.drop_duplicates(subset="job_id", keep="first")

#Remove any strange charater from "title"
df["title"] = df["title"].str.replace(r"[^a-zA-Z0-9\s,./\-&()]", "", regex=True)


# Capitalize the first letter of each word in the job titles
df["title"]= df["title"].str.title()



# Find Min and Max experience required for role
exp_pattern = r"((at least|min(?:imum)? of)\s*\d+\s*years?)|(\d+\+?\s*[-–]?\s*\d*\s*years?)"
# Find Min and Max Degree required for role
degree_pattern = r"(Bachelor(?:'s)?|BA|BS|BSc|Master(?:'s)?|MS|MSc|MBA|PhD|Doctorate|degree in [A-Za-z ]+)"

# Create Boolean columns to indicate whether a job description mentions
# experience or degree requirements based on regex pattern matching
df["Has_experience_requirement"] = df["description"].str.contains(exp_pattern, flags=re.IGNORECASE, regex=True)
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

# Create one column for each skill (True if the skill is mentioned)
for skill in skills:
    df[skill] = df["description"].str.contains(rf"\b{re.escape(skill)}\b", case=False, regex=True)

# Create a list of skills found in each row
def find_skills(text):
    return [skill for skill in skills if re.search(rf"\b{re.escape(skill)}\b", text, re.IGNORECASE)]

df["skills_found"] = df["description"].apply(find_skills)

# Count how often each skill appears
all_skills = [skill for row in df["skills_found"] for skill in row]
skill_counts = Counter(all_skills)

# Put into a DataFrame
skill_df = pd.DataFrame(skill_counts.most_common(20), columns=["Skill", "Frequency"])





