"""ERD for skills table"""
CREATE TABLE skills (
	skill_id SERIAL PRIMARY KEY,
	skill_name TEXT UNIQUE NOT NULL
);

CREATE TABLE job_skills (
	job_id TEXT REFERENCES jobs(job_id),
	skill_id INT REFERENCES skills(skill_id),
	PRIMARY KEY (job_id, skill_id)
);
"""Adding to jobs table"""

CREATE TABLE jobs (
    job_id TEXT PRIMARY KEY,
    title TEXT,
    company_id INT REFERENCES companies(company_id),
    date_time TIMESTAMP,
    has_experience_requirement BOOLEAN,
    has_degree_requirement BOOLEAN,
    work_from_home BOOLEAN
);
 
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