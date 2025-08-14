<img width="1536" height="1024" alt="Image" src="https://github.com/user-attachments/assets/69736c15-e31b-4237-bc2e-5598fbc98c57" />


## Project Overview

#### We are exploring job market trends for data analyst roles: skills in demand, salary shifts, and hiring patterns over time. This dataset compiles job postings for Data Analyst positions in the United States, sourced directly from Google's job search results. Data collection began on November 4th, 2022, and continues to grow with approximately 100 new postings added daily, offering a continuously updated snapshot of the current job market.

Link to dataset: https://www.kaggle.com/datasets/lukebarousse/data-analyst-job-postings-google-search

## Team members:

Kevin Bonilla<br>
LinkedIn: https://www.linkedin.com/in/kevin-bonilla-34578421a/

Maimuna Hamidu-Bawa<br>
LinkedIn: https://www.linkedin.com/in/maimuna-bawa/

Ricardi Jean<br>
LinkedIn: https://www.linkedin.com/in/ricardi-jean/

Daniel Quintanilla<br>
LinkedIn: https://www.linkedin.com/in/danieljquintanilla/

## This Project has 3 parts:
- Exploratory Data Analysis: Univariate, Bivariate, and Multivariate
- ETL Pipeline
- Tableau dashboard

## Original features of the dataset:
- title 
- company_name 
- location 
- schedule_type 
- work_from_home 
- posted_at 
- via
- salary 
- salary_rate
- salary_avg
- salary_min
- salary_max
- salary_hourly
- salary_yearly
- salary_standardized
- salary_pay
- description
- description_tokens
- extensions
- job_id
- thumbnail
- search_term
- search_location
- commute_time
- date_time
- index
- Unnamed: 0


## Final features of the dataset:

- Title
- Company_name
- Location
- Via
- Schedule_type
- Salary_standardized
- Degree_Requirement
- Has_Bachelor
- Has_Master
- Has_MBA
- Has_PhD
- Date
- Python
- R
- SQL
- Java
- Scala
- Excel
- Microsoft Excel
- Tableau
- Power BI
- Looker
- Google Sheets
- Matplotlib
- Seaborn
- Apache Airflow
- Dbt
- Apache NiFi
- SSIS
- Informatica
- Talend
- MySQL
- PostgreSQL
- Oracle
- Redshift
- Snowflake
- BigQuery
- MongoDB
- AWS
- Azure
- GCP
- Google Cloud Platform
- Apache Spark
- Hadoop
- Kafka
- Hive
- Presto
- Docker
- Kubernetes
- Terraform
- Git
- GitHub
- Scikit-learn
- TensorFlow
- Keras
- XGBoost
- Pandas
- NumPy
- Has_pay_range
- Visa_sponsorship_flag
- State
- Work_type
- AI_keywords
- Full-time
- Part-time
- Contractor
- Internship
- Temp work
- Per diem
- Volunteer
- Min_Years_Experience
- Seniority_Level


# Key findings:
The amount of jobs overall being posted is in a downtrend.
Roles labeled entry level are the most common we saw.
AI has made an appearance in job descriptions, up to 19% in 2024.
AI mentions have gone down to 15% in 2025 YTD.
Salaries have slightly decreased from an average of $90k to $89k.
Entry level averages are at $88k.
SQL is the most in demand skill with Excel, Tableau, Python and R, following in that order.


We created an Extract, Transform, Load (ETL) pipeline for this project. The data is collected via google searches into a Kaggle dataset. We use Kaggle’s API to then download a CSV file that contains the dataset. We then transformed the data after performing exploratory data analysis (EDA) using Python and its libraries such as pandas. Once the dataset has been cleaned and features we need are created, a clean version of the CSV is made, which is then loaded into Tableau. The dashboard is created and hosted on Tableau.


# Key metrics:
Data collection began in November 2022.
About 100 job postings are added daily.
Pre-clean CSV has 61953 rows and 27 features.
Cleaned CSV has 58775 rows and 72 features.
LinkedIn, Bebee, and Upwork are the top 3 job search sites.

# Tools:
- Python 🐍
- Pandas 🐼
- Matplotlib 📊
- Seaborn 🌊
- Regex 🔍
- Numpy ➗
- Counter 🔢
- OS 💻
- Zipfile 🗜️
- Tableau 📈
- Kaggle 🧠

Link to project report:
https://docs.google.com/document/d/1tRb0BxIn_sGou41dBsQ9uev3CC3PWv8sYdlx8wLXOzc/edit?tab=t.0

Link to Tableau dashboard:
https://public.tableau.com/app/profile/maimuna.hamidu.bawa/viz/JobMarketDashboard_17551838770280/Dashboard1

