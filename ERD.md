We could create 2 tables:
Job Listing and Salary

Job Listing would have: 
job_id as primary key, title, company_name, location, via, description, extensions, posted_at, schedule_type, description_tokens

Salary would have:
job_id as a foreign key, salary_rate, salary_avg, salary_min, salary_max, salary_standardized, salary_hourly

note: some features were not considered as they may be dropped.
