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
