+-----------------+         +-------------+        +-------------------+
|   JobListing    |         |   Salary    |        |  SearchMetadata   |
+-----------------+         +-------------+        +-------------------+
| job_id (PK)     |<--------| job_id (FK) |        | search_term       |
| title           |         | salary      |        | search_location   |
| company_name    |         | salary_pay  |        | date_time         |
| location        |         | salary_rate |        +-------------------+
| via             |         | salary_avg  |
| description     |         | salary_min  |
| extensions      |         | salary_max  |
| thumbnail       |         | salary_hourly|
| posted_at       |         | salary_yearly|
| schedule_type   |         | salary_standardized |
| work_from_home  |         +-------------+
| commute_time    |
| description_tokens |
+-----------------+
