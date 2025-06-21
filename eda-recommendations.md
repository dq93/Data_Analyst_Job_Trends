# EDA-Based Data Transformation Recommendations

## 1. Standardize `schedule_type` into Binary Columns
**Why**: The original `schedule_type` field includes multiple job types in one string, including ones that combine two or more types, making it hard to analyze.
**Recommendation**: Split this into binary flags like `Full-time`, `Part-time`, etc.

## 2. Clean `work_from_home` Field
**Why**: The field includes inconsistent formatting and missing values.
**Recommendation**: Standardize to `True`, `False`, or `Missing`. Consider renaming to `remote_option` for clarity.

## 3. Create `desc_wordcount`
**Why**: Description length may indicate posting quality or pay transparency.
**Recommendation**: Use `.apply(lambda x: len(x.split()))` on `description` and store in `desc_wordcount`.

## 4. Identify Instances of Pay Information in string columns.
**Why**: Many postings mention salary in text even if no numeric field is present.
**Recommendation**: Create a `has_pay_range` flag if pay appears in `description` or `extensions`.

## 5. Clean `state` Data
**Why**: Some location entries are inconsistent (`Anywhere`, `United States`, etc.)
**Recommendation**: Extract U.S. state abbreviations using regex and group others under `Non-specific`. Maybe compare states without the inconsistencies.

## 6. Separate Yearly vs Hourly Salaries
**Why**: The `salary_avg` column includes both annual and hourly wages, which can skew analysis if treated the same. For example, an hourly wage of $50 and a yearly salary of $100,000 exist on different scales and contexts.

**How**: We assumed that values:
- **Above $10,000** represent **yearly salaries**.
- **Below $200** represent **hourly wages**.

**Recommendation**:  
Maintain two separate subsets:
- `yearly_salaries` for aggregate or distribution analysis of full-time roles.
- `hourly_salaries` for roles like part-time or contract work.

This improves clarity and avoids distortion in pay-related plots.

## 7. Prepare Filtered Dataset for Salary and Commute Analysis
**Why**:  
We are exploring the relationship between salary and commute time, but these columns include missing and non-numeric values. Accurate analysis requires clean, numeric data.

**How**:  
- Selected only the `salary_avg` and `commute_time` columns.
- Converted both to numeric, coercing errors into NaNs.
- Dropped rows with missing values to ensure a clean dataset.

**Recommendation**:  
Use the resulting `filtered` dataset for correlation analysis, scatterplots, or regression. It ensures consistent and valid comparisons without noise from invalid entries.