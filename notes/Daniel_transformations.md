Features that need to be dropped:

Unnamed: 0 - self explanatory why this needs to be dropped
Index - this doesn't serve any real purpose as usual
search_term - the word/words that were entered at the time to make the search, therefore 'data analyst' is what appears for all rows
search_location - much like search_term, US is what appears for all rows because that is where the person searching is located
commute_time - has all nulls
thumbnail - there is no clear way to make any use of this since we can only see the metadata and not the image itself
salary - attempts were made to standardize this, only to realize that salary_standarized already has this covered
salary_pay - same as salary, it is way too all over the place and there is another feature that standardizes this
salary_yearly - has 57884 nulls and again, the salary_standarized column covers most of the salary related columns
work_from_home - has 33973 nulls out of the 61953 rows. the rest are all true so no false exists here
job_id - it's only purpose is to check for duplicate job postings, and then it can be dropped

Features that need to be normalized:

location - locations need to have whitespace around them removed
title - needs to combine prefixes like sr and senior being combined into one, as well as many other transformations
    such as punctutation, all made into lowercase, removing suffices, and possibly more.
via - each word needs to have the word via in front of it removed in order to prevent repeats being considered different
description - needs to be tokenized and then stop words have to be used in order to make use of this feature
description_tokens - also needs to be tokenized and then the most common words can be pulled to make this useful
extensions - may also need to be tokenized