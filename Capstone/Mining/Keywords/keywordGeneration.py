import pandas as pd

# Load Excel
df = pd.read_excel("Mining\Keywords\list.xlsx", sheet_name="Sheet1")

def build_search_queries(row):
    templates = [
        "site:leetcode.com {company} {role} {experience} online assessment questions",
        "site:geeksforgeeks.org {company} {role} {experience} interview experience",
        "site:glassdoor.com {company} {role} {experience} interview questions",
        "site:reddit.com {company} {role} {experience} oa round",
        "site:linkedin.com {company} {role} {experience} coding questions",
        "site:github.com {company} {role} {experience} interview prep",
    ]

    return [template.format(
        company=row['Company'], 
        role=row['Role'], 
        experience=row['YoE']) for template in templates]

# Generate search queries for each row
all_queries = []
for _, row in df.iterrows():
    all_queries.extend(build_search_queries(row))

# Save to file
with open("search_queries.txt", "w", encoding="utf-8") as f:
    for query in all_queries:
        f.write(query + "\n")

print(f"✅ Generated {len(all_queries)} search queries.")
