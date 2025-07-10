from together import Together
import pandas as pd
from tqdm import tqdm
import json
import re
import os
from dotenv import load_dotenv
load_dotenv()


# === SETUP ===
google_api_key=os.environ["GOOGLE_API_KEY"]
API_KEY = os.environ["TOGETHER_API_KEY"]
MODEL = "deepseek-ai/DeepSeek-V3" 
INPUT_CSV = "DatabaseCreation\scraped_results.csv"
OUTPUT_CSV = "questions_tagged_together.csv"

client = Together(api_key=API_KEY)
df = pd.read_csv(INPUT_CSV)
tqdm.pandas()

# Load progress if exists
if os.path.exists(OUTPUT_CSV):
    output_df = pd.read_csv(OUTPUT_CSV)
    processed_links = set(output_df["source_link"])
else:
    output_df = pd.DataFrame(columns=["company", "role", "experience_level", "question", "round", "topic", "difficulty", "source_link"])
    processed_links = set()

# 🔧 Extract experience level heuristically
def infer_experience(query: str) -> str:
    q = query.lower()
    if "new grad" in q or "fresher" in q or "college grad" in q:
        return "New Grad"
    elif "intern" in q:
        return "Intern"
    elif "0-1" in q or "0-2" in q or "entry" in q:
        return "Entry-Level"
    elif "senior" in q or "sde 3" in q:
        return "Senior"
    elif "mid" in q:
        return "Mid-Level"
    else:
        return "Unspecified"

def build_prompt(content, company, role):
    return f"""
You are a data extractor.

From the interview material below for company **{company}** and role **{role}**, extract a list of interview questions.

Respond ONLY with a JSON array. No explanation or comments.

Format:
[
  {{
    "question": "...",
    "round": "OA / Technical / HR / Design",
    "topic": "e.g. DP, Graphs, LLD, System Design",
    "difficulty": "Easy / Medium / Hard"
  }},
  ...
]

Interview content:
{content[:2500]}
"""

def extract_questions(prompt):
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = response.choices[0].message.content
        json_str = re.search(r"\[.*\]", raw, re.DOTALL)
        if json_str:
            return json.loads(json_str.group())
        else:
            raise ValueError("No JSON array found in response")
    except Exception as e:
        print("❌ API or JSON error:", e)
        return []

# === MAIN LOOP ===
for idx, row in tqdm(df.iterrows(), total=len(df)):
    content = row.get("content", "")
    if not isinstance(content, str) or len(content.strip()) < 300:
        continue

    source_link = row.get("link", "")
    if source_link in processed_links:
        continue  # Skip already processed

    query = row.get("query", "")
    company = query.split()[0] if query else "Unknown"
    role = " ".join(query.split()[1:]) if query else "Unknown"
    experience = infer_experience(query)  # 🔧

    prompt = build_prompt(content, company, role)
    extracted = extract_questions(prompt)

    for q in extracted:
        output_df.loc[len(output_df)] = {
            "company": company,
            "role": role,
            "experience_level": experience,  # 🔧
            "question": q.get("question", ""),
            "round": q.get("round", ""),
            "topic": q.get("topic", ""),
            "difficulty": q.get("difficulty", ""),
            "source_link": source_link
        }

    # ✅ Save after every row
    output_df.to_csv(OUTPUT_CSV, index=False)

print("✅ Done! Final results saved to:", OUTPUT_CSV)