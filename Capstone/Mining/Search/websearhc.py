import requests
import pandas as pd
import os

API_KEY = "66698cc19837b89646caaa7fecde3455c76124aba7ddb53a702a9eaf392578453a111" #CHANGED. WILL USE ENV LATER
INPUT_FILE = "Mining\Search\remaining_queries.txt"
OUTPUT_FILE = "serpapi_results.csv"
TEMP_FILE = "serpapi_results_temp.csv"
SAVE_EVERY = 5
DELAY_RANGE = 0  # seconds

def load_queries(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def load_cached_queries(file_path):
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        return set(df['query'].unique())
    return set()

def search_serpapi(query):
    url = "https://serpapi.com/search"
    params = {
        "engine": "google",
        "q": query,
        "api_key": API_KEY
    }
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        results = data.get("organic_results", [])
        return [
            {
                "query": query,
                "title": r.get("title"),
                "link": r.get("link"),
                "snippet": r.get("snippet", "")
            }
            for r in results
        ]
    except Exception as e:
        print(f"❌ Error for '{query}': {e}")
        return []

def main():
    queries = load_queries(INPUT_FILE)
    cached = load_cached_queries(TEMP_FILE)
    print(f"🔍 {len(queries)} queries loaded. {len(cached)} already done.")

    all_results = []

    for i, query in enumerate(queries):
        if query in cached:
            print(f"[{i+1}] ✅ Skipping (cached): {query}")
            continue

        print(f"[{i+1}] 🔍 Searching: {query}")
        results = search_serpapi(query)
        all_results.extend(results)

        # Save progress periodically
        if (i + 1) % SAVE_EVERY == 0 or i == len(queries) - 1:
            pd.DataFrame(all_results).to_csv(TEMP_FILE, index=False)
            print(f"💾 Progress saved to {TEMP_FILE}")

        pass

    pd.DataFrame(all_results).to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Done! Final results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
