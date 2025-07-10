import pandas as pd
import trafilatura
from newspaper import Article
from tqdm import tqdm
import os

INPUT_FILE = "serpapi_results_1.csv"
OUTPUT_FILE = "scraped_results.csv"
SAVE_INTERVAL = 20  # Save progress every n rows

def fetch_content(url):
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(downloaded)
            if text and len(text.strip()) > 200:
                return text.strip()
    except:
        pass

    # Fallback to newspaper3k
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text.strip()
    except:
        return None

def main():
    df = pd.read_csv(INPUT_FILE)

    if 'content' not in df.columns:
        df['content'] = None

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        if pd.notna(df.at[idx, 'content']):
            continue  # Already scraped

        url = row['link']
        content = fetch_content(url)
        df.at[idx, 'content'] = content

        # Periodic saving
        if idx % SAVE_INTERVAL == 0:
            df.to_csv(OUTPUT_FILE, index=False)

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Finished. Scraped content saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
