import requests
import json
import time
import os
import pandas as pd
import logging
from newspaper import Article
from sentence_transformers import SentenceTransformer, util
import spacy
import backoff
import openai

# ------------------------------
# CONFIGURATION
# ------------------------------

NEWSAPI_KEY = "d209e9bd7f5648f1831bdaa433c85d05"
openai.api_key = ""   # leave empty if you don't have OpenAI key


QUERY = "AI startups"
PAGE_SIZE = 20
MAX_PAGES = 2

EMBED_MODEL = "all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.80
OUTPUT_CSV = "autoanalytica_output.csv"

# ------------------------------
# LOAD MODELS
# ------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

embedder = SentenceTransformer(EMBED_MODEL)
nlp = spacy.load("en_core_web_sm")

# ------------------------------
# STEP 1: FETCH NEWS
# ------------------------------

@backoff.on_exception(backoff.expo, Exception, max_tries=3)
def fetch_page(page):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": QUERY,
        "pageSize": PAGE_SIZE,
        "page": page,
        "language": "en",
        "apiKey": NEWSAPI_KEY
    }
    res = requests.get(url, params=params, timeout=20)
    res.raise_for_status()
    return res.json()

def fetch_articles():
    all_articles = []
    for p in range(1, MAX_PAGES + 1):
        logging.info(f"Fetching page {p}...")
        data = fetch_page(p)
        all_articles.extend(data.get("articles", []))
    logging.info(f"Fetched {len(all_articles)} raw articles.")
    return all_articles

# ------------------------------
# STEP 2: CLEAN + PARSE ARTICLES
# ------------------------------

def extract_text(url, fallback):
    try:
        a = Article(url)
        a.download()
        a.parse()
        return a.text
    except:
        return fallback or ""

def normalize_articles(articles):
    cleaned = []
    for a in articles:
        text = extract_text(a.get("url"), a.get("description"))
        cleaned.append({
            "title": a.get("title", ""),
            "url": a.get("url"),
            "source": a.get("source", {}).get("name"),
            "publishedAt": a.get("publishedAt"),
            "text": text
        })
    logging.info("Normalized article content.")
    return cleaned

# ------------------------------
# STEP 3: DEDUPLICATION
# ------------------------------

def dedupe_articles(cleaned):
    contents = [c["title"] + "\n" + c["text"][:500] for c in cleaned]
    embeddings = embedder.encode(contents)

    used = set()
    final = []

    for i in range(len(cleaned)):
        if i in used:
            continue
        group = [i]
        for j in range(i + 1, len(cleaned)):
            if j not in used:
                sim = util.cos_sim(embeddings[i], embeddings[j]).item()
                if sim >= SIMILARITY_THRESHOLD:
                    group.append(j)
                    used.add(j)
        used.add(i)
        final.append(cleaned[i])
    logging.info(f"Deduped from {len(cleaned)} → {len(final)}")
    return final

# ------------------------------
# STEP 4: HYPE FILTER
# ------------------------------

def info_density(text):
    doc = nlp(text[:2000])
    ents = len(doc.ents)
    tokens = len([t for t in doc if not t.is_punct])
    if tokens == 0:
        return 0
    return ents / tokens

def filter_hype(articles):
    output = []
    for a in articles:
        density = info_density(a["text"])
        logging.info(f"Hype Check: {a['title'][:50]}... | Density = {density:.3f}")
        if density > 0.01:
            output.append(a)
    logging.info(f"Filtered → {len(output)} articles remain.")
    return output

# ------------------------------
# STEP 5: LLM JSON EXTRACTION
# ------------------------------

def extract_json(article):
    """
    Offline JSON extractor without OpenAI.
    Basic rule-based extraction.
    """
    title = article["title"]
    text = article["text"]

    # Extract company name (first org entity)
    doc = nlp(text[:400])
    companies = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
    company = companies[0] if companies else "Unknown"

    # Simple category detection
    if "funding" in text.lower() or "raise" in text.lower():
        category = "Funding"
        is_funding = True
    elif "launch" in text.lower() or "release" in text.lower():
        category = "Product Launch"
        is_funding = False
    else:
        category = "General AI News"
        is_funding = False

    # Sentiment (basic rule)
    sentiment = 0
    if "success" in text.lower() or "growth" in text.lower():
        sentiment = 0.5
    elif "problem" in text.lower() or "concern" in text.lower():
        sentiment = -0.5

    # Summary (first 2 sentences)
    summary = ". ".join(text.split(".")[:2]) + "."

    return {
        "company_name": company,
        "category": category,
        "sentiment_score": sentiment,
        "is_funding_news": is_funding,
        "summary": summary,
    }


# ------------------------------
# STEP 6: EXPORT RESULTS
# ------------------------------

def save_to_csv(rows):
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    logging.info(f"Saved {len(rows)} records → {OUTPUT_CSV}")

# ------------------------------
# MAIN PIPELINE
# ------------------------------

def run():
    t0 = time.time()

    articles = fetch_articles()
    cleaned = normalize_articles(articles)
    deduped = dedupe_articles(cleaned)
    filtered = filter_hype(deduped)

    results = []
    for a in filtered:
        out = extract_json(a)
        if out:
            out["url"] = a["url"]
            out["source"] = a["source"]
            out["publishedAt"] = a["publishedAt"]
            results.append(out)

    save_to_csv(results)

    logging.info(f"Pipeline finished in {time.time() - t0:.2f}s")

if __name__ == "__main__":
    run()
