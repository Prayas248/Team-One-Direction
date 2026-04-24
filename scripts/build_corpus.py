# scripts/build_corpus.py

import os
import requests
import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

# Get API key
CORE_API_KEY = os.environ["CORE_API_KEY"]

# ── Config ─────────────────────────────────────────────
CORE_BASE    = "https://api.core.ac.uk/v3"
HEADERS      = {"Authorization": f"Bearer {CORE_API_KEY}"}

CHUNK_SIZE   = 200   # words per chunk (spec requirement)
OVERLAP      = 50    # words of overlap (spec requirement)

QUERIES = [
    ("natural language processing", 40),
    ("climate change adaptation",   30),
    ("gene expression cancer",       30),
]

MODEL = SentenceTransformer("all-MiniLM-L6-v2")  # 80MB, runs locally

# ── Helpers ────────────────────────────────────────────

def fetch_papers(query: str, limit: int) -> list:
    """Fetch full-text papers from CORE API for a given query."""
    response = requests.post(
        f"{CORE_BASE}/search/works",
        json={"q": query, "limit": limit, "fullText": True},
        headers=HEADERS,
        timeout=30
    )
    response.raise_for_status()
    results = response.json().get("results", [])
    
    # Filter: only keep papers that actually have full text
    papers_with_text = [p for p in results if p.get("fullText")]
    print(f"  '{query}': {len(results)} returned, {len(papers_with_text)} with fullText")
    return papers_with_text


def chunk_text(text: str) -> list[str]:
    """Sliding window chunker — 200-word chunks, 50-word overlap."""
    words = text.split()
    chunks = []
    step = CHUNK_SIZE - OVERLAP
    for i in range(0, max(1, len(words) - CHUNK_SIZE + 1), step):
        chunk = " ".join(words[i : i + CHUNK_SIZE])
        if len(chunk.split()) >= 50:  # skip tiny trailing chunks
            chunks.append(chunk)
    return chunks


# ── Main Build ─────────────────────────────────────────

def build_corpus():
    client = chromadb.PersistentClient(path="data/chroma_index")
    
    # Delete existing collection if rebuilding
    try:
        client.delete_collection("corpus")
    except Exception:
        pass
    collection = client.create_collection("corpus")

    total_chunks = 0
    total_papers = 0

    for query, limit in QUERIES:
        print(f"\nFetching: {query}")
        papers = fetch_papers(query, limit)

        for paper in papers:
            full_text = paper.get("fullText", "")
            if not full_text or len(full_text.split()) < 100:
                continue  # skip near-empty papers

            chunks = chunk_text(full_text)
            if not chunks:
                continue

            # Embed in batches of 32 (memory-safe)
            embeddings = MODEL.encode(chunks, batch_size=32).tolist()

            paper_id = str(paper.get("id", "unknown"))
            collection.add(
                ids=[f"{paper_id}_{i}" for i in range(len(chunks))],
                embeddings=embeddings,
                documents=chunks,
                metadatas=[{
    "title":  str(paper.get("title") or "Unknown"),
    "doi":    str(paper.get("doi") or ""),
    "domain": str(query),
    "year":   str(paper.get("publishedDate") or "")[:4],
} for _ in chunks]
            )

            total_chunks += len(chunks)
            total_papers += 1
            print(f"  ✓ {paper.get('title', 'Untitled')[:60]} — {len(chunks)} chunks")

    print(f"\n{'='*50}")
    print(f"Corpus built: {total_papers} papers, {total_chunks} chunks")
    print(f"Stored at: data/chroma_index/")

    if total_chunks < 500:
        print(f"⚠️  WARNING: Only {total_chunks} chunks — spec requires ≥500")
        print("   → Increase the limit values in QUERIES above")
    else:
        print(f"✅ Corpus meets the ≥500 chunk requirement")


if __name__ == "__main__":
    build_corpus()