# scripts/build_corpus.py

import os
import json
import requests
import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import time

# Load environment variables
load_dotenv()

# Get API key
CORE_API_KEY = os.getenv("CORE_API_KEY")

if not CORE_API_KEY:
    raise ValueError("CORE_API_KEY not found in environment variables")

# ── Config ─────────────────────────────────────────────
CORE_BASE = "https://api.core.ac.uk/v3"
HEADERS = {"Authorization": f"Bearer {CORE_API_KEY}"}

CHUNK_SIZE = 200   # words per chunk
OVERLAP = 50       # words of overlap

# Queries with limits
QUERIES = [
    ("machine learning", 50),
    ("natural language processing", 50),
    ("climate change", 50),
    ("deep learning", 50),
    ("neural networks", 50),
    ("computer vision", 50),
    ("data science", 50),
    ("artificial intelligence", 50),
    ("renewable energy", 50),
    ("gene expression", 50),
]

MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# ── Helpers ────────────────────────────────────────────

def fetch_papers(query: str, limit: int, max_retries: int = 3) -> list:
    """
    Fetch papers from CORE API v3.
    
    CORE API v3 uses POST requests with JSON body (not GET with params).
    """
    
    # CORE API v3 search endpoint
    url = f"{CORE_BASE}/search/works"
    
    # Request body (JSON format)
    payload = {
        "q": query,              # Simple query string (no "fullText:" prefix)
        "limit": limit,
        "offset": 0,
        "stats": False,          # Don't need statistics
        "scroll": False          # Don't need scrolling
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                url,
                json=payload,      # ✅ POST with JSON body
                headers=HEADERS,
                timeout=30
            )
            
            # Check for rate limiting
            if response.status_code == 429:
                wait_time = int(response.headers.get("Retry-After", 60))
                print(f"  ⚠️  Rate limited. Waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            
            response.raise_for_status()
            
            data = response.json()
            
            # CORE API v3 response structure:
            # {
            #   "totalHits": 1234,
            #   "results": [...]
            # }
            
            results = data.get("results", [])
            
            # Filter for papers with full text
            papers_with_text = []
            for paper in results:
                # Check if paper has full text
                full_text = paper.get("fullText", "")
                if full_text and len(full_text.split()) >= 100:
                    papers_with_text.append(paper)
            
            print(f"  '{query}': {len(papers_with_text)}/{len(results)} papers with full text")
            return papers_with_text
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 500:
                print(f"  ⚠️  Server error (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    print(f"  ❌ Failed after {max_retries} attempts: {e}")
                    return []
            else:
                print(f"  ❌ HTTP error: {e}")
                return []
                
        except requests.exceptions.RequestException as e:
            print(f"  ❌ Request failed: {e}")
            return []
    
    return []


def chunk_text(text: str) -> list[str]:
    """Sliding window chunker — 200-word chunks, 50-word overlap."""
    words = text.split()
    chunks = []
    step = CHUNK_SIZE - OVERLAP
    
    for i in range(0, max(1, len(words) - CHUNK_SIZE + 1), step):
        chunk = " ".join(words[i : i + CHUNK_SIZE])
        if len(chunk.split()) >= 50:  # Skip tiny trailing chunks
            chunks.append(chunk)
    
    # If text is shorter than CHUNK_SIZE, return it as single chunk
    if not chunks and len(words) >= 50:
        chunks.append(" ".join(words))
    
    return chunks


# ── Main Build ─────────────────────────────────────────

def build_corpus():
    """Build ChromaDB corpus from CORE API papers."""
    
    print("\n" + "="*60)
    print("Building Corpus from CORE API")
    print("="*60 + "\n")
    
    # Initialize ChromaDB
    client = chromadb.PersistentClient(path="data/chroma_index")
    
    # Delete existing collection if rebuilding
    try:
        client.delete_collection("corpus")
        print("  Deleted existing corpus collection")
    except Exception:
        pass
    
    # Create new collection with cosine similarity
    collection = client.create_collection(
        name="corpus",
        metadata={"hnsw:space": "cosine"}  # ✅ Use cosine distance
    )
    
    total_chunks = 0
    total_papers = 0
    failed_queries = []

    # Fetch papers for each query
    for query, limit in QUERIES:
        print(f"\nFetching: {query}")
        papers = fetch_papers(query, limit)
        
        if not papers:
            failed_queries.append(query)
            continue

        for paper in papers:
            try:
                # Extract full text
                full_text = paper.get("fullText", "")
                if not full_text or len(full_text.split()) < 100:
                    continue
                
                # Chunk the text
                chunks = chunk_text(full_text)
                if not chunks:
                    continue

                # Generate embeddings (batch processing)
                embeddings = MODEL.encode(
                    chunks,
                    batch_size=32,
                    show_progress_bar=False,
                    normalize_embeddings=True  # ✅ Normalize for cosine similarity
                ).tolist()

                # Prepare metadata
                paper_id = str(paper.get("id", f"unknown_{total_papers}"))
                title = paper.get("title", "Unknown")
                doi = paper.get("doi", "")
                year = str(paper.get("yearPublished", ""))
                
                # Add to ChromaDB
                collection.add(
                    ids=[f"{paper_id}_chunk_{i}" for i in range(len(chunks))],
                    embeddings=embeddings,
                    documents=chunks,
                    metadatas=[{
                        "title": str(title)[:200],  # Truncate long titles
                        "doi": str(doi),
                        "domain": str(query),
                        "year": str(year),
                        "paper_id": str(paper_id)
                    } for _ in chunks]
                )

                total_chunks += len(chunks)
                total_papers += 1
                
                # Truncate title for display
                display_title = title[:60] + "..." if len(title) > 60 else title
                print(f"  ✓ {display_title} — {len(chunks)} chunks")
                
            except Exception as e:
                print(f"  ⚠️  Error processing paper: {e}")
                continue

    # Summary
    print("\n" + "="*60)
    print(f"Corpus Build Complete")
    print("="*60)
    print(f"  Papers processed: {total_papers}")
    print(f"  Total chunks: {total_chunks}")
    print(f"  Storage path: data/chroma_index/")
    
    if failed_queries:
        print(f"\n⚠️  Failed queries ({len(failed_queries)}):")
        for q in failed_queries:
            print(f"    - {q}")
    
    if total_chunks < 500:
        print(f"\n⚠️  WARNING: Only {total_chunks} chunks (spec requires ≥500)")
        print("   Recommendations:")
        print("   1. Increase limit values in QUERIES")
        print("   2. Add more query topics")
        print("   3. Check CORE API status")
    else:
        print(f"\n✅ Corpus meets ≥500 chunk requirement")
    
    print()


if __name__ == "__main__":
    build_corpus()