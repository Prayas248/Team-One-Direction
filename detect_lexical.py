# src/detect_lexical.py
# Layer 1: TF-IDF Lexical Detection against ChromaDB corpus

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import chromadb

# ── Config ─────────────────────────────────────────────────────────────────────
CHROMA_PATH       = "data/chroma_index"
COLLECTION_NAME   = "corpus"
TFIDF_THRESHOLD   = 0.70   # minimum cosine similarity to flag
MAX_CORPUS_DOCS   = 2000   # cap to avoid memory issues on large corpora


def _load_corpus_from_chroma() -> tuple:
    """
    Fetch raw document texts and metadata from ChromaDB.

    ChromaDB stores the original 200-word text chunks as 'documents'
    alongside their SBERT embeddings. L1 only needs the raw text —
    embeddings are irrelevant for TF-IDF.

    Returns:
        (docs, metadatas) — two parallel lists
        docs:      List[str]  — raw text chunks from corpus
        metadatas: List[dict] — title, doi, domain, year per chunk
    """
    try:
        client     = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = client.get_collection(COLLECTION_NAME)

        total = collection.count()
        print(f"  ChromaDB: {total} chunks in corpus")

        if total == 0:
            print("  WARNING: Corpus is empty — run scripts/build_corpus.py first")
            return [], []

        # Fetch all docs (cap at MAX_CORPUS_DOCS to avoid OOM)
        fetch_limit = min(total, MAX_CORPUS_DOCS)
        result = collection.get(
            limit=fetch_limit,
            include=["documents", "metadatas"]
        )

        docs      = result.get("documents", [])
        metadatas = result.get("metadatas", [])

        print(f"  Loaded {len(docs)} corpus chunks for TF-IDF")
        return docs, metadatas

    except Exception as e:
        print(f"  ERROR loading ChromaDB corpus: {e}")
        print("  → Make sure data/chroma_index/ exists and build_corpus.py has been run")
        return [], []


def detect_lexical(chunks: list, threshold: float = TFIDF_THRESHOLD) -> list:
    """
    TF-IDF cosine similarity of submitted chunks against corpus.

    Fetches raw document text from ChromaDB (the same chunks stored
    by build_corpus.py), builds a TF-IDF matrix across corpus + input,
    and flags any submitted chunk whose best cosine similarity to any
    corpus chunk meets the threshold.

    Args:
        chunks:    List of 200-word text chunks from the submitted paper
                   (output of ingest.chunk_text)
        threshold: Minimum cosine similarity to raise a flag (default 0.70)

    Returns:
        List of flag dicts compatible with score.aggregate_flags()
    """
    print("\n=== DETECT_LEXICAL (L1) CALLED ===")
    print(f"  Input chunks: {len(chunks)}")
    print(f"  Threshold: {threshold}")

    if not chunks:
        print("  No chunks to analyse — returning []")
        return []

    # ── Step 1: Load corpus from ChromaDB ─────────────────────────
    corpus_docs, corpus_meta = _load_corpus_from_chroma()

    if not corpus_docs:
        print("  No corpus docs available — L1 cannot run")
        return []

    # ── Step 2: Build TF-IDF matrix ───────────────────────────────
    # Corpus docs first, then submitted chunks
    # This ensures indices 0..n_corp-1 = corpus, n_corp.. = input
    all_texts = corpus_docs + chunks

    print(f"  Building TF-IDF matrix ({len(corpus_docs)} corpus + {len(chunks)} input)...")

    try:
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),   # unigrams + bigrams (spec requirement)
            max_features=50000,   # cap vocabulary size for memory
            sublinear_tf=True     # log normalization
        )
        tfidf_matrix = vectorizer.fit_transform(all_texts)
    except Exception as e:
        print(f"  ERROR building TF-IDF matrix: {e}")
        return []

    n_corp = len(corpus_docs)

    # ── Step 3: Compare each chunk against corpus ─────────────────
    flags = []

    for i, chunk in enumerate(chunks):
        chunk_vec = tfidf_matrix[n_corp + i]
        corpus_vecs = tfidf_matrix[:n_corp]

        sims = cosine_similarity(chunk_vec, corpus_vecs)[0]
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        if best_sim >= threshold:
            meta = corpus_meta[best_idx] if best_idx < len(corpus_meta) else {}
            print(f"  FLAG chunk {i}: sim={best_sim:.3f} | source='{meta.get('title', 'Unknown')[:50]}'")

            flags.append({
                # ── Fields required by score.aggregate_flags() ──
                'chunk':   chunk,
                'score':   best_sim,
                'matched': corpus_docs[best_idx],
                'meta':    meta,
                'layer':   1,
                'type':    'Verbatim/Near-Exact',
                'method':  'TF-IDF + Cosine',
            })

    print(f"  L1 complete — {len(flags)} flag(s) raised")
    print("===================================\n")
    return flags


# ── CLI test ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing detect_lexical against real ChromaDB corpus")
    print("=" * 60)

    # Use a real chunk that should exist in the corpus
    test_chunks = [
        "Natural language processing is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language.",
        "A completely unrelated sentence about cooking pasta at home for dinner tonight.",
    ]

    flags = detect_lexical(test_chunks, threshold=0.70)

    print("\nFlags Detected:")
    if flags:
        for flag in flags:
            print(f"  Score: {flag['score']:.3f}")
            print(f"  Type:  {flag['type']}")
            print(f"  Meta:  {flag['meta']}")
            print(f"  Chunk: {flag['chunk'][:100]}...")
            print()
    else:
        print("  No flags — either corpus is empty or no matches above threshold")
        print("  Run: python3 scripts/build_corpus.py  to populate the corpus first")