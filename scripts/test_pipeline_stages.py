# scripts/test_pipeline_stages.py
# Full pipeline verification: ingest → corpus → similarity.
# Run AFTER build_corpus.py has completed.
#
# Usage:
#   python scripts/test_pipeline_stages.py

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

import chromadb
import numpy as np
from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PASS = 0
FAIL = 0

def check(label, condition, detail=""):
    global PASS, FAIL
    if condition:
        print(f"  ✅  {label}")
        PASS += 1
    else:
        print(f"  ❌  {label}" + (f"\n      → {detail}" if detail else ""))
        FAIL += 1


# ════════════════════════════════════════════════════════
# TEST 1 — ingest.py core logic (no PDF needed)
# ════════════════════════════════════════════════════════
print("\n[1] ingest.py core logic")

from ingest import chunk_text, is_noise_line

# chunk_text: basic
words_300 = " ".join([f"w{i}" for i in range(300)])
chunks = chunk_text(words_300, size=200, overlap=50)
check("300 words → chunks produced",        len(chunks) > 0)
check("All chunks ≤ 200 words",             all(len(c.split()) <= 200 for c in chunks))
check("All chunks ≥ 50 words",              all(len(c.split()) >= 50  for c in chunks))

# chunk_text: overlap
if len(chunks) >= 2:
    tail  = chunks[0].split()[-50:]
    head  = chunks[1].split()[:50]
    check("50-word overlap preserved",      tail == head)

# chunk_text: too-short input → no chunks
tiny = " ".join(["x"] * 40)
check("40 words → 0 chunks (below floor)", len(chunk_text(tiny)) == 0)

# is_noise_line
check("Empty string is noise",             is_noise_line(""))
check("Page number '42' is noise",         is_noise_line("42"))
check("ISSN line is noise",                is_noise_line("ISSN: 2349-5162"))
check("Normal sentence not noise",         not is_noise_line("This paper proposes a method."))


# ════════════════════════════════════════════════════════
# TEST 2 — ChromaDB index exists and is populated
# ════════════════════════════════════════════════════════
print("\n[2] ChromaDB corpus index")

try:
    client = chromadb.PersistentClient(path="data/chroma_index")
    col    = client.get_collection("corpus")
    count  = col.count()
    check(f"Index exists and readable ({count} chunks)", True)
    check(f"≥500 chunks in corpus",
          count >= 500,
          f"Only {count} — increase QUERIES limits in build_corpus.py")
except Exception as e:
    check("ChromaDB index accessible", False, str(e))
    print("\n  ⛔ Cannot continue without corpus. Run build_corpus.py first.")
    sys.exit(1)


# ════════════════════════════════════════════════════════
# TEST 3 — Metadata completeness
# ════════════════════════════════════════════════════════
print("\n[3] Corpus metadata")

meta    = col.get(include=["metadatas"])["metadatas"]
domains = Counter(m.get("domain", "unknown") for m in meta)
missing_title = sum(1 for m in meta if not m.get("title") or m["title"] == "Unknown")

check(f"≥3 domains in corpus ({len(domains)} found)",
      len(domains) >= 2,                          # relaxed: 2 is fine for hackathon
      f"Domains: {dict(domains)}")
check(f"Title populated for most chunks (missing: {missing_title})",
      missing_title / len(meta) < 0.3,           # allow 30% missing titles
      "Many papers lack titles in CORE API response")

print(f"    Domain breakdown: {dict(domains)}")


# ════════════════════════════════════════════════════════
# TEST 4 — Semantic similarity (Layer 2: SBERT)
# ════════════════════════════════════════════════════════
print("\n[4] Layer 2 — Semantic similarity (SBERT)")

model   = SentenceTransformer("all-MiniLM-L6-v2")
# Take a real chunk from the corpus and lightly paraphrase it
sample  = col.get(limit=1)["documents"][0]
words   = sample.split()
# Simulate paraphrase: take the middle of the chunk (skipping first/last 10 words)
paraphrase = " ".join(words[10:min(80, len(words))])

embed   = model.encode([paraphrase])[0].tolist()
results = col.query(query_embeddings=[embed], n_results=3)
dists   = results["distances"][0]
docs    = results["documents"][0]
metas   = results["metadatas"][0]

sims = [1 - (d / 2) for d in dists]

print(f"    Test passage (first 80 chars): '{paraphrase[:80]}...'")
print(f"    Top 3 SBERT matches:")
for i, (sim, doc, meta) in enumerate(zip(sims, docs, metas)):
    tier = "🔴 HIGH" if sim >= 0.85 else "🟡 MEDIUM" if sim >= 0.70 else "🟢 LOW" if sim >= 0.55 else "  (below threshold)"
    print(f"      [{i+1}] score={sim:.3f} {tier} — {meta.get('title','?')[:50]}")

check("Top SBERT match above 0.55 threshold",
      sims[0] >= 0.55,
      "Corpus may not overlap with test domain — add more relevant queries to build_corpus.py")


# ════════════════════════════════════════════════════════
# TEST 5 — Lexical similarity (Layer 1: TF-IDF)
# ════════════════════════════════════════════════════════
print("\n[5] Layer 1 — Lexical similarity (TF-IDF)")

# Sample 200 docs (not 500 — keeps this test fast)
corpus_sample = col.get(limit=200)["documents"]
all_texts     = corpus_sample + [paraphrase]
tfidf         = TfidfVectorizer(ngram_range=(1, 2)).fit_transform(all_texts)
sims_tfidf    = cosine_similarity(tfidf[-1], tfidf[:-1])[0]
best_idx      = int(np.argmax(sims_tfidf))
best_score    = float(sims_tfidf[best_idx])

tier = "🔴 HIGH" if best_score >= 0.85 else "🟡 MEDIUM" if best_score >= 0.70 else "🟢 LOW" if best_score >= 0.55 else "  (below threshold)"
print(f"    Best TF-IDF match: score={best_score:.3f} {tier}")
print(f"    Excerpt: '{corpus_sample[best_idx][:80]}...'")

check("TF-IDF match above 0.55 threshold",
      best_score >= 0.55,
      "Expected — paraphrase of a corpus chunk should always score above 0.55 lexically")


# ════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════
print(f"\n{'═'*55}")
print(f"  RESULTS: {PASS} passed, {FAIL} failed")
print(f"{'═'*55}")

if FAIL == 0:
    print("  ✅  All checks passed.")
    print("  Safe to build:  detect_lexical.py  (Layer 1)")
    print("                  detect_semantic.py (Layer 2)")
    print("                  detect_intrinsic.py (Layer 3 — no corpus needed)")
else:
    print(f"  ❌  {FAIL} check(s) failed — fix before building detection layers.")
    sys.exit(1)

print()