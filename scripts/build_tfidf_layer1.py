"""
Build TF-IDF artifacts for Layer 1 (separate from ChromaDB corpus)

This script:
1. Reads corpus from existing ChromaDB
2. Pre-fits TF-IDF vectorizer on corpus only (for reproducibility)
3. Saves artifacts to data/tfidf_corpus/ (for Layer 1 to load)

Usage:
  python scripts/build_corpus.py      # Build ChromaDB (Layer 2/3 use this)
  python scripts/build_tfidf_layer1.py # Build TF-IDF (Layer 1 uses this)

Architecture:
  ✅ Separation of concerns: corpus building vs. Layer 1 preparation
  ✅ Reusable: Layer 2/3 don't depend on TF-IDF code
  ✅ Optional: Can skip if only using Layer 2/3
"""

import chromadb
import pickle
import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp


def build_tfidf_layer1():
    """Build pre-fitted TF-IDF artifacts from existing ChromaDB corpus."""
    
    print("\n" + "="*80)
    print("BUILDING TF-IDF ARTIFACTS FOR LAYER 1")
    print("="*80)
    
    # ──────────────────────────────────────────────────────────────
    # Step 1: Load corpus from ChromaDB
    # ──────────────────────────────────────────────────────────────
    print("\nStep 1: Loading corpus from ChromaDB...")
    
    try:
        client = chromadb.PersistentClient(path="data/chroma_index")
        collection = client.get_collection("corpus")
    except Exception as e:
        print(f"❌ Failed to load ChromaDB corpus")
        print(f"   Error: {e}")
        print(f"   Did you run: python scripts/build_corpus.py?")
        return False
    
    # Retrieve all corpus documents and metadata
    try:
        corpus_data = collection.get(include=["documents", "metadatas"])
        corpus_chunks = corpus_data["documents"]
        corpus_metadata = corpus_data["metadatas"]
    except Exception as e:
        print(f"❌ Failed to retrieve corpus data: {e}")
        return False
    
    if not corpus_chunks:
        print(f"❌ Corpus is empty. Run: python scripts/build_corpus.py")
        return False
    
    print(f"  ✓ Loaded {len(corpus_chunks)} chunks from ChromaDB")
    print(f"  ✓ Total size: {sum(len(c.split()) for c in corpus_chunks):,} words")
    
    # ──────────────────────────────────────────────────────────────
    # Step 2: Pre-fit TF-IDF vectorizer on corpus only
    # ──────────────────────────────────────────────────────────────
    print("\nStep 2: Pre-fitting TF-IDF vectorizer...")
    
    try:
        # Create tfidf_corpus directory
        tfidf_dir = Path("data/tfidf_corpus")
        tfidf_dir.mkdir(parents=True, exist_ok=True)
        
        # Fit TF-IDF with optimized parameters
        # max_features=10000: Cap vocabulary to top 10k terms (prevents bloat)
        # min_df=2: Ignore terms in only 1 document (noise reduction)
        # max_df=0.95: Ignore very common terms (stop words)
        # ngram_range=(1, 2): Use unigrams + bigrams
        # sublinear_tf=True: Sublinear TF scaling (better for large docs)
        # stop_words='english': Remove English stop words
        
        print(f"  Pre-fitting on {len(corpus_chunks)} corpus chunks...")
        tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            min_df=2,
            max_df=0.95,
            ngram_range=(1, 2),
            lowercase=True,
            stop_words='english',
            sublinear_tf=True
        )
        
        # CRITICAL: Fit ONLY on corpus (for reproducibility FR-7)
        # If we fit on corpus + manuscript, vocabulary changes per manuscript
        tfidf_vectorizer.fit(corpus_chunks)
        print(f"  ✓ Vectorizer fitted on corpus")
        
        # ──────────────────────────────────────────────────────────────
        # Step 3: Transform corpus in fitted vector space
        # ──────────────────────────────────────────────────────────────
        print(f"  Computing TF-IDF vectors for all corpus chunks...")
        corpus_vectors = tfidf_vectorizer.transform(corpus_chunks)
        print(f"  ✓ Transformed to sparse matrix: {corpus_vectors.shape}")
        
        # ──────────────────────────────────────────────────────────────
        # Step 4: Save all artifacts
        # ──────────────────────────────────────────────────────────────
        print(f"\nStep 3: Saving artifacts to {tfidf_dir}/...")
        
        # 1. Pre-fitted vectorizer (sklearn TfidfVectorizer object)
        with open(tfidf_dir / "vectorizer.pkl", "wb") as f:
            pickle.dump(tfidf_vectorizer, f)
        print(f"  ✓ vectorizer.pkl ({Path(tfidf_dir / 'vectorizer.pkl').stat().st_size / 1024 / 1024:.2f} MB)")
        
        # 2. Pre-computed corpus vectors (sparse matrix)
        sp.save_npz(tfidf_dir / "corpus_vectors.npz", corpus_vectors)
        print(f"  ✓ corpus_vectors.npz ({Path(tfidf_dir / 'corpus_vectors.npz').stat().st_size / 1024 / 1024:.2f} MB)")
        
        # 3. Corpus documents (for comparison in detect_lexical)
        with open(tfidf_dir / "corpus_chunks.pkl", "wb") as f:
            pickle.dump(corpus_chunks, f)
        print(f"  ✓ corpus_chunks.pkl ({Path(tfidf_dir / 'corpus_chunks.pkl').stat().st_size / 1024 / 1024:.2f} MB)")
        
        # 4. Corpus metadata (for flag results)
        with open(tfidf_dir / "corpus_metadata.json", "w") as f:
            json.dump(corpus_metadata, f, indent=2)
        print(f"  ✓ corpus_metadata.json ({Path(tfidf_dir / 'corpus_metadata.json').stat().st_size / 1024 / 1024:.2f} MB)")
        
        # ──────────────────────────────────────────────────────────────
        # Step 5: Summary
        # ──────────────────────────────────────────────────────────────
        vocab_size = len(tfidf_vectorizer.get_feature_names_out())
        
        print("\n" + "="*80)
        print("✅ TF-IDF PRE-FITTING COMPLETE")
        print("="*80)
        print(f"\nVocabulary Statistics:")
        print(f"  Vocabulary size: {vocab_size:,} terms")
        print(f"  Corpus vectors: {corpus_vectors.shape}")
        print(f"  Sparsity: {(1 - corpus_vectors.nnz / (corpus_vectors.shape[0] * corpus_vectors.shape[1])) * 100:.1f}%")
        
        print(f"\nArtifacts saved to: {tfidf_dir}/")
        print(f"  • vectorizer.pkl         ← Pre-fitted model (reproducible)")
        print(f"  • corpus_vectors.npz     ← Pre-computed vectors (fast)")
        print(f"  • corpus_chunks.pkl      ← Corpus documents")
        print(f"  • corpus_metadata.json   ← Titles, DOIs, domains, years")
        
        print(f"\nFR-7 Compliance:")
        print(f"  ✅ Fixed vocabulary (fitted on corpus only)")
        print(f"  ✅ Reproducible: Same input → Same output")
        print(f"  ✅ Ready for Layer 1 detection")
        
        print(f"\nNext step:")
        print(f"  python backend/detect_lexical.py      # Test Layer 1")
        print(f"  python scripts/test_papers_layer1.py  # Test on papers")
        
        return True
        
    except Exception as e:
        print(f"❌ TF-IDF pre-fitting failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    success = build_tfidf_layer1()
    sys.exit(0 if success else 1)
