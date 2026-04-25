"""
Layer 1: TF-IDF Lexical Similarity Detection

Pre-fitted vectorizer for fast, reproducible plagiarism detection.
Vectorizer is pre-fitted on corpus at build time, then loaded at runtime.
"""

import chromadb
import pickle
import json
import numpy as np
from pathlib import Path
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp
from config.settings import settings

# Constants for chunking
CHUNK_SIZE = 200   # words per chunk
OVERLAP = 50       # words of overlap


# Risk tier thresholds (from spec FR-3)
THRESHOLDS = {
    'HIGH': 0.85,
    'MEDIUM': 0.70,
    'LOW': 0.55
}


def _get_risk_tier(score: float) -> str:
    """Classify score into risk tier."""
    if score >= THRESHOLDS['HIGH']:
        return 'HIGH'
    elif score >= THRESHOLDS['MEDIUM']:
        return 'MEDIUM'
    elif score >= THRESHOLDS['LOW']:
        return 'LOW'
    else:
        return None  # Below threshold


def _chunk_text(text: str) -> List[str]:
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


def _extract_text_from_file(file_path: str) -> str:
    """Extract text from PDF file."""
    from backend.ingest import extract_text
    
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if file_path.suffix.lower() != '.pdf':
        raise ValueError(f"Only PDF files supported, got: {file_path.suffix}")
    
    # extract_text returns dict of sections
    sections = extract_text(str(file_path))
    
    if not sections:
        raise RuntimeError(f"Failed to extract text from {file_path.name}")
    
    # Join all sections into full text
    full_text = " ".join(sections.values())
    
    return full_text


def detect_lexical(
    chunks: List[str],
    threshold: float = 0.55,
    corpus_limit: int = None,
    tfidf_path: str = None,
    chroma_path: str = None
) -> List[Dict]:
    """
    Layer 1: TF-IDF lexical similarity detection.
    
    Detects verbatim and near-exact matches by comparing manuscript chunks
    against a pre-fitted TF-IDF vectorizer and corpus.
    
    REPRODUCIBILITY (FR-7): Vectorizer is pre-fitted on corpus only at build time.
    This ensures identical inputs produce identical outputs across runs.
    
    PERFORMANCE (FR-6): Pre-fitted artifacts loaded from disk (~50-100ms per chunk).
    
    Args:
        chunks: List of manuscript text chunks to analyze
        threshold: Minimum similarity score (0.55 = LOW tier per spec)
        corpus_limit: Max corpus chunks to compare (None = all)
        tfidf_path: Path to pre-fitted TF-IDF artifacts (uses config if None)
        chroma_path: Path to ChromaDB (uses config if None)
    
    Returns:
        List of flagged passages with structure:
        {
            'chunk': str,              # Original manuscript chunk
            'chunk_index': int,        # Position in manuscript (for deduplication)
            'score': float,            # Cosine similarity (0.00-1.00)
            'matched': str,            # Matched corpus chunk
            'matched_index': int,      # Position in corpus
            'meta': {                  # Corpus metadata
                'title': str,
                'doi': str,
                'domain': str,
                'year': str,
                'chunk_index': int     # Position in source paper
            },
            'layer': 1,                # Detection layer
            'type': 'Verbatim/Near-Exact',
            'risk_tier': str           # HIGH / MEDIUM / LOW
        }
    
    Raises:
        FileNotFoundError: If pre-fitted artifacts not found
        RuntimeError: If corpus is empty
    """
    
    # Use config paths if not provided
    if tfidf_path is None:
        tfidf_path = settings.TFIDF_CORPUS_PATH
    if chroma_path is None:
        chroma_path = settings.CHROMA_INDEX_PATH
    
    # ──────────────────────────────────────────────────────────────
    # Step 1: Load pre-fitted TF-IDF vectorizer
    # ──────────────────────────────────────────────────────────────
    tfidf_dir = Path(tfidf_path)
    
    if not tfidf_dir.exists():
        raise FileNotFoundError(
            f"TF-IDF artifacts not found at '{tfidf_path}'. "
            f"Run: python scripts/build_corpus.py"
        )
    
    try:
        # Load pre-fitted vectorizer (fitted on corpus only for reproducibility)
        with open(tfidf_dir / "vectorizer.pkl", "rb") as f:
            tfidf_vectorizer = pickle.load(f)
        
        # Load pre-computed corpus vectors
        corpus_vectors = sp.load_npz(tfidf_dir / "corpus_vectors.npz")
        
        # Load corpus documents (for comparison)
        with open(tfidf_dir / "corpus_chunks.pkl", "rb") as f:
            corpus_chunks = pickle.load(f)
        
        # Load corpus metadata
        with open(tfidf_dir / "corpus_metadata.json", "r") as f:
            corpus_metadata = json.load(f)
        
    except Exception as e:
        raise FileNotFoundError(
            f"Failed to load TF-IDF artifacts: {e}\n"
            f"Run: python scripts/build_corpus.py"
        )
    
    if not corpus_chunks:
        raise RuntimeError("Corpus is empty. Run build_corpus.py first.")
    
    # Apply corpus limit if specified
    if corpus_limit is not None and corpus_limit < len(corpus_chunks):
        corpus_vectors = corpus_vectors[:corpus_limit]
        corpus_chunks = corpus_chunks[:corpus_limit]
        corpus_metadata = corpus_metadata[:corpus_limit]
    
    # ──────────────────────────────────────────────────────────────
    # Step 2: Transform manuscript chunks using pre-fitted vectorizer
    # ──────────────────────────────────────────────────────────────
    try:
        # Transform manuscript in same vector space as corpus
        # CRITICAL: Use same vectorizer that was fitted on corpus only
        manuscript_vectors = tfidf_vectorizer.transform(chunks)
    except Exception as e:
        raise RuntimeError(f"TF-IDF transformation failed: {e}")
    
    # ──────────────────────────────────────────────────────────────
    # Step 3: Compare each manuscript chunk against corpus
    # ──────────────────────────────────────────────────────────────
    flags = []
    
    for chunk_idx, (chunk, manuscript_vec) in enumerate(zip(chunks, manuscript_vectors)):
        # Compare this manuscript vector against all corpus vectors
        similarities = cosine_similarity(manuscript_vec, corpus_vectors)[0]
        
        # Find best match
        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])
        
        # ──────────────────────────────────────────────────────────
        # Step 4: Flag if above threshold and classify risk tier
        # ──────────────────────────────────────────────────────────
        if best_score >= threshold:
            risk_tier = _get_risk_tier(best_score)
            
            if risk_tier:  # Only flag if tier is not None
                flags.append({
                    'chunk': chunk,
                    'chunk_index': chunk_idx,               # Position in manuscript
                    'score': best_score,
                    'matched': corpus_chunks[best_idx],
                    'matched_index': best_idx,             # Position in corpus
                    'meta': corpus_metadata[best_idx],      # Includes chunk_index
                    'layer': 1,
                    'type': 'Verbatim/Near-Exact',
                    'risk_tier': risk_tier                  # HIGH/MEDIUM/LOW
                })
    
    return flags


def detect_lexical_file(
    file_path: str,
    threshold: float = 0.55,
    corpus_limit: int = None,
    tfidf_path: str = "data/tfidf_corpus",
    chroma_path: str = "data/chroma_index"
) -> Dict:
    """
    Layer 1 detection on entire PDF paper.
    
    Extracts text from PDF, chunks it, and runs lexical similarity detection
    on all chunks. Returns comprehensive results with summary statistics.
    
    Args:
        file_path: Path to PDF file
        threshold: Minimum similarity score (0.55 = LOW tier per spec)
        corpus_limit: Max corpus chunks to compare (None = all)
        tfidf_path: Path to pre-fitted TF-IDF artifacts
        chroma_path: Path to ChromaDB
    
    Returns:
        Dictionary with structure:
        {
            'file': str,              # Original file path
            'file_name': str,         # Just the filename
            'status': str,            # 'success' or 'error'
            'total_chunks': int,      # Total chunks extracted
            'flagged_count': int,     # Number of flagged chunks
            'high_risk': int,         # Count of HIGH tier flags
            'medium_risk': int,       # Count of MEDIUM tier flags
            'low_risk': int,          # Count of LOW tier flags
            'flags': List[Dict],      # All flagged chunks (full details)
            'summary': str            # Human-readable summary
        }
    """
    
    result = {
        'file': str(file_path),
        'file_name': Path(file_path).name,
        'status': 'error',
        'total_chunks': 0,
        'flagged_count': 0,
        'high_risk': 0,
        'medium_risk': 0,
        'low_risk': 0,
        'flags': [],
        'summary': ''
    }
    
    try:
        # Extract text from PDF
        full_text = _extract_text_from_file(file_path)
        
        # Chunk the text
        chunks = _chunk_text(full_text)
        result['total_chunks'] = len(chunks)
        
        if not chunks:
            result['summary'] = f"No chunks extracted from {result['file_name']}"
            return result
        
        # Run Layer 1 detection on all chunks
        flags = detect_lexical(
            chunks,
            threshold=threshold,
            corpus_limit=corpus_limit,
            tfidf_path=tfidf_path,
            chroma_path=chroma_path
        )
        
        # Process results
        result['flags'] = flags
        result['flagged_count'] = len(flags)
        
        # Count by risk tier
        for flag in flags:
            tier = flag.get('risk_tier')
            if tier == 'HIGH':
                result['high_risk'] += 1
            elif tier == 'MEDIUM':
                result['medium_risk'] += 1
            elif tier == 'LOW':
                result['low_risk'] += 1
        
        result['status'] = 'success'
        
        # Generate summary
        if result['flagged_count'] == 0:
            result['summary'] = f"✅ {result['file_name']}: {result['total_chunks']} chunks, NO matches found"
        else:
            result['summary'] = (
                f"⚠️  {result['file_name']}: {result['total_chunks']} chunks, "
                f"{result['flagged_count']} flagged "
                f"(🔴 {result['high_risk']} HIGH, 🟡 {result['medium_risk']} MEDIUM, 🟢 {result['low_risk']} LOW)"
            )
        
        return result
        
    except Exception as e:
        result['summary'] = f"❌ Error: {str(e)}"
        return result

