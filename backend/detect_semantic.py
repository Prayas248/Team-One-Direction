"""
Layer 2: Semantic Plagiarism Detection (SBERT + ChromaDB)

Detects paraphrased text and semantic reuse using pre-computed SBERT embeddings
stored in ChromaDB. This layer catches rephrasing that Layer 1 (TF-IDF) misses.

Research: Reimers & Gurevych (2019), Bohra & Barwar (2022), PlagiSense (2025)
"""

import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict


def detect_semantic(
    chunks: List[str],
    threshold: float = 0.75,
    top_k: int = 3,
    corpus_path: str = "data/chroma_index",
    model_name: str = "all-MiniLM-L6-v2"
) -> List[Dict]:
    """
    Layer 2: SBERT semantic similarity detection.
    
    Encodes manuscript chunks using SBERT embeddings and queries ChromaDB
    for semantically similar corpus chunks using nearest-neighbor search.
    
    Args:
        chunks: List of manuscript text chunks to analyze
        threshold: Minimum similarity score (0.75 = MEDIUM, per spec)
        top_k: Number of top matches to retrieve per chunk
        corpus_path: Path to ChromaDB persistent storage
        model_name: SBERT model identifier
    
    Returns:
        List of flagged passages:
        {
            'chunk': str,           # Original manuscript chunk
            'score': float,         # Cosine similarity (0.00-1.00)
            'matched': str,         # Matched corpus chunk
            'meta': {               # Corpus metadata
                'title': str,
                'doi': str,
                'domain': str,
                'year': str
            },
            'layer': 2,             # Detection layer
            'type': 'Paraphrase/Semantic'
        }
    
    Raises:
        FileNotFoundError: If ChromaDB index not found
        RuntimeError: If corpus is empty
    """
    
    # ──────────────────────────────────────────────────────────────
    # Step 1: Load SBERT model
    # ──────────────────────────────────────────────────────────────
    try:
        model = SentenceTransformer(model_name)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load SBERT model '{model_name}': {e}\n"
            f"Install: pip install sentence-transformers"
        )
    
    # ──────────────────────────────────────────────────────────────
    # Step 2: Connect to ChromaDB and retrieve corpus
    # ──────────────────────────────────────────────────────────────
    try:
        client = chromadb.PersistentClient(path=corpus_path)
        collection = client.get_collection("corpus")
    except Exception as e:
        raise FileNotFoundError(
            f"ChromaDB corpus not found at '{corpus_path}'. "
            f"Run: python scripts/build_corpus.py\n"
            f"Error: {e}"
        )
    
    corpus_data = collection.get(include=["documents", "metadatas", "embeddings"])
    corpus_docs = corpus_data["documents"]
    corpus_meta = corpus_data["metadatas"]
    
    if not corpus_docs:
        raise RuntimeError("Corpus is empty. Run build_corpus.py first.")
    
    # ──────────────────────────────────────────────────────────────
    # Step 3: Encode manuscript chunks using SBERT
    # ──────────────────────────────────────────────────────────────
    try:
        chunk_embeddings = model.encode(
            chunks,
            batch_size=32,
            show_progress_bar=False,
            convert_to_tensor=False
        )
    except Exception as e:
        raise RuntimeError(f"SBERT encoding failed: {e}")
    
    # ──────────────────────────────────────────────────────────────
    # Step 4: Query ChromaDB for nearest neighbors
    # ──────────────────────────────────────────────────────────────
    flags = []
    
    for chunk_idx, chunk_embedding in enumerate(chunk_embeddings):
        # Query ChromaDB nearest-neighbor search
        results = collection.query(
            query_embeddings=[chunk_embedding.tolist()],
            n_results=top_k
        )
        
        distances = results["distances"][0]
        matched_docs = results["documents"][0]
        matched_metas = results["metadatas"][0]
        matched_indices = results["ids"][0]  # Get corpus document IDs
        
        # ChromaDB cosine space: distance = 1 - cosine_similarity
        for dist, matched_doc, matched_meta, matched_id in zip(distances, matched_docs, matched_metas, matched_indices):
            similarity = 1 - dist  # Convert cosine distance to similarity
            
            if similarity >= threshold:
                # Extract corpus chunk index from metadata
                corpus_chunk_idx = matched_meta.get('chunk_index', 0)
                
                flags.append({
                    'chunk': chunks[chunk_idx],           # Actual manuscript chunk
                    'chunk_index': chunk_idx,             # Position in manuscript
                    'score': float(similarity),
                    'matched': matched_doc,
                    'matched_index': corpus_chunk_idx,    # Position in corpus
                    'meta': matched_meta,                 # Corpus metadata
                    'layer': 2,
                    'type': 'Paraphrase/Semantic'
                })
    
    return flags

