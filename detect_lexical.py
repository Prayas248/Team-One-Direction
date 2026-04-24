from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def detect_lexical(chunks: list, threshold=0.70) -> list:
    """TF-IDF cosine similarity against entire corpus."""
    
    # Define the corpus directly in the function
    corpus = [
        "This is a sample text from the corpus.",
        "Another example of a text document in the corpus.",
        "This document contains academic content for testing."
    ]
    
    # Use the corpus directly instead of retrieving it from ChromaDB
    corpus_docs = corpus  # Replace ChromaDB retrieval with this static corpus
    
    # Combine the corpus and input chunks
    all_texts = corpus_docs + chunks
    
    # Compute TF-IDF vectors for all texts
    tfidf = TfidfVectorizer(ngram_range=(1, 2)).fit_transform(all_texts)
    n_corp = len(corpus_docs)
    
    # Initialize the list to store flagged chunks
    flags = []
    
    # Compare each chunk against the corpus
    for i, chunk in enumerate(chunks):
        sims = cosine_similarity(tfidf[n_corp + i], tfidf[:n_corp])[0]
        best = int(np.argmax(sims))
        if sims[best] >= threshold:
            flags.append({
                'chunk': chunk,
                'score': float(sims[best]),
                'matched': corpus_docs[best],
                'meta': {},  # Metadata is empty since no ChromaDB is used
                'layer': 1,
                'type': 'Verbatim/Near-Exact'
            })
    
    return flags

# Add the test case here
if __name__ == "__main__":
    # Sample corpus
    corpus = [
        "This is a sample text from the corpus.",
        "Another example of a text document in the corpus.",
        "This document contains academic content for testing."
    ]
    
    # Sample chunks to test
    chunks = [
        "This is a sample text from the corpus.",
        "A completely new text chunk for testing purposes."
    ]
    
    # Call the detect_lexical function
    flags = detect_lexical(chunks, threshold=0.75)
    
    # Print the results
    print("Flags Detected:")
    for flag in flags:
        print(flag)