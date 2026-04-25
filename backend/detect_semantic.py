# Debugging imports
print("Importing SentenceTransformer...")
from sentence_transformers import SentenceTransformer
print("SentenceTransformer imported successfully.")

print("Connecting to ChromaDB...")
import chromadb
print("ChromaDB connected successfully.")

import os
from config.settings import settings

# Load the SBERT model
print("Loading SBERT model...")
MODEL = SentenceTransformer('all-MiniLM-L6-v2')  # Pre-trained SBERT model
print("SBERT model loaded successfully.")

# Function to verify PDF file input
def verify_pdf_file(pdf_path: str):
    """
    Verify if the PDF file exists and is accessible.
    
    Args:
        pdf_path (str): Path to the PDF file.
    
    Returns:
        bool: True if the file is accessible, False otherwise.
    """
    print(f"Checking if PDF file exists at: {pdf_path}")
    if os.path.exists(pdf_path):
        print("PDF file is accessible.")
        return True
    else:
        print("PDF file not found. Please check the file path.")
        return False

# Function to verify ChromaDB index
def verify_chromadb_index(index_path: str):
    """
    Verify if the ChromaDB index exists and is accessible.
    
    Args:
        index_path (str): Path to the ChromaDB index.
    
    Returns:
        bool: True if the index is accessible, False otherwise.
    """
    print(f"Checking if ChromaDB index exists at: {index_path}")
    if os.path.exists(index_path):
        print("ChromaDB index is accessible.")
        return True
    else:
        print("ChromaDB index not found. Please check the index path.")
        return False

# Function to check ChromaDB corpus collection
def check_chromadb_corpus(index_path: str):
    """
    Check if the ChromaDB corpus collection is available and contains data.
    
    Args:
        index_path (str): Path to the ChromaDB index.
    
    Returns:
        bool: True if the corpus collection is valid, False otherwise.
    """
    try:
        client = chromadb.PersistentClient(path=index_path)
        col = client.get_collection('corpus')
        docs = col.get()
        print(f"Number of documents in corpus: {len(docs['documents'])}")
        if len(docs['documents']) > 0:
            print("ChromaDB corpus collection is valid and contains data.")
            return True
        else:
            print("ChromaDB corpus collection is empty.")
            return False
    except Exception as e:
        print(f"Error accessing ChromaDB corpus collection: {e}")
        return False

# Semantic detection function
def detect_semantic(chunks: list, threshold=0.75, top_k=3) -> list:
    """
    SBERT embeddings + ChromaDB nearest-neighbor search for semantic detection.
    
    Args:
        chunks (list): List of text chunks from the manuscript.
        threshold (float): Similarity threshold for flagging matches.
        top_k (int): Number of top matches to retrieve for each chunk.
    
    Returns:
        list: Flags containing information about detected semantic matches.
    """
    from config.settings import settings
    
    # Verify ChromaDB index
    index_path = settings.CHROMA_INDEX_PATH
    if not verify_chromadb_index(index_path):
        print("ChromaDB index verification failed. Exiting.")
        return []

    # Check ChromaDB corpus collection
    if not check_chromadb_corpus(index_path):
        print("ChromaDB corpus collection check failed. Exiting.")
        return []

    # Connect to the ChromaDB persistent client
    print("Connecting to ChromaDB persistent client...")
    client = chromadb.PersistentClient(path=index_path)
    print("Connected to ChromaDB persistent client successfully.")
    
    col = client.get_collection('corpus')  # Reference corpus collection
    print("Retrieved reference corpus collection from ChromaDB.")

    flags = []  # List to store flagged matches

    # Encode the manuscript chunks into embeddings
    print("Encoding manuscript chunks into embeddings...")
    embeds = MODEL.encode(chunks, batch_size=32, show_progress_bar=True)
    print("Manuscript chunks encoded successfully.")

    # Perform nearest-neighbor search for each chunk
    print("Performing nearest-neighbor search for each chunk...")
    for i, (chunk, embed) in enumerate(zip(chunks, embeds)):
        results = col.query(query_embeddings=[embed.tolist()], n_results=top_k)

        # Process the results
        for dist, doc, meta in zip(
            results['distances'][0], results['documents'][0], results['metadatas'][0]
        ):
            # Convert L2 distance to cosine similarity
            similarity = 1 - (dist / 2)
            if similarity >= threshold:
                flags.append({
                    'chunk': chunk,  # The manuscript chunk
                    'score': similarity,  # Similarity score
                    'matched': doc,  # Matched source document
                    'meta': meta,  # Metadata of the matched source
                    'layer': 2,  # Detection layer (Semantic)
                    'type': 'Paraphrase/Semantic'  # Type of match
                })

    print("Nearest-neighbor search completed.")
    return flags

# Example usage
if __name__ == "__main__":
    # Directory path for papers
    papers_path = "/Users/anusripriya.s/Documents/one direction/Team-One-Direction/papers"
    
    # Verify directory
    if os.path.exists(papers_path):
        print("Processing PDF files in the directory...")
        for file_name in os.listdir(papers_path):
            if file_name.endswith(".pdf"):
                pdf_path = os.path.join(papers_path, file_name)
                print(f"Processing file: {pdf_path}")
                
                # Verify PDF file
                if verify_pdf_file(pdf_path):
                    print("Proceeding with semantic detection...")
                    # Example chunks (replace with actual chunks from your PDF processing)
                    chunks = ["This is a sample chunk of text.", "Another chunk of text for testing."]
                    flags = detect_semantic(chunks)
                    print(f"Detection completed for {file_name}. Flags:", flags)
                else:
                    print(f"PDF file verification failed for {file_name}. Skipping.")
    else:
        print("Directory not found. Please check the path.")