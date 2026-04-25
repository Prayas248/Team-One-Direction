from flask import Flask, request, jsonify
from backend.ingest import extract_text, chunk_text
from backend.detect_lexical import detect_lexical
from backend.detect_semantic import detect_semantic
from backend.detect_intrinsic import detect_intrinsic
from backend.score import aggregate_flags
from backend.explain import explain_flag, GROQ_STATUS
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile, os
from pathlib import Path

app = Flask(__name__)
print("API script started...")

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "status": "running",
        "message": "Plagiarism API is live. Use POST /analyse"
    })

@app.route('/status', methods=['GET'])
def status():
    """Check system status and component health"""
    import chromadb
    from scipy import sparse
    import json
    
    status_info = {
        "api": "✓ Running",
        "components": {}
    }
    
    # Check ChromaDB
    try:
        client = chromadb.PersistentClient(path='data/chroma_index')
        col = client.get_collection('corpus')
        docs = col.get()
        status_info["components"]["chromadb"] = {
            "status": "✓",
            "chunks": len(docs['documents'])
        }
    except Exception as e:
        status_info["components"]["chromadb"] = {"status": "✗", "error": str(e)}
    
    # Check TF-IDF
    try:
        if Path('data/tfidf_corpus/corpus_metadata.json').exists():
            meta = json.load(open('data/tfidf_corpus/corpus_metadata.json'))
            vecs = sparse.load_npz('data/tfidf_corpus/corpus_vectors.npz')
            status_info["components"]["tfidf"] = {
                "status": "✓",
                "chunks": vecs.shape[0],
                "vocabulary": vecs.shape[1]
            }
        else:
            status_info["components"]["tfidf"] = {"status": "✗", "error": "Files not found"}
    except Exception as e:
        status_info["components"]["tfidf"] = {"status": "✗", "error": str(e)}
    
    # Check SBERT
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        status_info["components"]["sbert"] = {
            "status": "✓",
            "model": "all-MiniLM-L6-v2",
            "dimensions": 384
        }
    except Exception as e:
        status_info["components"]["sbert"] = {"status": "✗", "error": str(e)}
    
    # Check Groq
    status_info["components"]["groq"] = {
        "status": GROQ_STATUS
    }
    
    # Check config
    try:
        from config.settings import settings
        status_info["components"]["config"] = {
            "status": "✓",
            "has_groq_key": "GROQ_API_KEY" in os.environ or hasattr(settings, 'GROQ_API_KEY')
        }
    except Exception as e:
        status_info["components"]["config"] = {"status": "✗", "error": str(e)}
    
    return jsonify(status_info)

@app.route('/analyse', methods=['POST'])
def analyse():
    # Save the uploaded PDF temporarily
    f = request.files['pdf']
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
        f.save(tmp.name)

    # Extract text and sections
    sections = extract_text(tmp.name)
    os.unlink(tmp.name)  # Delete the temporary file

    # DEBUG
    print("=== SECTIONS FOUND ===")
    print("Count:", len(sections))
    for name, text in sections.items():
        print(f"  '{name}': {len(text.split())} words")
    print("======================")

    full_text = ' '.join(sections.values())
    chunks = chunk_text(full_text)

    # DEBUG
    print(f"Total chunks: {len(chunks)}")

    # Run detection layers in parallel (FR-2: three layers operating in parallel)
    print("\n🚀 Starting parallel detection (3 layers)...")
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        # Submit all three detection tasks
        l1_future = executor.submit(detect_lexical, chunks)
        l2_future = executor.submit(detect_semantic, chunks)
        l3_future = executor.submit(detect_intrinsic, sections)
        
        # Collect results as they complete
        l1 = l1_future.result()
        l2 = l2_future.result()
        l3 = l3_future.result()
    
    print("✓ All detection layers complete\n")

    # Aggregate results
    result = aggregate_flags(l1, l2, l3)

    # Add explanations for High and Medium flags (Stage 4)
    print("\n📝 Stage 4: Generating LLM explanations for HIGH/MEDIUM flags...")
    for flag in result['flags']:
        if flag['tier'] in ('HIGH', 'MEDIUM'):
            flag['explanation'] = explain_flag(flag)
    print("✓ Explanations complete\n")

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=5000)