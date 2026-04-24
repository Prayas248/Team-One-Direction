"""
Test Layer 1 (TF-IDF) Lexical Detection Module

Standalone test script to verify Layer 1 functionality.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path so we can import backend
sys.path.insert(0, str(Path(__file__).parent.parent))

import chromadb
from backend.detect_lexical import detect_lexical


def test_layer1():
    """Test Layer 1 with a corpus chunk."""
    
    print("Layer 1 (TF-IDF) Lexical Detection Module")
    print("=" * 60)
    
    # Test with a real corpus chunk
    try:
        # Get corpus chunk for testing
        client = chromadb.PersistentClient(path="data/chroma_index")
        collection = client.get_collection("corpus")
        corpus_data = collection.get(limit=5)
        
        # Use first corpus chunk as test (exact match)
        sample_chunk = corpus_data["documents"][0]
        test_chunks = [sample_chunk]
        
        print(f"\nTest chunk (first 100 chars): '{sample_chunk[:100]}...'\n")
        
        flags = detect_lexical(test_chunks, threshold=0.50)
        
        print(f"Flags detected: {len(flags)}")
        if flags:
            flag = flags[0]
            print(f"  Score: {flag['score']:.3f}")
            print(f"  Risk Tier: {flag['risk_tier']}")
            print(f"  Type: {flag['type']}")
            print(f"  Chunk Index: {flag['chunk_index']}")
            print(f"  Source: {flag['meta'].get('title', 'Unknown')[:60]}")
        
        print("\n✅ Module working correctly")
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(test_layer1())
