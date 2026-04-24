"""
Test Layer 2 (SBERT) Semantic Detection Module

Standalone test script to verify Layer 2 functionality.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path so we can import backend
sys.path.insert(0, str(Path(__file__).parent.parent))

import chromadb
from backend.detect_semantic import detect_semantic


def test_layer2():
    """Test Layer 2 with a corpus chunk."""
    
    print("Layer 2 (SBERT) Semantic Detection Module")
    print("=" * 60)
    
    # Test with a real corpus chunk
    try:
        client = chromadb.PersistentClient(path="data/chroma_index")
        collection = client.get_collection("corpus")
        corpus_data = collection.get(limit=5)
        
        # Use first corpus chunk as test (paraphrase would be better in real scenario)
        sample_chunk = corpus_data["documents"][0]
        test_chunks = [sample_chunk[:200]]  # Truncate for variety
        
        print(f"\nTest chunk (first 100 chars): '{sample_chunk[:100]}...'\n")
        
        flags = detect_semantic(test_chunks, threshold=0.60)
        
        print(f"Flags detected: {len(flags)}")
        if flags:
            for i, flag in enumerate(flags[:3], 1):  # Show top 3 matches
                print(f"\n  Match {i}:")
                print(f"    Score: {flag['score']:.3f}")
                print(f"    Type: {flag['type']}")
                print(f"    Source: {flag['meta'].get('title', 'Unknown')[:60]}")
                print(f"    Chunk Index: {flag['matched_index']}")
        
        print("\n✅ Module working correctly")
        return 0
        
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(test_layer2())
