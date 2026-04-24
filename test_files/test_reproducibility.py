"""
Reproducibility Verification Test for Layer 1 (TF-IDF)

Tests that identical manuscript inputs produce identical TF-IDF scores
across multiple runs (FR-7 compliance).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

import chromadb
from detect_lexical import detect_lexical


def test_reproducibility():
    """
    Verify that running detect_lexical twice on the same manuscript
    produces identical results (scores).
    
    This tests FR-7: Reproducibility requirement
    """
    print("\n" + "="*70)
    print("REPRODUCIBILITY TEST: Layer 1 (TF-IDF)")
    print("="*70)
    print("\nRequirement (FR-7):")
    print("  'identical inputs SHALL produce identical outputs'")
    print("  → Same manuscript → Same scores every time\n")
    
    # Get a test chunk from corpus
    try:
        client = chromadb.PersistentClient(path="data/chroma_index")
        collection = client.get_collection("corpus")
        corpus_data = collection.get(limit=10)
        
        # Use first corpus chunk as test manuscript
        test_chunk = corpus_data["documents"][0]
        test_chunks = [test_chunk]
        
        print(f"Test input (first 80 chars): '{test_chunk[:80]}...'\n")
        
        # Run 1
        print("RUN 1:")
        flags_run1 = detect_lexical(test_chunks, threshold=0.50)
        scores_run1 = [f['score'] for f in flags_run1]
        print(f"  Flags: {len(flags_run1)}")
        print(f"  Scores: {scores_run1}")
        
        # Run 2 (identical input)
        print("\nRUN 2 (identical input):")
        flags_run2 = detect_lexical(test_chunks, threshold=0.50)
        scores_run2 = [f['score'] for f in flags_run2]
        print(f"  Flags: {len(flags_run2)}")
        print(f"  Scores: {scores_run2}")
        
        # Run 3 (one more time)
        print("\nRUN 3 (identical input again):")
        flags_run3 = detect_lexical(test_chunks, threshold=0.50)
        scores_run3 = [f['score'] for f in flags_run3]
        print(f"  Flags: {len(flags_run3)}")
        print(f"  Scores: {scores_run3}")
        
        # Verify reproducibility
        print("\n" + "-"*70)
        print("VERIFICATION:")
        
        all_same = True
        tolerance = 1e-10  # Floating point tolerance
        
        # Check Run 1 vs Run 2
        if len(scores_run1) != len(scores_run2):
            print("  ❌ Number of flags differs between Run 1 and Run 2")
            all_same = False
        else:
            score_diff_1_2 = max(abs(s1 - s2) for s1, s2 in zip(scores_run1, scores_run2))
            if score_diff_1_2 > tolerance:
                print(f"  ❌ Scores differ between Run 1 and Run 2 (max diff: {score_diff_1_2})")
                all_same = False
            else:
                print(f"  ✅ Run 1 == Run 2 (scores identical, diff < {tolerance})")
        
        # Check Run 2 vs Run 3
        if len(scores_run2) != len(scores_run3):
            print("  ❌ Number of flags differs between Run 2 and Run 3")
            all_same = False
        else:
            score_diff_2_3 = max(abs(s2 - s3) for s2, s3 in zip(scores_run2, scores_run3))
            if score_diff_2_3 > tolerance:
                print(f"  ❌ Scores differ between Run 2 and Run 3 (max diff: {score_diff_2_3})")
                all_same = False
            else:
                print(f"  ✅ Run 2 == Run 3 (scores identical, diff < {tolerance})")
        
        # Overall result
        print("\n" + "="*70)
        if all_same:
            print("✅ REPRODUCIBILITY TEST PASSED")
            print("   FR-7 requirement: COMPLIANT")
            print("   Identical inputs consistently produce identical outputs")
            return True
        else:
            print("❌ REPRODUCIBILITY TEST FAILED")
            print("   FR-7 requirement: VIOLATED")
            print("   Scores are not reproducible across runs")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_reproducibility()
    sys.exit(0 if success else 1)
