"""
Test Layer 1 (TF-IDF) on all papers in papers/ folder

Uses the new detect_lexical_file() function to process entire papers.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.detect_lexical import detect_lexical_file


def test_all_papers():
    """Run Layer 1 on all papers in papers/ folder."""
    
    print("\n" + "=" * 80)
    print("LAYER 1 (TF-IDF) DETECTION ON ALL PAPERS")
    print("=" * 80 + "\n")
    
    papers_dir = Path("papers")
    
    if not papers_dir.exists():
        print(f"❌ Papers directory not found: {papers_dir}")
        return 1
    
    # Find all PDF files
    pdf_files = sorted(papers_dir.glob("*.pdf"))
    
    if not pdf_files:
        print(f"❌ No PDF files found in {papers_dir}")
        return 1
    
    print(f"📄 Found {len(pdf_files)} papers\n")
    
    # Process each paper
    all_results = []
    total_flagged = 0
    total_chunks = 0
    
    for pdf_file in pdf_files:
        print(f"Processing: {pdf_file.name}...")
        
        result = detect_lexical_file(
            str(pdf_file),
            threshold=0.55
        )
        
        all_results.append(result)
        total_flagged += result['flagged_count']
        total_chunks += result['total_chunks']
        
        # Print result
        print(f"  {result['summary']}")
        
        # Show details if flagged
        if result['flags']:
            for i, flag in enumerate(result['flags'], 1):
                print(f"\n    Match {i}:")
                print(f"      Score: {flag['score']:.3f} ({flag['risk_tier']})")
                print(f"      Type: {flag['type']}")
                print(f"      Source: {flag['meta'].get('title', 'Unknown')[:70]}")
                print(f"      Matched text: {flag['matched'][:100]}...")
        
        print()
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    for result in all_results:
        status_icon = "✅" if result['flagged_count'] == 0 else "⚠️ "
        print(f"{status_icon} {result['file_name']:50} {result['total_chunks']:3} chunks, {result['flagged_count']:2} matches")
    
    print("\n" + "-" * 80)
    print(f"Total papers:     {len(pdf_files)}")
    print(f"Total chunks:     {total_chunks}")
    print(f"Total flagged:    {total_flagged}")
    print("=" * 80 + "\n")
    
    return 0 if total_flagged == 0 else 1


if __name__ == "__main__":
    sys.exit(test_all_papers())
