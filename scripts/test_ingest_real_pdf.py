# scripts/test_ingest_real_pdf.py
# Tests ingest.py on a real PDF file.
#
# Usage:
#   python scripts/test_ingest_real_pdf.py papers/yourpaper.pdf
#
# This is STAGE 1 verification. Run this before build_corpus.py.

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from ingest import extract_text, chunk_text

if len(sys.argv) < 2:
    print("Usage: python scripts/test_ingest_real_pdf.py papers/yourpaper.pdf")
    sys.exit(1)

PDF_PATH = sys.argv[1]

if not os.path.exists(PDF_PATH):
    print(f"❌ File not found: {PDF_PATH}")
    sys.exit(1)

print(f"\n=== Real PDF Ingest Test: {os.path.basename(PDF_PATH)} ===\n")

# ── Extract ──────────────────────────────────────────────────────────
sections = extract_text(PDF_PATH)
print(f"Sections detected ({len(sections)}):")
for name, text in sections.items():
    wc = len(text.split())
    print(f"  [{name}]: {wc:,} words")

# ── Chunk ────────────────────────────────────────────────────────────
full_text = " ".join(sections.values())
chunks = chunk_text(full_text)
total_words = len(full_text.split())

print(f"\nTotal words : {total_words:,}")
print(f"Total chunks: {len(chunks)}")

if chunks:
    print(f"\nFirst chunk preview:")
    print(f"  '{chunks[0][:200]}...'")
    print(f"\nLast chunk preview:")
    print(f"  '{chunks[-1][:200]}...'")

def check_extraction_quality(sections: dict, full_text: str) -> dict:
    """
    Check for signs of poor PDF extraction.
    Returns quality metrics.
    """
    quality = {
        "has_concatenated_words": False,
        "avg_word_length": 0,
        "suspicious_section_names": False,
        "text_quality": "good"
    }
    
    # Check for unnaturally long words (sign of concatenation)
    all_words = full_text.split()
    if all_words:
        quality["avg_word_length"] = sum(len(w) for w in all_words) / len(all_words)
        # English avg word length is ~4.5-5 chars; >7 suggests issues
        if quality["avg_word_length"] > 7:
            quality["has_concatenated_words"] = True
    
    # Check for malformed section names (sign of extraction issues)
    for section_name in sections.keys():
        # Sections with >50 chars are usually malformed
        if len(section_name) > 50 or (
            "are" in section_name and "grateful" in section_name
        ):
            quality["suspicious_section_names"] = True
    
    if quality["has_concatenated_words"] or quality["suspicious_section_names"]:
        quality["text_quality"] = "poor"
    
    return quality


# ── Assertions ───────────────────────────────────────────────────────
print("\n--- Checks ---")
ok = True

# Check extraction quality
quality = check_extraction_quality(sections, full_text)

if quality["text_quality"] == "poor":
    print(f"⚠️  TEXT EXTRACTION QUALITY: POOR")
    if quality["has_concatenated_words"]:
        print(f"   → Avg word length: {quality['avg_word_length']:.1f} chars (suggests concatenated words)")
    if quality["suspicious_section_names"]:
        print(f"   → Malformed section names detected (likely encoding issue)")
    print(f"   This PDF may have font/encoding issues. Results are still usable but quality is degraded.")
    print(f"   Consider using a tool like Acrobat to re-export the PDF.")
else:
    print(f"✅ Text extraction quality: Good (avg word length: {quality['avg_word_length']:.1f} chars)")

if len(sections) == 0:
    print("❌ No sections extracted at all. PDF may be image-based (scanned).")
    ok = False
elif list(sections.keys()) == ["preamble"]:
    print("⚠  All text in 'preamble' — no section headers matched.")
    print("   This is okay for detection layers, but check if your PDF has selectable text.")
else:
    print(f"✅ Sections detected: {list(sections.keys())}")

if total_words < 500:
    print(f"⚠  Only {total_words} words extracted. Minimum expected ~1000 for a real paper.")
    print("   If this is a scanned PDF, pdfplumber cannot extract text from images.")
    ok = False
else:
    print(f"✅ Word count OK: {total_words:,} words")

if len(chunks) == 0:
    print("❌ Zero chunks produced. Paper too short or extraction failed.")
    ok = False
elif len(chunks) < 3:
    print(f"⚠  Only {len(chunks)} chunks. Very short paper — detection may be unreliable.")
else:
    print(f"✅ Chunk count OK: {len(chunks)} chunks")

# Check chunk format is correct for ChromaDB ingestion downstream
if chunks:
    assert all(isinstance(c, str) for c in chunks), "Chunks must be strings"
    assert all(len(c.split()) >= 50 for c in chunks), "All chunks must be ≥50 words"
    print(f"✅ Chunk format valid (all strings, all ≥50 words) — ready for ChromaDB")

print()
if ok:
    print("✅ STAGE 1 VERIFIED — ingest.py works on this PDF.")
    if quality["text_quality"] == "poor":
        print("   ⚠️  (with degraded text quality — see warnings above)")
    print("   Next: python scripts/build_corpus.py")
else:
    print("❌ Ingest issues found — fix before proceeding to corpus build.")
    sys.exit(1)