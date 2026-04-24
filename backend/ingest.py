# backend/ingest.py

import pdfplumber
import re
import sys
from pathlib import Path
from typing import List, Dict

# ── Section header patterns ────────────────────────────────────────────────────
SECTION_HEADERS = [
    r"abstract",
    r"introduction",
    r"related work",
    r"literature review",
    r"background",
    r"methodology",
    r"methods",
    r"materials and methods",
    r"experimental( setup)?",
    r"results( and discussion)?",
    r"discussion",
    r"conclusion(s)?",
    r"future (scope|work)",
    r"acknowledgements?",
    r"references",
    r"appendix",
]

# ── Noise line patterns ────────────────────────────────────────────────────────
NOISE_PATTERNS = [
    r"www\.\S+\.com",
    r"editor@\S+",
    r"@international journal",
    r"issn\s*:",
    r"e-issn",
    r"impact\s+factor",
    r"vol\.\s*\d+",
    r"^\s*\d+\s*$",
    r"int(ernational)?\s+peer",
    r"all rights reserved",
    r"doi\s*:",
    r"received\s*:\s*\d",
    r"accepted\s*:\s*\d",
    r"published\s*:\s*\d",
    r"©\s*20\d\d",
    r"page\s+\d+\s+of\s+\d+",
]


def clean_encoding_artifacts(text: str) -> str:
    """Remove PDF encoding artifacts like (cid:X) and normalize spaces."""
    # Remove encoding markers: (cid:123), (cid:1), etc.
    text = re.sub(r"\(cid:\d+\)", "", text)
    # Remove other common PDF artifacts
    text = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f]", "", text)
    # Normalize whitespace: collapse multiple spaces, preserve single newlines
    text = re.sub(r" +", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Fix common ligatures and symbols that might appear as artifacts
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")
    text = text.replace("ﬀ", "ff").replace("ﬃ", "ffi").replace("ﬄ", "ffl")
    return text


def is_noise_line(line: str) -> bool:
    low = line.strip().lower()
    if not low:
        return True
    if len(low.split()) <= 2 and low.replace(" ", "").isdigit():
        return True
    if any(re.search(p, low) for p in NOISE_PATTERNS):
        return True
    return False


def extract_text(pdf_path: str) -> Dict[str, str]:
    """
    Extract text from a PDF and split into named sections.
    Falls back to 'preamble' if no headers detected.
    """
    sections = {}
    current_section = "preamble"
    buffer = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # Try extraction with char_margin to preserve word spacing better
            # layout=False extracts text in reading order without layout preservation
            text = page.extract_text(
                x_tolerance=3,
                y_tolerance=3,
                layout=False,
                char_margin=10  # Better word spacing detection
            ) or ""
            # Clean encoding artifacts and normalize spacing
            text = clean_encoding_artifacts(text)
            for line in text.split("\n"):
                if is_noise_line(line):
                    continue
                stripped = re.sub(r"^([\dIVXivx]+[\.\d]*)\s*", "", line.strip()).lower()
                is_header = (
                    any(re.match(h, stripped) for h in SECTION_HEADERS)
                    and len(stripped.split()) <= 6
                    and len(stripped) >= 4
                )
                if is_header:
                    sections[current_section] = " ".join(buffer)
                    current_section = stripped
                    buffer = []
                else:
                    buffer.append(line)

    sections[current_section] = " ".join(buffer)
    return {k: v for k, v in sections.items() if v.strip()}


def chunk_text(text: str, size: int = 200, overlap: int = 50) -> List[str]:
    """Sliding window chunker. Skips fragments under 50 words."""
    words = text.split()
    chunks = []
    step = size - overlap
    for i in range(0, max(1, len(words) - size + 1), step):
        chunk = " ".join(words[i: i + size])
        if len(chunk.split()) >= 50:
            chunks.append(chunk)
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# CLI: batch run over a papers/ folder
# Usage:  python ingest.py papers/
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("papers")
    pdfs = sorted(folder.glob("*.pdf"))

    if not pdfs:
        print(f"No PDFs found in '{folder}'. Put your papers there and retry.")
        sys.exit(1)

    print(f"\nFound {len(pdfs)} PDFs in '{folder}'\n")
    print(f"{'File':<45} {'Sections':>9} {'Words':>8} {'Chunks':>8}  Notes")
    print("-" * 85)

    for pdf_path in pdfs:
        try:
            sections = extract_text(str(pdf_path))
            full_text = " ".join(sections.values())
            chunks = chunk_text(full_text)
            words = len(full_text.split())

            note = ""
            if list(sections.keys()) == ["preamble"]:
                note = "⚠ no headers detected"
            if words < 500:
                note += " ⚠ very short"
            if len(chunks) == 0:
                note += " ❌ zero chunks"

            status = "✅" if not note else "⚠ "
            print(f"{status} {pdf_path.name:<43} {len(sections):>9} {words:>8,} {len(chunks):>8}  {note}")

        except Exception as e:
            print(f"❌ {pdf_path.name:<43} ERROR: {e}")

    print()