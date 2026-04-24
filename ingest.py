import pdfplumber
import re
from typing import List, Dict

# Define section headers for boundary detection
SECTION_HEADERS = [
    r"abstract", r"introduction", r"related work", r"methodology", 
    r"methods", r"results", r"discussion", r"conclusion", r"references"
]

def extract_text(pdf_path: str) -> Dict[str, str]:
    """Extract text and detect section boundaries."""
    sections, current_section, buffer = {}, 'preamble', []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
            for line in text.split('\n'):
                low = line.strip().lower()
                if any(re.match(h, low) for h in SECTION_HEADERS):
                    sections[current_section] = ' '.join(buffer)
                    current_section, buffer = low, []
                else:
                    buffer.append(line)
        sections[current_section] = ' '.join(buffer)
    return sections

def chunk_text(text: str, size=200, overlap=50) -> List[str]:
    """Sliding window chunker — preserves context at boundaries."""
    words = text.split()
    return [' '.join(words[i:i+size]) for i in range(0, len(words)-size+1, size-overlap)]

# Debugging: Print available names in the current module
print(dir())