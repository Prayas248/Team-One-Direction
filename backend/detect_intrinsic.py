import logging
import re
import unicodedata
from typing import Dict, List, Optional, Tuple
import numpy as np
import textstat

logger = logging.getLogger(__name__)

# ─── Tuning constants (match spec FR-4 and Section 2.5) ──────────────────────
# sections below this are skipped
# need this many valid sections for z-score baseline
# standard deviations to flag (spec Section 2.5)
# z=4.0 maps to score=1.0; z=2.0 maps to score=0.5
MIN_SECTION_WORDS = 50
MIN_SECTIONS = 3
Z_THRESHOLD = 2.0
Z_NORMALISER = 4.0
CHUNK_CHARS = 500  # excerpt length for UI display

FEATURE_LABELS = {
    "ttr": "vocabulary richness (Type-Token Ratio)",
    "avg_sent_len": "average sentence length",
    "readability": "readability score (Flesch-Kincaid)",
}


# ─── Dynamic threshold ────────────────────────────────────────────────────────
def _get_dynamic_threshold(n_sections: int, base_threshold: float = Z_THRESHOLD) -> float:
    """
    Scale z-threshold based on number of valid sections.

    With very few sections, z-scores are mathematically constrained
    (3 sections → max z ≈ 1.15). Scaling the threshold down for smaller
    section counts ensures anomalies can still be detected regardless of
    paper structure, while preserving the spec threshold of 2.0 for
    well-structured papers with many sections.

    n_sections <= 3  → 1.0
    n_sections <= 5  → 1.3
    n_sections <= 8  → 1.6
    n_sections >  8  → 2.0 (spec default)
    """
    if n_sections <= 3:
        return 1.0
    elif n_sections <= 5:
        return 1.3
    elif n_sections <= 8:
        return 1.6
    else:
        return base_threshold


# ─── Text cleaning ────────────────────────────────────────────────────────────
def _clean_section_text(raw: str) -> str:
    """
    Clean text that has come through pdfplumber extraction.

    pdfplumber introduces several artefacts that corrupt stylometry:
    - Unicode ligatures: "fi" "fl" not split into fi fl → inflates TTR
    - Soft hyphens and hyphenated line breaks: "meth-\nodology" → "methodology"
    - Non-breaking spaces, zero-width joiners
    - Stray page numbers mid-text: "\n12\n"
    - Multiple spaces from column layout

    All of these are fixed here before any feature is computed.
    """
    if not raw or not raw.strip():
        return ""

    # Normalise unicode (resolves ligatures, accented chars, etc.)
    text = unicodedata.normalize("NFKC", raw)

    # Rejoin PDF hyphenated line breaks
    text = re.sub(r"-\s*\n\s*", "", text)

    # Remove stray page numbers injected mid-text by pdfplumber
    text = re.sub(r"\n\s*\d{1,3}\s*\n", " ", text)

    # Collapse tabs, carriage returns, form feeds to space
    text = re.sub(r"[\t\r\f\v]+", " ", text)

    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)

    # Collapse multiple newlines to single space
    text = re.sub(r"\n+", " ", text)

    return text.strip()


def _split_sentences(text: str) -> List[str]:
    """
    Split text into sentences without splitting on abbreviations,
    decimal numbers, or initials — all common in academic papers.

    Handles: et al. / Fig. / Eq. / vs. / i.e. / e.g. / cf.
    Dr. / Prof. / p < 0.05 / 3.14 / J. Smith
    """
    ABBREV = (
        r"et al|fig|figs|eq|eqs|vs|i\.e|e\.g|cf|dr|prof|mr|mrs|ms|"
        r"no|vol|pp|ch|sec|ref|approx|dept|est|max|min|avg|std|"
        r"jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec|"
        r"[a-z]"  # single-letter initials
    )

    # Temporarily protect abbreviation periods
    text = re.sub(rf"\b({ABBREV})\.", r"\1§", text, flags=re.IGNORECASE)

    # Protect decimal numbers
    text = re.sub(r"(\d)\.(\d)", r"\1§\2", text)

    # Split on sentence-ending punctuation followed by space + capital
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"\(])", text)

    # Restore protected tokens
    sentences = [p.replace("§", ".") for p in parts]

    return [s.strip() for s in sentences if len(s.strip()) > 10]


# ─── Feature extraction ───────────────────────────────────────────────────────
def compute_style_features(text: str) -> Optional[Dict[str, float]]:
    """
    Compute three stylometric features for one section of text.

    FEATURE 1: Type-Token Ratio (TTR)
    unique_words / total_words (after lowercasing)

    Measures vocabulary richness. Plagiarised text from another
    author spikes or drops TTR relative to the paper's own baseline.

    Range: 0.0 → 1.0. Academic prose typically 0.40–0.65.

    FEATURE 2: Average Sentence Length
    mean(words per sentence) across all sentences in the section.

    Different authors write with different rhythms. Dense theoretical
    prose: 25–40 words/sentence. Empirical sections: 12–18.
    A sudden shift indicates a different voice.

    FEATURE 3: Flesch Reading Ease
    A validated formula combining syllable count and sentence length.

    0–30 = very difficult (dense academic)
    30–50 = difficult
    50–70 = standard
    70+ = easy / conversational

    This single number captures both vocabulary AND structural
    complexity — the strongest single stylometric discriminator.

    Args:
        text: Raw section text. Cleaned internally.

    Returns:
        Dict with float values for ttr, avg_sent_len, readability.
        Returns None if text is too short or empty after cleaning.
    """
    text = _clean_section_text(text)
    words = text.split()

    if len(words) < 10:
        return None

    # TTR — lowercase to avoid "The" ≠ "the" inflating unique count
    lower_words = [
        w.lower().strip(".,;:()[]\"'")
        for w in words
        if w.strip(".,;:()[]\"'")
    ]
    ttr = len(set(lower_words)) / len(lower_words) if lower_words else 0.0

    # Average sentence length
    sentences = _split_sentences(text)
    if sentences:
        avg_sent_len = float(np.mean([len(s.split()) for s in sentences]))
    else:
        avg_sent_len = float(len(words))  # treat whole section as one sentence

    # Flesch Reading Ease — pure Python implementation (industry-standard)
    def _count_syllables(word: str) -> int:
        word = re.sub(r"[^a-z]", "", word.lower())
        if not word:
            return 1

        vowels = "aeiouy"
        count = 0
        prev_vowel = False

        for ch in word:
            is_vowel = ch in vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel

        # silent trailing e
        if word.endswith("e") and count > 1:
            count -= 1

        return max(count, 1)

    try:
        total_words = len(words)
        total_sentences = max(len(sentences), 1)
        total_syllables = sum(_count_syllables(w) for w in lower_words)

        readability = (
            206.835
            - 1.015 * (total_words / total_sentences)
            - 84.6 * (total_syllables / total_words)
        )

        readability = max(-100.0, min(121.0, readability))

    except Exception:
        readability = 50.0

    return {
        "ttr": round(ttr, 6),
        "avg_sent_len": round(avg_sent_len, 6),
        "readability": round(readability, 6),
    }


# ─── Z-score helpers ──────────────────────────────────────────────────────────
def _zscore_array(values: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Z-score an array. Returns (zscores, mean, std).

    Returns all-zeros zscores if std < 1e-6 (no variance to analyse).
    """
    mean = float(values.mean())
    std = float(values.std())

    if std < 1e-6:
        return np.zeros_like(values, dtype=float), mean, std

    return (values - mean) / std, mean, std


def _score_from_z(z: float) -> float:
    """
    Normalise z-score magnitude to 0.0–1.0 for score.aggregate_flags().

    This maps Layer 3 onto the same scale as Layers 1 and 2 so that
    score.py can apply tier thresholds uniformly across all layers.

    z=2.0 → 0.50 (sits at LOW/MEDIUM boundary)
    z=2.8 → 0.70 (MEDIUM tier threshold per spec)
    z=3.4 → 0.85 (HIGH tier threshold per spec)
    z=4.0 → 1.00 (capped at maximum)
    """
    return round(min(abs(z) / Z_NORMALISER, 1.0), 6)


# ─── Main detection function ──────────────────────────────────────────────────
def detect_intrinsic(
    sections: Dict[str, str],
    z_threshold: float = Z_THRESHOLD,
) -> List[dict]:
    """
    Detect writing style anomalies across paper sections.

    This is the function called by api.py:
    l3 = detect_intrinsic(sections)

    Args:
        sections: Direct output of ingest.extract_text().
        {'section_name': 'full section text', ...}

        z_threshold: Standard deviations to flag. Default 2.0 per spec.
        Dynamically scaled down for papers with fewer sections.

    Returns:
        List of flag dicts for score.aggregate_flags().
        Always a list. Never None.
    """
    print("\n=== DETECT_INTRINSIC CALLED ===")
    print(f"Sections received: {list(sections.keys())}")

    if not sections:
        print("ERROR: empty sections dict")
        return []

    # ── Step 1: Validate and extract features per section ─────────
    valid: Dict[str, dict] = {}

    for name, text in sections.items():
        if not isinstance(text, str):
            print(f"  SKIP '{name}': not a string")
            continue

        word_count = len(text.split())
        print(f"  '{name}': {word_count} words", end="")

        if word_count < MIN_SECTION_WORDS:
            print(f" — SKIP (< {MIN_SECTION_WORDS} words)")
            continue

        features = compute_style_features(text)

        if features is None:
            print(f" — SKIP (feature extraction failed)")
            continue

        print(f" — OK | ttr={features['ttr']:.4f} avg_sent={features['avg_sent_len']:.1f} read={features['readability']:.1f}")

        valid[name] = {
            "text": text,
            "features": features,
            "words": word_count,
        }

    print(f"Valid sections after filtering: {len(valid)} / {len(sections)}")

    # ── Step 2: Check minimum sections for meaningful baseline ────
    if len(valid) < MIN_SECTIONS:
        print(f"BAIL: only {len(valid)} valid section(s), need {MIN_SECTIONS} — returning []")
        return []

    # ── Step 3: Determine effective threshold based on section count ──
    effective_threshold = _get_dynamic_threshold(len(valid), z_threshold)
    print(f"\nSections: {len(valid)} → effective z-threshold: {effective_threshold}")

    # ── Step 4: Z-score each feature across all valid sections ────
    flags: List[dict] = []
    section_names = list(valid.keys())

    print(f"Z-score analysis:")

    for feat in ["ttr", "avg_sent_len", "readability"]:
        values = np.array(
            [valid[s]["features"][feat] for s in section_names]
        )

        zscores, mean, std = _zscore_array(values)

        if std < 1e-6:
            print(f"  {feat}: zero variance — skip")
            continue

        print(f"  {feat}: mean={mean:.4f} std={std:.4f}")

        for sec_name, z in zip(section_names, zscores):
            z = float(z)
            tag = " ← ANOMALY" if abs(z) >= effective_threshold else ""
            print(f"    '{sec_name}': z={z:+.3f}{tag}")

            if abs(z) < effective_threshold:
                continue

            score = _score_from_z(z)
            section_text = valid[sec_name]["text"]

            flags.append(
                {
                    # ── Fields required by score.aggregate_flags() ───
                    "score": score,
                    "layer": 3,
                    "type": "Style Anomaly (Intrinsic)",
                    "chunk": section_text[:CHUNK_CHARS],

                    # ── Fields required by explain.explain_flag() ────
                    # explain.py layer==3 branch uses section, feature, z_score
                    "section": sec_name,
                    "feature": feat,
                    "z_score": round(z, 6),

                    # ── Fields for Streamlit UI (P4 app.py) ──────────
                    "meta": {
                        "feature_value": round(
                            float(valid[sec_name]["features"][feat]),
                            6,
                        ),
                        "paper_mean": round(mean, 6),
                        "paper_std": round(std, 6),
                        "feature_label": FEATURE_LABELS[feat],
                        "word_count": valid[sec_name]["words"],
                        "all_sections": {
                            s: round(
                                float(valid[s]["features"][feat]),
                                4,
                            )
                            for s in section_names
                        },
                    },

                    # NOTE: no "matched" key — Layer 3 has no corpus source.
                    # explain.py must handle layer==3 without accessing "matched".
                }
            )

    print(f"\nLayer 3 complete — {len(flags)} flag(s) raised")
    print("================================\n")
    return flags