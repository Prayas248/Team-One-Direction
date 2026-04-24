# PRSE Codebase Guide for AI Agents

## Project Overview

**Plagiarism Risk Signal Engine (PRSE)** — An AI-powered editorial tool for detecting plagiarism in academic manuscripts. Unlike verdict tools, PRSE surfaces risk signals with full explainability to support human editorial judgment.

**Hackathon**: Starter (23-24 April 2026), Taylor & Francis Group  
**Theme**: EdTech — Academic plagiarism detection

---

## Architecture: 4-Stage Pipeline

```
STAGE 1: INGESTION
  Input: PDF manuscript
  Output: Extracted sections + text chunks (200-word windows, 50-word overlap)
  Module: backend/ingest.py
  Status: ✅ COMPLETE

STAGE 2: PARALLEL DETECTION (3 layers)
  ├─ Layer 1 (TF-IDF): Verbatim/near-exact copy detection
  │  Module: backend/detect_lexical.py
  │  Approach: Compute on-the-fly from ChromaDB corpus
  │  Status: ✅ WORKING
  │
  ├─ Layer 2 (SBERT): Paraphrase/semantic reuse detection
  │  Module: backend/detect_semantic.py
  │  Approach: Pre-computed embeddings in ChromaDB
  │  Status: ⏳ TEMPLATE CREATED (needs debugging)
  │
  └─ Layer 3 (Intrinsic): Style shift detection (corpus-independent)
     Module: detect_intrinsic.py (⚠️ at repo root, needs move to backend/)
     Approach: Stylometric analysis (readability, TTR, sentence length)
     Status: ✅ EXISTS
  
  Output: Merged flags from all 3 layers

STAGE 3: RISK AGGREGATION
  Input: All flags from Stage 2
  Output: Deduplicated flags + composite risk score (0-100)
  Module: score.py (⚠️ at repo root, needs move to backend/)
  Status: ⏳ EXISTS (needs integration)
  
STAGE 4: EXPLAINABILITY + REPORT
  Input: HIGH/MEDIUM flags
  Output: Editorial explanations + risk dashboard
  Module: explain.py (⚠️ at repo root, needs move to backend/)
  Status: ⏳ EXISTS (needs integration)
```

---

## Key Architecture Decisions

### ✅ TF-IDF Strategy: Compute On-The-Fly
- **What**: Calculate TF-IDF vectors during analysis, don't pre-store
- **Why**: TF-IDF is fast math (~50-100ms), not expensive ML like SBERT
- **Impact**: Simpler architecture, no duplicate storage, single source of truth
- **Reference**: [LAYER1_INTEGRATION_GUIDE.md](LAYER1_INTEGRATION_GUIDE.md#architecture-decision-summary)

### ✅ ChromaDB as Single Source of Truth
- **What**: All corpus data (documents, metadata, embeddings) lives in ChromaDB
- **Why**: Layer 1 reads documents, Layer 2 reads embeddings, both from same index
- **Impact**: No sync issues, easier maintenance, natural data flow

### ✅ Reproducibility Fix Applied (24 April 2026)
- **Issue**: TF-IDF vocabulary indices changed between runs
- **Fix**: Fit vectorizer on **CORPUS ONLY**, transform manuscript separately
- **Result**: Identical inputs now produce identical outputs (FR-7 compliant)
- **Test**: `scripts/test_reproducibility.py` (passes all 3 runs)
- **Details**: See [LAYER1_REPRODUCIBILITY_FIX.md](LAYER1_REPRODUCIBILITY_FIX.md)

### ❌ Common TF-IDF Mistake (DO NOT REPEAT)
```python
# WRONG: Fits on [corpus + manuscript], vocabulary changes per run
tfidf = TfidfVectorizer(...).fit_transform(corpus_docs + chunks)

# CORRECT: Fits on corpus only, transforms both in same space
tfidf.fit(corpus_docs)
corpus_vectors = tfidf.transform(corpus_docs)
manuscript_vectors = tfidf.transform(chunks)
```
**Impact**: Fitting on manuscript causes different vocabularies → violates FR-7
All detection layers return flags with consistent schema:
```python
{
    'chunk': str,              # Original text from manuscript
    'score': float,            # 0.00-1.00 (cosine similarity)
    'matched': str,            # Matched corpus chunk
    'meta': {
        'title': str,
        'doi': str,
        'domain': str,         # {nlp, climate, biology}
        'year': str
    },
    'layer': int,              # 1, 2, or 3
    'type': str                # Detection type
}
```

### ✅ Threshold Consistency (All Layers)
- **LOW**: 0.55–0.69 (flagged, needs review)
- **MEDIUM**: 0.70–0.84 (significant similarity)
- **HIGH**: ≥0.85 (very likely plagiarism signal)

---

## Repository Structure

### Current (Needs Reorganization)
```
.
├── backend/                    # Stage 1-2a complete
│   ├── ingest.py             # ✅ STAGE 1
│   ├── detect_lexical.py      # ✅ STAGE 2a (Layer 1)
│   └── detect_semantic.py     # ⏳ STAGE 2b template (Layer 2)
├── scripts/                    # Testing & corpus building
│   ├── build_corpus.py        # Build ChromaDB corpus
│   ├── test_ingest_real_pdf.py
│   ├── test_pipeline_stages.py
│   └── test_layers_1_2_3.py   # (create for Layer 1-3 testing)
├── config/
│   └── settings.py
├── detect_intrinsic.py        # ⚠️ SHOULD MOVE TO backend/
├── explain.py                 # ⚠️ SHOULD MOVE TO backend/
├── score.py                   # ⚠️ SHOULD MOVE TO backend/
├── api.py                     # ⚠️ SHOULD MOVE TO backend/ + refactor
├── test_stages_3_4.py         # ⏳ Verify Stage 3-4
├── LAYER1_INTEGRATION_GUIDE.md
├── LAYER1_QUICK_REFERENCE.md
└── data/
    └── chroma_index/          # ChromaDB corpus (4,201 chunks)
```

### Target (After Reorganization)
```
.
├── backend/
│   ├── ingest.py              # STAGE 1
│   ├── detect_lexical.py      # Layer 1
│   ├── detect_semantic.py     # Layer 2
│   ├── detect_intrinsic.py    # Layer 3
│   ├── stage2_detect.py       # Coordinator (run layers in parallel)
│   ├── score.py               # STAGE 3
│   ├── explain.py             # STAGE 4
│   └── api.py                 # Flask server
├── scripts/
│   ├── build_corpus.py
│   ├── test_ingest_real_pdf.py
│   ├── test_pipeline_stages.py
│   └── test_full_pipeline.py  # STAGE 1-4 end-to-end
```

---

## Essential Commands

### Setup
```bash
# Activate venv
source venv/bin/activate

# Install deps (Stage 1-2 only)
pip install -r requirements.txt

# Full deps (if adding SBERT/ChromaDB layers)
pip install sentence-transformers chromadb pdfplumber scikit-learn
```

### Build Corpus (Required Before Testing Detection)
```bash
python scripts/build_corpus.py
# Output: ChromaDB index with 4,201+ chunks from 3 domains
```

### Test Individual Stages
```bash
# Test Stage 1: Ingestion
python scripts/test_ingest_real_pdf.py papers/paper.pdf

# Test Stage 2: Detection layers
python scripts/test_pipeline_stages.py

# Test Layer 1 (TF-IDF) in isolation
python backend/detect_lexical.py

# Test Layer 2 (SBERT) in isolation
python backend/detect_semantic.py
```

### Test End-to-End (Once Stage 2 Coordinator Created)
```bash
python backend/api.py
# Then POST PDF to http://localhost:5000/analyse
```

---

## Module Responsibilities

### `backend/ingest.py` (STAGE 1: Extraction)
**Responsibility**: Extract text from PDFs, detect section boundaries, create chunks

**Key Functions**:
- `extract_text(pdf_path: str) -> Dict[str, str]` — Returns {section_name: text}
- `chunk_text(text: str, size=200, overlap=50) -> List[str]` — Sliding window chunker

**Key Patterns**:
- Handles PDF encoding artifacts (ligatures, (cid:X) markers)
- Detects 16+ section headers (abstract, introduction, methodology, etc.)
- Filters 18+ noise patterns (page numbers, ISSN, URLs, etc.)
- Returns section-level text for Layer 3 (intrinsic analysis)

**Quality Checks**: Use `test_ingest_real_pdf.py` to detect extraction quality issues

---

### `backend/detect_lexical.py` (LAYER 1: TF-IDF)
**Responsibility**: Detect verbatim and near-exact matches

**Key Function**:
- `detect_lexical(chunks, threshold=0.55, corpus_path="data/chroma_index") -> List[Dict]`

**Approach**:
1. Connect to ChromaDB corpus (4,201 chunks)
2. Build TF-IDF vectorizer on corpus + manuscript chunks
3. Compare each manuscript chunk to corpus using cosine similarity
4. Return matches ≥0.55 threshold with metadata

**Why Compute On-The-Fly**:
- TF-IDF is fast (~50-100ms) compared to neural models
- No need to store separate vectors
- Corpus already in ChromaDB

---

### `backend/detect_semantic.py` (LAYER 2: SBERT)
**Responsibility**: Detect paraphrased text and semantic reuse

**Key Function**:
- `detect_semantic(chunks, threshold=0.75, top_k=3) -> List[Dict]`

**Approach**:
1. Load SBERT model: `all-MiniLM-L6-v2` (80MB, local)
2. Encode manuscript chunks using SBERT
3. Query ChromaDB nearest-neighbor search (embeddings pre-computed)
4. Return matches ≥0.75 threshold with metadata

**Status**: Template exists, needs flag construction debugging

---

### `detect_intrinsic.py` (LAYER 3: Style Analysis — ⚠️ Needs Move to backend/)
**Responsibility**: Detect writing style shifts (corpus-independent)

**Key Function**:
- `detect_intrinsic(sections: Dict[str, str], z_threshold=2.0) -> List[Dict]`

**Approach**:
1. Compute 3 stylometric features per section:
   - Flesch-Kincaid readability
   - Type-Token Ratio (TTR)
   - Average sentence length
2. Calculate z-scores relative to document-wide mean
3. Flag sections with |z| ≥ 2.0 on any feature

**Advantage**: Works without corpus (catches plagiarism from any source)

---

### `score.py` (STAGE 3: Aggregation — ⚠️ Needs Move to backend/)
**Responsibility**: Deduplicate overlapping flags, assign risk tiers, compute composite score

**Key Function**:
- `aggregate_flags(l1_flags, l2_flags, l3_flags) -> Dict`

**Logic**:
1. Merge flags from all 3 layers
2. Deduplicate by chunk similarity (keep highest score)
3. Assign tier: HIGH (≥0.85), MEDIUM (0.70-0.84), LOW (0.55-0.69)
4. Composite score: weighted average of top-5 flags

---

### `explain.py` (STAGE 4: LLM Explanations — ⚠️ Needs Move to backend/)
**Responsibility**: Generate editorial explanations for HIGH/MEDIUM flags

**Key Function**:
- `explain_flag(flag: Dict) -> str`

**Anti-Hallucination Design**:
- LLM receives **both passages** as grounded context
- LLM **never decides** whether a match exists (cosine similarity did)
- Temperature set to 0 for deterministic output
- Returns 1-2 sentence editorial explanation

---

### `api.py` (Flask Server — ⚠️ Needs Move to backend/ + Refactor)
**Responsibility**: HTTP endpoint orchestrating full pipeline

**Current Endpoint**: `POST /analyse`
```python
# Expected input: PDF file
# Expected output: JSON with flags, risk score, tier breakdown
```

**TODO**: Refactor to use Stage 2 coordinator (parallel Layer 1-3 execution)

---

## Corpus & Data

### ChromaDB Index Location
```
data/chroma_index/
├── chroma.sqlite3
└── [embeddings UUID]/
```

### Corpus Stats
- **Total chunks**: 4,201
- **Domains**: 3 (natural language processing, climate change, biology)
- **Metadata per chunk**: title, DOI, domain, year
- **Model**: all-MiniLM-L6-v2 SBERT embeddings

### Building Corpus
```bash
python scripts/build_corpus.py
```
Uses CORE API (core.ac.uk) to fetch open-access papers. Requires `CORE_API_KEY` in `.env`

---

## Common Patterns & Conventions

### Flag Schema (Standard Across All Layers)
```python
{
    'chunk': str,              # Original manuscript text (~200 words)
    'score': float,            # Cosine similarity 0.00-1.00
    'matched': str,            # Matched corpus chunk
    'meta': {                  # Corpus metadata
        'title': str,
        'doi': str,
        'domain': str,         # Query domain used in corpus build
        'year': str
    },
    'layer': int,              # 1 (TF-IDF), 2 (SBERT), or 3 (Style)
    'type': str                # e.g., 'Verbatim/Near-Exact', 'Paraphrase/Semantic'
}
```

### Error Handling Pattern
All detection layers should:
1. Check if ChromaDB index exists → `FileNotFoundError` with setup instructions
2. Validate corpus is not empty → `RuntimeError`
3. Handle API failures gracefully

### Testing Pattern
Each detection layer has `if __name__ == "__main__"` test block:
```bash
python backend/detect_lexical.py    # Test Layer 1
python backend/detect_semantic.py   # Test Layer 2
```

---

## Common Pitfalls & How to Avoid Them

### ❌ Don't Pre-Compute TF-IDF Vectors
- Increases storage overhead
- TF-IDF changes with corpus size
- Harder to maintain two indices

✅ **Do**: Compute TF-IDF on-demand from ChromaDB documents

### ❌ Don't Ignore Metadata in Flags
- Makes flags unusable in UI
- Editorial explanation can't reference source

✅ **Do**: Always include title, DOI, domain, year from ChromaDB metadata

### ❌ Don't Use Hardcoded Dummy Corpus
- Breaks Layer 1 testing
- Prevents real plagiarism detection

✅ **Do**: Always connect to real ChromaDB via `chromadb.PersistentClient(path='data/chroma_index')`

### ❌ Don't Mix Threshold Values
- Causes inconsistent risk tier assignment

✅ **Do**: Use spec-defined thresholds: 0.55 (LOW), 0.70 (MEDIUM), 0.85 (HIGH)

### ❌ Don't Let LLM Decide Matches
- Hallucination risk
- "Explainability" becomes unreliable

✅ **Do**: LLM only explains matches already found by cosine similarity (temperature=0)

---

## Development Workflow

### Before Making Changes
1. Read [LAYER1_INTEGRATION_GUIDE.md](LAYER1_INTEGRATION_GUIDE.md) (architecture decisions)
2. Understand current module organization and stage dependencies
3. Check if ChromaDB corpus exists: `ls data/chroma_index/chroma.sqlite3`

### When Adding New Features
1. **Preserve flag schema** — All layers must return same structure
2. **Add to correct stage** — Don't mix stages (ingest → detect → score → explain)
3. **Test in isolation first** — Run individual detection layer tests
4. **Update repo structure** — Follow recommended organization

### When Debugging
1. Test ingestion first: `python scripts/test_ingest_real_pdf.py papers/paper.pdf`
2. Check corpus exists: `python backend/detect_lexical.py`
3. Review flag output: Ensure metadata is populated
4. Verify thresholds: Apply 0.55/0.70/0.85 logic

---

## Quick Reference Links

- **Integration Guide**: [LAYER1_INTEGRATION_GUIDE.md](LAYER1_INTEGRATION_GUIDE.md)
- **Quick Reference**: [LAYER1_QUICK_REFERENCE.md](LAYER1_QUICK_REFERENCE.md)
- **Technical Spec**: Embedded in repo root (see git history or hackathon materials)
- **Corpus Build**: [scripts/build_corpus.py](scripts/build_corpus.py)
- **Test Commands**: Documented in individual `test_*.py` files

---

## Next Steps for Agents

1. **Reorganize repository** — Move modules to backend/ as shown in "Target" structure
2. **Create Stage 2 Coordinator** — Parallelize Layer 1-3 execution
3. **Fix Layer 2 (SBERT)** — Debug flag construction in detect_semantic.py
4. **Refactor api.py** — Use new coordinator for orchestration
5. **Create end-to-end test** — Verify full STAGE 1-4 pipeline
6. **Create UI** — Streamlit dashboard (not yet started)

---

## Quick Checks for New Work

Before implementing anything, verify:
- [ ] All detection layers return **standard flag schema**
- [ ] Thresholds follow **0.55/0.70/0.85 spec**
- [ ] Metadata includes **title, DOI, domain, year**
- [ ] Code connects to **real ChromaDB** (not dummy corpus)
- [ ] LLM only **explains**, never **decides** matches
- [ ] Tests exist and pass: `python backend/detect_lexical.py` ✅
