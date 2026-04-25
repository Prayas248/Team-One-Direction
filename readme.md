# PRSE Codebase Guide for AI Agents

## Project Overview

**Plagiarism Risk Signal Engine** — An AI-powered editorial tool for detecting plagiarism in academic manuscripts. Unlike verdict tools, PRSE surfaces risk signals with full explainability to support human editorial judgment.

---

---
# PRSE Quick Start Guide

## 30-Second Setup

### Terminal 1 - Start Flask API
```bash
cd ~/Documents/Team-One-Direction
source venv/bin/activate
python api.py
```

### Terminal 2 - Start Streamlit UI  
```bash
cd ~/Documents/Team-One-Direction
source venv/bin/activate
streamlit run frontend/app.py
```

### Browser
Open: **http://localhost:8501**

---

## How to Use

1. **Upload** a PDF manuscript
2. Click **"🔍 Analyze for Plagiarism"**
3. Watch progress through 4 stages:
   - Extraction (10%)
   - Parallel detection (40%)
   - Risk scoring (70%)
   - Explanations (100%)
4. View results:
   - **Risk score** (0-100)
   - **Flagged passages** with explanations
   - **Detection breakdown** (L1, L2, L3)

---

## What You Get

### Risk Score Interpretation
| Score | Status | Meaning |
|-------|--------|---------|
| 80-100 | 🔴 HIGH | Substantial plagiarism signals |
| 50-79 | 🟡 MEDIUM | Concerning similarities |
| 25-49 | 🟢 LOW | Minor similarities |
| 0-24 | ✅ CLEAN | No signals detected |

### Each Flag Shows
- **Source passage** (what submitted text matched)
- **Matched passage** (where it came from)
- **Detection layer** (L1: lexical, L2: semantic, L3: style)
- **Score** (0.0-1.0)
- **Explanation** (why it's flagged)

---

## Test Files

Sample PDFs to test:
- `papers/paper.pdf` - Expected: ~40/100
- `papers/1706.03762v7-2.pdf` - Transformer paper
- `papers/23092015_Double_Column_Research_Paper_Format.pdf`

---

## Architecture

```
Your PDF
   ↓
Streamlit (frontend/app.py)
   ↓ HTTP POST
Flask API (api.py)
   ↓
Stage 1: Extract text + chunk
   ↓
Stage 2: Run 3 layers in parallel
   ├─ Layer 1: TF-IDF (lexical)
   ├─ Layer 2: SBERT (semantic)
   └─ Layer 3: Stylometry (intrinsic)
   ↓
Stage 3: Aggregate flags → compute score
   ↓
Stage 4: LLM explanations (with fallback)
   ↓
Results JSON
   ↓
Streamlit displays results
```

---

## Performance

- **Small PDF (5 pages)**: ~0.5 seconds
- **Medium PDF (20 pages)**: ~2 seconds
- **Large PDF (50 pages)**: ~5 seconds

Bottleneck: LLM explanation generation (2-5s per HIGH/MEDIUM flag)

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

### ✅ Threshold Consistency (All Layers)
- **LOW**: 0.55–0.69 (flagged, needs review)
- **MEDIUM**: 0.70–0.84 (significant similarity)
- **HIGH**: ≥0.85 (very likely plagiarism signal)
