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
