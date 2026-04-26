# PRSE — Plagiarism Risk Signal Engine

An AI-powered editorial tool for detecting plagiarism in academic manuscripts. PRSE surfaces risk signals with full explainability to support human editorial judgment.

---

## Quick Start

### Setup (30 seconds)

```bash
cd ~/Documents/Team-One-Direction
source venv/bin/activate
```

**Terminal 1 — Start API:**
```bash
python api.py
```

**Terminal 2 — Start UI:**
```bash
streamlit run frontend/app.py
```

Then open: http://localhost:8501

---

## How to Use

1. **Upload** a PDF manuscript
2. Click **"Analyze for Plagiarism"**
3. Monitor progress through 4 stages:
   - Extraction (extracting text & chunking)
   - Parallel detection (3 layers running simultaneously)
   - Risk scoring (aggregating flags)
   - Explanations (generating editorial notes)
4. View results:
   - Overall risk score (0-100)
   - Flagged passages with detailed explanations
   - Detection breakdown (L1, L2, L3 counts)

---

## Understanding Results

### Risk Score Scale
| Score | Risk Level | Interpretation |
|-------|------------|----------------|
| 85-100 | 🔴 HIGH | Substantial plagiarism signals |
| 70-84 | 🟡 MEDIUM | Concerning similarities |
| 55-69 | 🟢 LOW | Minor similarities |
| 0-54 | CLEAN | No significant signals |

### Each Flagged Passage Shows
- **Submitted Passage** — The text from your manuscript
- **Matched Source** — Where it was found (L1 & L2 only)
- **Detection Layer** — Which method caught it (Lexical/Semantic/Intrinsic)
- **Score** — Confidence level (0.0-1.0)
- **Explanation** — Why it was flagged (AI-generated editorial note)

**For Style Anomalies (Layer 3):**
- **Section** — Which part of the paper
- **Anomaly Type** — What changed (vocabulary, sentence length, readability)
- **Z-Score** — Statistical deviation from paper baseline
- **Context** — This section's value vs paper average

---

## System Architecture

### 4-Stage Pipeline

```
Your PDF
   ↓
Streamlit Frontend (frontend/app.py)
   ↓ HTTP POST /analyse
Flask API (api.py)
   ↓
Stage 1: Ingestion
  └─ Extract sections, create chunks (backend/ingest.py)
   ↓
Stage 2: Parallel Detection (ThreadPoolExecutor)
  ├─ Layer 1: Lexical — TF-IDF cosine similarity
  ├─ Layer 2: Semantic — SBERT embeddings + ChromaDB search
  └─ Layer 3: Intrinsic — Stylometry (z-score analysis)
   ↓
Stage 3: Risk Aggregation & Scoring (backend/score.py)
  └─ Deduplicate, tier assign, weighted composite score
   ↓
Stage 4: LLM Explanations (backend/explain.py)
  └─ Groq API (with fallback templates)
   ↓
Results JSON
   ↓
Streamlit displays results with visualizations
```

### Detection Layers

**Layer 1 — Lexical (TF-IDF)**
- Detects verbatim and near-exact matches
- Pre-fitted vectorizer on 76-paper corpus (10K vocabulary)
- Cosine similarity scoring
- Thresholds: 0.55 (LOW), 0.70 (MEDIUM), 0.85 (HIGH)

**Layer 2 — Semantic (SBERT)**
- Detects paraphrased and semantically similar content
- Model: `all-MiniLM-L6-v2` (384-dimensional embeddings)
- ChromaDB nearest-neighbor search (k=3)
- Threshold: 0.75 for flagging

**Layer 3 — Intrinsic (Stylometry)**
- Detects writing style anomalies (corpus-independent)
- Features: Type-Token Ratio (TTR), average sentence length, readability (Flesch-Kincaid)
- Z-score statistical analysis against section baseline
- Dynamic threshold: 2.0σ (adjusted for paper structure)
- Score mapping: z=2.0 → 0.55 (LOW tier), z=4.0 → 1.0 (HIGH)
