# config/settings.py

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings:
    """Configuration for PRSE System (all 4 stages)"""
    
    # Base directory
    BASE_DIR = Path(__file__).parent.parent
    
    # API Keys (prioritize Groq, fallback to xAI)
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    XAI_API_KEY = os.getenv("XAI_API_KEY", "")  # Fallback if Groq unavailable
    
    # Database & Corpus Paths (relative to BASE_DIR)
    CHROMA_INDEX_PATH = os.getenv("CHROMA_INDEX_PATH", str(BASE_DIR / "data" / "chroma_index"))
    TFIDF_CORPUS_PATH = os.getenv("TFIDF_CORPUS_PATH", str(BASE_DIR / "data" / "tfidf_corpus"))
    
    # Thresholds
    LEXICAL_THRESHOLD = float(os.getenv("LEXICAL_THRESHOLD", "0.70"))
    SEMANTIC_THRESHOLD = float(os.getenv("SEMANTIC_THRESHOLD", "0.75"))
    INTRINSIC_Z_THRESHOLD = float(os.getenv("INTRINSIC_Z_THRESHOLD", "2.0"))
    
    # Risk Tiers
    HIGH_RISK_MIN = 0.85
    MEDIUM_RISK_MIN = 0.70
    LOW_RISK_MIN = 0.55
    
    @classmethod
    def get_risk_tier(cls, score: float) -> tuple:
        """Get risk tier information based on score"""
        if score >= cls.HIGH_RISK_MIN:
            return ("HIGH", "🔴", "#C0392B")
        elif score >= cls.MEDIUM_RISK_MIN:
            return ("MEDIUM", "🟡", "#B7791F")
        elif score >= cls.LOW_RISK_MIN:
            return ("LOW", "🟢", "#1A6B3C")
        else:
            return ("NONE", "⚪", "#95A5A6")
    
    @classmethod
    def validate(cls):
        """Validate that required settings are present and paths exist"""
        # Check API key
        if not cls.GROQ_API_KEY:
            if not cls.XAI_API_KEY:
                print("⚠️  No LLM API key set - explanations will use fallback")
            else:
                print("⚠️  GROQ_API_KEY not set - using xAI fallback (requires credits)")
        else:
            print("✓ GROQ_API_KEY configured")
        
        # Check paths
        chroma_path = Path(cls.CHROMA_INDEX_PATH)
        if not chroma_path.exists():
            raise FileNotFoundError(f"ChromaDB index not found at {cls.CHROMA_INDEX_PATH}")
        print(f"✓ ChromaDB index found: {cls.CHROMA_INDEX_PATH}")
        
        tfidf_path = Path(cls.TFIDF_CORPUS_PATH)
        if not tfidf_path.exists():
            raise FileNotFoundError(f"TF-IDF corpus not found at {cls.TFIDF_CORPUS_PATH}")
        print(f"✓ TF-IDF corpus found: {cls.TFIDF_CORPUS_PATH}")
        
        return True

# Create settings instance
settings = Settings()
