# config/settings.py

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings:
    """Configuration for Stage 3 & 4"""
    
    # Base directory
    BASE_DIR = Path(__file__).parent.parent
    
    # API Keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    
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
        """Validate that required settings are present"""
        if not cls.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not found in .env file")
        print("✓ Settings validated successfully")
        return True

# Create settings instance
settings = Settings()
