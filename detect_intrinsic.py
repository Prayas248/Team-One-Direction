import textstat
import numpy as np
from typing import Dict, List

def style_features(text: str) -> dict:
    """Compute three stylometric features for a text section."""
    sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 10]
    words = text.split()
    ttr = len(set(words)) / len(words) if words else 0  # Type-Token Ratio
    avg_sent_len = np.mean([len(s.split()) for s in sentences]) if sentences else 0
    readability = textstat.flesch_reading_ease(text)
    return {'ttr': ttr, 'avg_sent_len': avg_sent_len, 'readability': readability}

def detect_intrinsic(sections: Dict[str, str], z_threshold=2.0) -> List[dict]:
    """Flag sections whose style deviates significantly from the doc baseline."""
    feats = {sec: style_features(txt) for sec, txt in sections.items() if len(txt.split()) > 50}
    if len(feats) < 3:
        return []  # Need enough sections for meaningful baseline
    flags = []
    for feat_name in ['ttr', 'avg_sent_len', 'readability']:
        values = np.array([feats[s][feat_name] for s in feats])
        mean, std = values.mean(), values.std()
        if std < 1e-6:
            continue
        zscores = (values - mean) / std
        for (section, _), z in zip(feats.items(), zscores):
            if abs(z) >= z_threshold:
                flags.append({
                    'section': section,
                    'feature': feat_name,
                    'z_score': float(z),
                    'score': min(abs(z) / 4, 1.0),  # Normalize to 0-1
                    'layer': 3,
                    'type': 'Style Anomaly (Intrinsic)',
                    'chunk': sections[section][:500]  # Excerpt
                })
    return flags

# Add this block at the end of the script
if __name__ == "__main__":
    print("Script executed successfully!")