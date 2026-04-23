# src/score.py
# Stage 3: Risk Aggregation & Scoring
# T&F Specification - Starter Hackathon 2026

from typing import List, Dict
from config.settings import settings

def assign_tier(score: float) -> str:
    """
    Assign risk tier based on similarity score
    
    T&F Specification:
    - HIGH: ≥0.85
    - MEDIUM: 0.70-0.84
    - LOW: 0.55-0.69
    
    Args:
        score: Similarity score (0.0-1.0)
    
    Returns:
        Risk tier: 'HIGH', 'MEDIUM', or 'LOW'
    """
    if score >= 0.85:
        return 'HIGH'
    if score >= 0.70:
        return 'MEDIUM'
    if score >= 0.55:
        return 'LOW'
    return 'NONE'  # Below threshold

def aggregate_flags(l1: List[Dict], l2: List[Dict], l3: List[Dict]) -> Dict:
    """
    Combine all flags from 3 layers, deduplicate, and compute composite score
    
    Deduplication Strategy:
    - If same chunk flagged by multiple layers, keep the one with highest score
    - Uses first 80 characters of chunk as deduplication key
    
    Composite Score Calculation:
    - Takes top 5 flags by score
    - Applies tier-based weights: HIGH=1.0, MEDIUM=0.6, LOW=0.3
    - Normalizes to 0-100 scale
    
    Args:
        l1: Layer 1 flags (TF-IDF lexical matches)
        l2: Layer 2 flags (SBERT semantic matches)
        l3: Layer 3 flags (Intrinsic style anomalies)
    
    Returns:
        Dictionary containing:
        - flags: Deduplicated list of all flags with tier assignments
        - risk_score: Composite manuscript risk score (0-100)
        - tier_counts: Count of flags per tier
        - layer_breakdown: Count of flags per layer
    """
    
    print("\n📊 Stage 3: Risk Aggregation & Scoring")
    print("=" * 60)
    
    # Combine all flags
    all_flags = l1 + l2 + l3
    
    print(f"Total flags before deduplication: {len(all_flags)}")
    print(f"  - Layer 1 (Lexical): {len(l1)}")
    print(f"  - Layer 2 (Semantic): {len(l2)}")
    print(f"  - Layer 3 (Intrinsic): {len(l3)}")
    
    if not all_flags:
        print("✓ No flags detected - document appears clean")
        return {
            'flags': [],
            'risk_score': 0,
            'tier_counts': {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0, 'NONE': 0},
            'layer_breakdown': {
                'layer1_lexical': 0,
                'layer2_semantic': 0,
                'layer3_intrinsic': 0
            }
        }
    
    # Deduplicate: if same chunk flagged by L1 and L2, keep higher score
    seen = {}
    deduped = []
    
    # Sort by score descending to keep highest score per chunk
    for flag in sorted(all_flags, key=lambda x: -x['score']):
        # Create deduplication key
        # For Layer 3 (intrinsic), use section name
        # For Layer 1 & 2, use first 80 chars of chunk
        if flag.get('layer') == 3:
            key = f"layer3_{flag.get('section', 'unknown')}"
        else:
            chunk_text = flag.get('chunk', '')
            key = chunk_text[:80] if chunk_text else f"flag_{len(deduped)}"
        
        # Check if we've seen this chunk before
        if key not in seen:
            seen[key] = True
            
            # Assign tier based on score
            tier = assign_tier(flag['score'])
            flag['tier'] = tier
            
            # Add color and emoji for UI
            tier_info = settings.get_risk_tier(flag['score'])
            flag['tier_emoji'] = tier_info[1]
            flag['tier_color'] = tier_info[2]
            
            deduped.append(flag)
    
    print(f"\nFlags after deduplication: {len(deduped)}")
    
    # Calculate tier counts
    tier_counts = {
        'HIGH': sum(1 for f in deduped if f['tier'] == 'HIGH'),
        'MEDIUM': sum(1 for f in deduped if f['tier'] == 'MEDIUM'),
        'LOW': sum(1 for f in deduped if f['tier'] == 'LOW'),
        'NONE': sum(1 for f in deduped if f['tier'] == 'NONE')
    }
    
    print(f"\nTier Distribution:")
    print(f"  🔴 HIGH:   {tier_counts['HIGH']}")
    print(f"  🟡 MEDIUM: {tier_counts['MEDIUM']}")
    print(f"  🟢 LOW:    {tier_counts['LOW']}")
    
    # Compute composite score: weighted sum of top-5 flags
    weights = {
        'HIGH': 1.0,
        'MEDIUM': 0.6,
        'LOW': 0.3,
        'NONE': 0.0
    }
    
    # Take top 5 flags by score
    top5 = sorted(deduped, key=lambda x: -x['score'])[:5]
    
    if not top5:
        composite_score = 0
    else:
        # Calculate weighted average
        weighted_sum = sum(
            f['score'] * weights[f['tier']] * 100
            for f in top5
        )
        composite_score = min(100, int(weighted_sum / len(top5)))
    
    print(f"\nComposite Risk Score: {composite_score}/100")
    print(f"  (Based on top {len(top5)} flags)")
    
    # Layer breakdown
    layer_breakdown = {
        'layer1_lexical': len(l1),
        'layer2_semantic': len(l2),
        'layer3_intrinsic': len(l3)
    }
    
    print("\n" + "=" * 60)
    print("✓ Risk aggregation complete")
    
    return {
        'flags': deduped,
        'risk_score': composite_score,
        'tier_counts': tier_counts,
        'layer_breakdown': layer_breakdown,
        'top5_flags': top5  # For detailed reporting
    }


def get_flag_summary(flags: List[Dict]) -> Dict:
    """
    Generate summary statistics for flags
    
    Args:
        flags: List of flag dictionaries
    
    Returns:
        Summary statistics
    """
    if not flags:
        return {
            'total': 0,
            'avg_score': 0.0,
            'max_score': 0.0,
            'min_score': 0.0
        }
    
    scores = [f['score'] for f in flags]
    
    return {
        'total': len(flags),
        'avg_score': sum(scores) / len(scores),
        'max_score': max(scores),
        'min_score': min(scores)
    }


def rank_flags_by_severity(flags: List[Dict]) -> List[Dict]:
    """
    Rank flags by severity for prioritized review
    
    Ranking criteria:
    1. Tier (HIGH > MEDIUM > LOW)
    2. Score (within same tier)
    3. Layer (Layer 1 > Layer 2 > Layer 3 for same score)
    
    Args:
        flags: List of flag dictionaries
    
    Returns:
        Sorted list of flags
    """
    tier_priority = {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1, 'NONE': 0}
    
    return sorted(
        flags,
        key=lambda f: (
            tier_priority.get(f.get('tier', 'NONE'), 0),  # Tier priority
            f.get('score', 0),                              # Score
            -f.get('layer', 999)                            # Layer (lower is higher priority)
        ),
        reverse=True
    )


# Example usage and testing
if __name__ == "__main__":
    print("Testing Risk Aggregation & Scoring Module")
    print("=" * 60)
    
    # Mock flags from 3 layers
    mock_l1 = [
        {
            'chunk': 'This is a test passage that appears verbatim in another source.',
            'score': 0.92,
            'matched': 'This is a test passage that appears verbatim in another source.',
            'meta': {'title': 'Test Paper 1', 'doi': '10.1234/test1'},
            'layer': 1,
            'type': 'Verbatim/Near-Exact',
            'method': 'TF-IDF + Cosine'
        }
    ]
    
    mock_l2 = [
        {
            'chunk': 'Machine learning algorithms can be used for classification tasks.',
            'score': 0.78,
            'matched': 'Classification tasks are commonly solved using machine learning methods.',
            'meta': {'title': 'Test Paper 2', 'doi': '10.1234/test2'},
            'layer': 2,
            'type': 'Paraphrase/Semantic',
            'method': 'SBERT + ChromaDB'
        },
        {
            'chunk': 'This is a test passage that appears verbatim in another source.',  # Duplicate
            'score': 0.88,  # Lower than L1
            'matched': 'Similar text',
            'meta': {'title': 'Test Paper 3', 'doi': '10.1234/test3'},
            'layer': 2,
            'type': 'Paraphrase/Semantic',
            'method': 'SBERT + ChromaDB'
        }
    ]
    
    mock_l3 = [
        {
            'section': 'methodology',
            'feature': 'readability',
            'z_score': 2.5,
            'score': 0.625,
            'layer': 3,
            'type': 'Style Anomaly (Intrinsic)',
            'method': 'Stylometric Analysis',
            'chunk': 'The methodology section with different writing style...'
        }
    ]
    
    # Run aggregation
    result = aggregate_flags(mock_l1, mock_l2, mock_l3)
    
    print("\n" + "=" * 60)
    print("RESULTS:")
    print("=" * 60)
    print(f"Risk Score: {result['risk_score']}/100")
    print(f"Total Flags: {len(result['flags'])}")
    print(f"Tier Counts: {result['tier_counts']}")
    print(f"Layer Breakdown: {result['layer_breakdown']}")
    
    print("\nTop Flags:")
    for i, flag in enumerate(result['flags'][:3], 1):
        print(f"\n{i}. {flag['tier']} - Score: {flag['score']:.3f}")
        print(f"   Type: {flag['type']}")
        print(f"   Layer: {flag['layer']}")