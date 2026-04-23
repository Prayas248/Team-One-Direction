#test_stages_3_4.py
# Test script for Stage 3 & 4
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from src.score import aggregate_flags, rank_flags_by_severity
from src.explain import explain_flag, explain_multiple_flags
def test_stage_3():
    """Test risk aggregation and scoring"""
    
    print("\n" + "="*80)
    print("TESTING STAGE 3: RISK AGGREGATION & SCORING")
    print("="*80)
    
    # Mock flags from 3 layers
    layer1_flags = [
        {
            'chunk': 'This is a verbatim copy of text from another source that appears word for word.',
            'score': 0.95,
            'matched': 'This is a verbatim copy of text from another source that appears word for word.',
            'meta': {'title': 'Source Paper A', 'doi': '10.1234/test1'},
            'layer': 1,
            'type': 'Verbatim/Near-Exact',
            'method': 'TF-IDF + Cosine'
        },
        {
            'chunk': 'Another exact match with very high similarity score indicating direct copying.',
            'score': 0.88,
            'matched': 'Another exact match with very high similarity score indicating direct copying.',
            'meta': {'title': 'Source Paper B', 'doi': '10.1234/test2'},
            'layer': 1,
            'type': 'Verbatim/Near-Exact',
            'method': 'TF-IDF + Cosine'
        }
    ]
    
    layer2_flags = [
        {
            'chunk': 'Machine learning algorithms can be used for classification tasks in various domains.',
            'score': 0.78,
            'matched': 'Classification tasks are commonly solved using machine learning methods across different fields.',
            'meta': {'title': 'Source Paper C', 'doi': '10.1234/test3'},
            'layer': 2,
            'type': 'Paraphrase/Semantic',
            'method': 'SBERT + ChromaDB'
        },
        {
            'chunk': 'This is a verbatim copy of text from another source that appears word for word.',  # Duplicate
            'score': 0.85,  # Lower than Layer 1 version
            'matched': 'Similar text',
            'meta': {'title': 'Source Paper D', 'doi': '10.1234/test4'},
            'layer': 2,
            'type': 'Paraphrase/Semantic',
            'method': 'SBERT + ChromaDB'
        },
        {
            'chunk': 'Deep learning has revolutionized computer vision applications.',
            'score': 0.72,
            'matched': 'Computer vision has been transformed by deep learning techniques.',
            'meta': {'title': 'Source Paper E', 'doi': '10.1234/test5'},
            'layer': 2,
            'type': 'Paraphrase/Semantic',
            'method': 'SBERT + ChromaDB'
        }
    ]
    
    layer3_flags = [
        {
            'section': 'methodology',
            'feature': 'readability',
            'z_score': 2.5,
            'score': 0.625,
            'layer': 3,
            'type': 'Style Anomaly (Intrinsic)',
            'method': 'Stylometric Analysis',
            'chunk': 'The methodology section exhibits significantly different readability characteristics...'
        },
        {
            'section': 'results',
            'feature': 'avg_sent_len',
            'z_score': -2.2,
            'score': 0.55,
            'layer': 3,
            'type': 'Style Anomaly (Intrinsic)',
            'method': 'Stylometric Analysis',
            'chunk': 'The results section shows unusual sentence length patterns...'
        }
    ]
    
    # Run aggregation
    result = aggregate_flags(layer1_flags, layer2_flags, layer3_flags)
    
    # Display results
    print("\n" + "="*80)
    print("AGGREGATION RESULTS:")
    print("="*80)
    print(f"\n📊 Composite Risk Score: {result['risk_score']}/100")
    print(f"\n📋 Total Flags: {len(result['flags'])} (after deduplication)")
    print(f"   Original: {len(layer1_flags) + len(layer2_flags) + len(layer3_flags)}")
    
    print(f"\n🎯 Tier Distribution:")
    for tier, count in result['tier_counts'].items():
        emoji = {'HIGH': '🔴', 'MEDIUM': '🟡', 'LOW': '🟢', 'NONE': '⚪'}.get(tier, '⚪')
        print(f"   {emoji} {tier}: {count}")
    
    print(f"\n🔬 Layer Breakdown:")
    print(f"   Layer 1 (Lexical): {result['layer_breakdown']['layer1_lexical']}")
    print(f"   Layer 2 (Semantic): {result['layer_breakdown']['layer2_semantic']}")
    print(f"   Layer 3 (Intrinsic): {result['layer_breakdown']['layer3_intrinsic']}")
    
    print(f"\n🏆 Top 5 Flags:")
    for i, flag in enumerate(result['top5_flags'], 1):
        print(f"   {i}. {flag['tier']} - Score: {flag['score']:.3f} - {flag['type']}")
    
    return result
def test_stage_4(flags):
    """Test LLM explainability"""
    
    print("\n" + "="*80)
    print("TESTING STAGE 4: LLM EXPLAINABILITY")
    print("="*80)
    
    # Generate explanations
    flags_with_explanations = explain_multiple_flags(flags, max_explanations=5)
    
    # Display explanations
    print("\n" + "="*80)
    print("GENERATED EXPLANATIONS:")
    print("="*80)
    
    for i, flag in enumerate(flags_with_explanations[:5], 1):
        if flag.get('explanation'):
            print(f"\n{i}. {flag['tier']} - {flag['type']}")
            print(f"   Score: {flag['score']:.3f}")
            print(f"   Explanation: {flag['explanation']}")
    
    return flags_with_explanations
if __name__ == "__main__":
    # Test Stage 3
    result = test_stage_3()
    
    # Test Stage 4
    flags_with_explanations = test_stage_4(result['flags'])
    
    print("\n" + "="*80)
    print("✅ ALL TESTS COMPLETE")
    print("="*80)
