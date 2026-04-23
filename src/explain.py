# src/explain.py
# Stage 4: LLM Explainability
# T&F Specification - Starter Hackathon 2026

"""
ANTI-HALLUCINATION CORE:
The LLM receives both passages as grounded context and writes an editorial explanation.
It NEVER decides whether a match exists — that was decided by cosine similarity.

Key Design Principles:
1. LLM only explains EXISTING matches (never detects)
2. Temperature = 0 (deterministic, reproducible)
3. Both passages provided as context (grounded generation)
4. Editorial language only (no plagiarism verdicts)
"""

import os
from typing import Dict
from config.settings import settings

# Use Groq only
try:
    from groq import Groq
    CLIENT = Groq(api_key=settings.GROQ_API_KEY)
    print("✓ Using Groq API for explanations")
except Exception as e:
    print(f"⚠️  Error: Groq API not available - {e}")
    print(f"⚠️  Make sure to install: pip install groq")
    print(f"⚠️  And add GROQ_API_KEY to .env file")
    CLIENT = None

# System prompt - defines LLM's role and constraints
SYSTEM_PROMPT = """You are a senior editorial analyst at an academic publisher.

You are given two text passages and a similarity score computed by a mathematical embedding model.

Your task is to explain, in 1-2 sentences, WHY these passages are semantically similar — focusing on shared ideas, phrasing patterns, or structural similarities.

CRITICAL RULES:
- Do NOT conclude that plagiarism has occurred
- Do NOT make accusations or judgments
- Use editorial language: 'risk signal', 'flagged for review', 'warrants attention', 'suggests similarity'
- Be factual and concise
- Focus on WHAT is similar, not WHETHER it's plagiarism
- You are explaining an EXISTING match found by mathematical analysis, not detecting one

Remember: You are a tool to help human editors understand similarity patterns, not to make final decisions."""


def explain_flag(flag: Dict) -> str:
    """
    Generate editorial explanation for a detected flag
    
    This is the anti-hallucination core:
    - LLM receives both passages as input (grounded context)
    - LLM only explains WHY they're similar
    - LLM never decides IF they're similar (that's already determined)
    - Temperature = 0 for deterministic output
    
    Args:
        flag: Dictionary containing:
            - layer: Detection layer (1, 2, or 3)
            - chunk: Submitted passage
            - matched: Matched corpus passage (for layers 1 & 2)
            - score: Similarity score (0.0-1.0)
            - section: Section name (for layer 3)
            - feature: Style feature (for layer 3)
            - z_score: Statistical deviation (for layer 3)
    
    Returns:
        1-2 sentence editorial explanation
    """
    
    # Layer 3 (Intrinsic) - No matched source, style anomaly
    if flag.get('layer') == 3:
        section = flag.get('section', 'section')
        z_score = flag.get('z_score', 0)
        feature = flag.get('feature', 'feature')
        
        return (
            f"The '{section}' section shows a statistical style deviation "
            f"(z-score: {z_score:.2f}) on the {feature} metric, suggesting a "
            f"possible change in authorial voice that warrants editorial review. "
            f"This pattern may indicate text from a different source or author."
        )
    
    # Layers 1 & 2 - Have matched sources, need LLM explanation
    if not CLIENT:
        # Fallback if no LLM available
        return (
            f"High similarity detected (score: {flag.get('score', 0):.2f}) "
            f"between submitted text and source material. Manual review recommended."
        )
    
    # Prepare user message with both passages
    chunk = flag.get('chunk', '')[:400]  # Limit to 400 chars
    matched = flag.get('matched', '')[:400]
    score = flag.get('score', 0)
    method = flag.get('method', 'embedding model')
    
    user_msg = f"""Submitted passage (under review):
"{chunk}"

Matched corpus passage:
"{matched}"

Similarity score: {score:.2f} (scale 0-1, computed by {method})

Task: Explain in 1-2 sentences WHY these passages are similar. Focus on shared ideas, phrasing patterns, or structural similarities. Use editorial language appropriate for academic publishing."""
    
    try:
        # Groq API call
        response = CLIENT.chat.completions.create(
            model='llama-3.3-70b-versatile',  # Best Groq model for this task
            messages=[
                {
                    'role': 'system',
                    'content': SYSTEM_PROMPT
                },
                {
                    'role': 'user',
                    'content': user_msg
                }
            ],
            temperature=0,  # CRITICAL: Deterministic output
            max_tokens=200,
            top_p=1,
            stream=False
        )
        
        explanation = response.choices[0].message.content.strip()
        
        # Validate explanation (basic sanity check)
        if not explanation or len(explanation) < 20:
            raise ValueError("Explanation too short")
        
        # Ensure no plagiarism verdict language
        forbidden_words = ['plagiarized', 'plagiarism', 'copied', 'stolen', 'cheated']
        explanation_lower = explanation.lower()
        
        if any(word in explanation_lower for word in forbidden_words):
            print(f"⚠️  Warning: LLM used forbidden language, using fallback")
            return generate_fallback_explanation(flag)
        
        return explanation
    
    except Exception as e:
        print(f"⚠️  LLM explanation failed: {e}")
        return generate_fallback_explanation(flag)


def generate_fallback_explanation(flag: Dict) -> str:
    """
    Generate fallback explanation if LLM fails
    
    Args:
        flag: Flag dictionary
    
    Returns:
        Generic but accurate explanation
    """
    score = flag.get('score', 0)
    layer = flag.get('layer', 0)
    flag_type = flag.get('type', 'similarity')
    
    if layer == 1:
        return (
            f"This passage shows {score:.0%} lexical similarity to the matched source, "
            f"indicating substantial overlap in word choice and phrasing that warrants review."
        )
    elif layer == 2:
        return (
            f"This passage shows {score:.0%} semantic similarity to the matched source, "
            f"suggesting shared conceptual content or paraphrased ideas that merit editorial attention."
        )
    else:
        return (
            f"Similarity detected (score: {score:.2f}). "
            f"Manual review recommended to assess the nature and extent of overlap."
        )


def explain_multiple_flags(flags: list, max_explanations: int = 10) -> list:
    """
    Generate explanations for multiple flags efficiently
    
    Only explains HIGH and MEDIUM tier flags to conserve API calls
    
    Args:
        flags: List of flag dictionaries
        max_explanations: Maximum number of explanations to generate
    
    Returns:
        List of flags with 'explanation' field added
    """
    print("\n💡 Stage 4: Generating Explanations")
    print("=" * 60)
    
    # Filter to HIGH and MEDIUM only
    flags_to_explain = [
        f for f in flags
        if f.get('tier') in ('HIGH', 'MEDIUM')
    ][:max_explanations]
    
    print(f"Generating explanations for {len(flags_to_explain)} flags")
    print(f"  (HIGH and MEDIUM tier only)")
    
    explained_count = 0
    
    for i, flag in enumerate(flags_to_explain, 1):
        print(f"  {i}/{len(flags_to_explain)}: {flag.get('type', 'Unknown')} (score: {flag.get('score', 0):.3f})")
        
        try:
            explanation = explain_flag(flag)
            flag['explanation'] = explanation
            explained_count += 1
        except Exception as e:
            print(f"    ⚠️  Failed: {e}")
            flag['explanation'] = generate_fallback_explanation(flag)
    
    print(f"\n✓ Generated {explained_count} explanations")
    print("=" * 60)
    
    return flags


# Example usage and testing
if __name__ == "__main__":
    print("Testing LLM Explainability Module (Groq)")
    print("=" * 60)
    
    # Test Layer 1 flag (lexical)
    test_flag_l1 = {
        'chunk': 'Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed.',
        'matched': 'Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed.',
        'score': 0.95,
        'layer': 1,
        'type': 'Verbatim/Near-Exact',
        'method': 'TF-IDF + Cosine',
        'tier': 'HIGH'
    }
    
    print("\nTest 1: Layer 1 (Verbatim Match)")
    print("-" * 60)
    explanation = explain_flag(test_flag_l1)
    print(f"Explanation: {explanation}")
    
    # Test Layer 2 flag (semantic)
    test_flag_l2 = {
        'chunk': 'Neural networks are computational models inspired by biological neurons.',
        'matched': 'Artificial neural networks mimic the structure of biological brain cells.',
        'score': 0.82,
        'layer': 2,
        'type': 'Paraphrase/Semantic',
        'method': 'SBERT + ChromaDB',
        'tier': 'MEDIUM'
    }
    
    print("\nTest 2: Layer 2 (Semantic Match)")
    print("-" * 60)
    explanation = explain_flag(test_flag_l2)
    print(f"Explanation: {explanation}")
    
    # Test Layer 3 flag (intrinsic)
    test_flag_l3 = {
        'section': 'methodology',
        'feature': 'readability',
        'z_score': 2.8,
        'score': 0.70,
        'layer': 3,
        'type': 'Style Anomaly (Intrinsic)',
        'tier': 'MEDIUM'
    }
    
    print("\nTest 3: Layer 3 (Style Anomaly)")
    print("-" * 60)
    explanation = explain_flag(test_flag_l3)
    print(f"Explanation: {explanation}")
    
    print("\n" + "=" * 60)
    print("✓ Testing complete")