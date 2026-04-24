import anthropic

CLIENT = anthropic.Anthropic()  # Reads ANTHROPIC_API_KEY from environment

SYSTEM_PROMPT = """You are a senior editorial analyst at an academic publisher.
You are given two text passages and a similarity score computed by a mathematical embedding model. 
Your task is to explain, in 1-2 sentences, WHY these passages are semantically similar — focusing on shared ideas, phrasing patterns, or structural similarities. 
Do NOT conclude that plagiarism has occurred. Use editorial language: 'risk signal', 'flagged for review', 'warrants attention'. Be factual and concise."""

def explain_flag(flag: dict) -> str:
    """Generate an editorial explanation for a flagged passage."""
    if flag['layer'] == 3:  # Style anomaly — no matched source
        return (f"The '{flag['section']}' section shows a statistical "
                f"style deviation (z={flag['z_score']:.2f}) on the "
                f"{flag['feature']} metric, suggesting a possible change "
                f"in authorial voice that warrants editorial review.")
    
    user_msg = f"""Submitted passage (under review): "{flag['chunk'][:400]}"
Matched corpus passage: "{flag['matched'][:400]}"
Similarity score: {flag['score']:.2f} (scale 0-1, computed by embedding model)"""
    
    response = CLIENT.messages.create(
        model='claude-sonnet-4-20250514',
        max_tokens=200,
        temperature=0,  # Deterministic — no creative generation
        system=SYSTEM_PROMPT,
        messages=[{'role': 'user', 'content': user_msg}]
    )
    return response.content[0].text