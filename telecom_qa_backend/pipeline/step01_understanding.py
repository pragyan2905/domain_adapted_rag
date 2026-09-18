import json
from core.schemas import QueryContext, QueryUnderstanding, AnswerabilityType
from core.interfaces import Generator

def execute(context: QueryContext, generator: Generator) -> QueryContext:
    """
    Step 1: Query Understanding
    Uses the LLM to normalize telecom acronyms and detect intent before retrieval.
    """
    prompt = f"""
You are a 3GPP telecom standards query understanding agent.
Analyze the following query: "{context.original_query}"

Return a JSON object with the following exact keys:
1. "normalized_query": The query with telecom acronyms expanded (e.g., UE to User Equipment, RRC to Radio Resource Control).
2. "expanded_acronyms": A dictionary mapping the original acronym to the expansion.
3. "detected_intent": A short string describing what the user wants.
4. "answerability": Must be exactly one of: "factual_lookup", "multi_step_procedural", "out_of_domain".

Output strictly raw JSON. Do not use markdown formatting like ```json or add any other text.
"""
    
    # Empty context because we haven't retrieved anything yet
    result = generator.generate(prompt=prompt, context=[])
    
    # Clean possible markdown from the LLM output
    raw_content = result.content.strip()
    if raw_content.startswith("```json"):
        raw_content = raw_content[7:]
    if raw_content.endswith("```"):
        raw_content = raw_content[:-3]
        
    try:
        data = json.loads(raw_content.strip())
        context.understanding = QueryUnderstanding(**data)
    except Exception as e:
        # Fallback if LLM fails to output valid JSON or fails Pydantic validation
        context.understanding = QueryUnderstanding(
            normalized_query=context.original_query,
            detected_intent="unknown",
            answerability=AnswerabilityType.FACTUAL_LOOKUP
        )
        
    return context
