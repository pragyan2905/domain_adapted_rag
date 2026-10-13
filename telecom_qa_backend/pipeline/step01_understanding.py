from core.schemas import QueryContext, QueryUnderstanding, AnswerabilityType
from core.interfaces import Generator
from core.tracer import trace_step
from core.structured_llm import generate_structured

def execute(context: QueryContext, generator: Generator) -> QueryContext:
    """
    Step 1: Query Understanding & Decomposition
    Expands acronyms, determines answerability, and decomposes multi-hop queries.
    """
    with trace_step(context, "Step 01: Understanding", input_count=1) as t:
        prompt = f"""
You are a 3GPP standards query understanding agent.
Analyze the following user query: "{context.original_query}"

Task:
1. Normalize the query by expanding technical acronyms (e.g., UE -> User Equipment, AMF -> Access and Mobility Management Function).
2. Provide a dictionary of expanded acronyms.
3. Detect intent (e.g., "compare handover procedures", "identify timer values", "describe architecture").
4. Classify answerability as exactly one of: "factual_lookup", "multi_step_procedural", "out_of_domain".
5. If the query is multi-step, comparative, or mentions multiple concepts/releases, decompose it into 2-3 focused atomic sub-queries for independent retrieval. If simple, keep sub_queries as a single element list containing the normalized query.

Return strictly a valid JSON object matching the schema:
{{
  "normalized_query": "string",
  "expanded_acronyms": {{"ACRONYM": "Expansion"}},
  "detected_intent": "string",
  "answerability": "factual_lookup" | "multi_step_procedural" | "out_of_domain",
  "sub_queries": ["sub query 1", "sub query 2"]
}}
"""
        try:
            understanding = generate_structured(
                generator=generator,
                prompt=prompt,
                response_model=QueryUnderstanding,
                max_retries=1
            )
            context.understanding = understanding
            t["extra"] = {
                "answerability": understanding.answerability.value,
                "acronyms_expanded": len(understanding.expanded_acronyms),
                "sub_queries_count": len(understanding.sub_queries),
            }
        except Exception as e:
            # Fallback on parse failure
            context.understanding = QueryUnderstanding(
                normalized_query=context.original_query,
                detected_intent="technical_lookup",
                answerability=AnswerabilityType.FACTUAL_LOOKUP,
                sub_queries=[context.original_query]
            )
            t["extra"] = {"fallback": True, "error": str(e)}

        t["output_count"] = 1

    return context
