import re
from typing import Optional
from core.schemas import QueryContext, AnswerabilityType
from services.embed_bge import BGEEmbedder
from core.interfaces import Generator
from core.tracer import trace_step

def execute(
    context: QueryContext,
    embedder: BGEEmbedder,
    generator: Optional[Generator] = None,
    enable_hyde: bool = True
) -> QueryContext:
    """
    Step 2: Query Representation (with optional HyDE)
    Converts query into dense and sparse embeddings.
    If HyDE (Hypothetical Document Embeddings) is active and the query is factual,
    synthesizes a formal 3GPP-style clause paragraph to bridge style mismatch before embedding.
    """
    query = context.understanding.normalized_query if context.understanding else context.original_query

    with trace_step(context, "Step 02: Representation", input_count=1) as t:
        text_to_embed = query

        # Hypothetical Document Embedding (HyDE)
        if enable_hyde and generator and context.understanding and context.understanding.answerability == AnswerabilityType.FACTUAL_LOOKUP:
            try:
                hyde_prompt = f"""
Write a 2-3 sentence hypothetical excerpt from a 3GPP Technical Specification (TS) that precisely answers:
"{query}"
Use formal 3GPP specification language (e.g., "The UE shall...", "Upon reception of...", timer designations).
Output strictly the excerpt text without introductory conversational phrases.
"""
                hypo_result = generator.generate(prompt=hyde_prompt, context=[])
                hypo_doc = hypo_result.content.strip()
                if len(hypo_doc) > 30:
                    context.hypothetical_document = hypo_doc
                    # Combine original query with the hypothetical technical clause for robust matching
                    text_to_embed = f"{query} [SPEC EXCERPT]: {hypo_doc}"
            except Exception as e:
                # Soft fallback to raw query
                pass

        # 1. Embed dense + sparse simultaneously via BGE-M3
        dense_vecs, sparse_vecs = embedder.embed([text_to_embed])

        if dense_vecs and sparse_vecs:
            context.query_dense_vector = dense_vecs[0]
            context.query_sparse_vector = sparse_vecs[0]

        # 2. Extract structured filters (e.g., TS number or Release version)
        ts_pattern = re.compile(r"TS\s*(\d{2}\.\d{3})", re.IGNORECASE)
        rel_pattern = re.compile(r"(?:Rel|Release)[-\s]*(\d{2})", re.IGNORECASE)

        ts_matches = ts_pattern.findall(context.original_query)
        if ts_matches:
            context.target_ts_numbers = list(set(ts_matches))

        rel_matches = rel_pattern.findall(context.original_query)
        if rel_matches:
            context.target_releases = [f"Rel-{m}" for m in set(rel_matches)]

        t["output_count"] = 1
        t["extra"] = {
            "hyde_active": context.hypothetical_document is not None,
            "ts_filters": context.target_ts_numbers,
            "release_filters": context.target_releases,
            "dense_dim": len(context.query_dense_vector) if context.query_dense_vector else 0,
        }

    return context
