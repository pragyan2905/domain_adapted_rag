import re
from core.schemas import QueryContext
from core.tracer import trace_step

def execute(context: QueryContext) -> QueryContext:
    """
    Step 12: Guardrails (Post-Processing)
    Strict, deterministic Python checks to verify the LLM's output before returning it.
    Catches hallucinations that the LLM itself cannot catch.
    """
    if context.is_abstained or not context.final_answer:
        return context

    with trace_step(context, "Step 12: Guardrails", input_count=1) as t:
        valid_chunk_ids = {chunk.metadata.chunk_id for chunk in context.selected_evidence}
        combined_evidence_text = " ".join([chunk.text for chunk in context.selected_evidence])

        # 1. Strict Citation Check
        resolved_citations = []
        for cit in context.citations:
            if cit in valid_chunk_ids:
                resolved_citations.append(cit)
            else:
                context.guardrails.unresolved_citations.append(cit)
        context.citations = resolved_citations

        # 2. Strict Numeric Grounding Check
        numbers_in_answer = set(re.findall(r'\b\d+(?:\.\d+)?\b', context.final_answer))
        for num_str in numbers_in_answer:
            if num_str not in combined_evidence_text:
                context.guardrails.ungrounded_numbers.append(num_str)

        # 3. Version Conflict Check
        releases_used = {chunk.metadata.release_version for chunk in context.selected_evidence}
        if len(releases_used) > 1:
            context.guardrails.version_conflict_detected = True

        # 4. Web Fallback & Domain Knowledge Disclaimers
        if context.guardrails.web_fallback_used:
            disclaimer = (
                "\n\n[DISCLAIMER: This answer incorporates evidence retrieved from the live web "
                "via fallback search, as local database specification chunks were insufficient.]"
            )
            context.final_answer += disclaimer
        elif context.guardrails.answered_from_general_knowledge:
            disclaimer = (
                "\n\n[NOTICE: Synthesized using general telecommunications and wireless engineering "
                "domain knowledge, as specific 3GPP specification clauses were not indexed in the local database.]"
            )
            context.final_answer += disclaimer

        t["output_count"] = 1
        t["extra"] = {
            "unresolved_citations": len(context.guardrails.unresolved_citations),
            "ungrounded_numbers": len(context.guardrails.ungrounded_numbers),
            "version_conflict": context.guardrails.version_conflict_detected,
        }

    return context
