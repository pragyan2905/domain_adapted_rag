from core.schemas import QueryContext, AnswerabilityType
from core.tracer import trace_step

def execute(context: QueryContext) -> QueryContext:
    """
    Step 10: Abstention Gate
    Halts synthesis only if:
      1. Query is strictly out-of-domain (no relation to telecom/wireless/3GPP).
      2. Direct standards contradiction was identified.
    In-domain queries proceed to Step 11 for synthesis via web fallback
    and broad telecommunications engineering domain knowledge.
    """
    with trace_step(context, "Step 10: Abstention Gate", input_count=1) as t:
        # 1. Hard check for non-telecom out-of-domain queries
        if context.understanding and context.understanding.answerability == AnswerabilityType.OUT_OF_DOMAIN:
            context.is_abstained = True
            context.abstention_reason = "This query is outside the scope of telecommunications and 3GPP standards."
            t["output_count"] = 0
            t["extra"] = {"abstained": True, "reason": context.abstention_reason, "type": "out_of_domain"}
            return context

        # 2. Hard check for direct standards contradiction
        if context.evidence_validation and context.evidence_validation.contradiction_found:
            context.is_abstained = True
            context.abstention_reason = f"Contradiction found in standards evidence: {context.evidence_validation.contradiction_details or context.evidence_validation.reasoning}"
            t["output_count"] = 0
            t["extra"] = {"abstained": True, "reason": context.abstention_reason, "type": "contradiction"}
            return context

        # 3. In-domain queries: proceed to Step 11
        # If local database chunks were insufficient, Step 11 will synthesize
        # using web search evidence and broad telecommunications domain knowledge.
        t["output_count"] = 1
        t["extra"] = {
            "abstained": False,
            "synthesizing_via_domain_knowledge": not (context.evidence_validation and context.evidence_validation.is_sufficient)
        }

    return context
