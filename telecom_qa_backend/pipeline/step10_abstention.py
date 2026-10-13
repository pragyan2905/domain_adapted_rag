from core.schemas import QueryContext
from core.tracer import trace_step

def execute(context: QueryContext) -> QueryContext:
    """
    Step 10: Abstention
    If we reached this point and the evidence is STILL insufficient
    (even after the Web Fallback branch), we pull the emergency brake.
    We refuse to generate an answer to prevent hallucinations.
    """
    with trace_step(context, "Step 10: Abstention Gate", input_count=1) as t:
        if context.evidence_validation and not context.evidence_validation.is_sufficient:
            context.is_abstained = True
            context.abstention_reason = context.evidence_validation.reasoning
            t["output_count"] = 0
            t["extra"] = {"abstained": True, "reason": context.abstention_reason}
        else:
            t["output_count"] = 1
            t["extra"] = {"abstained": False}

    return context
