from core.schemas import QueryContext

def execute(context: QueryContext) -> QueryContext:
    """
    Step 10: Abstention
    If we reached this point and the evidence is STILL insufficient 
    (even after the Web Fallback branch), we pull the emergency brake.
    We refuse to generate an answer to prevent hallucinations.
    """
    if context.evidence_validation and not context.evidence_validation.is_sufficient:
        context.is_abstained = True
        context.abstention_reason = context.evidence_validation.reasoning
        
    return context
