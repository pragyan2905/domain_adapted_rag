import time
from contextlib import contextmanager
from core.schemas import QueryContext, StepTrace

@contextmanager
def trace_step(context: QueryContext, step_name: str, input_count: int = 0, extra: dict = None):
    """
    Context manager that wraps a pipeline step and records timing + throughput
    into context.pipeline_trace.steps.

    Usage:
        with trace_step(context, "Step 01: Understanding", input_count=1) as t:
            # ... do work ...
            t["output_count"] = 1  # set output inside the block if needed
    """
    tracker = {"output_count": input_count, "extra": extra or {}}
    t_start = time.perf_counter()
    try:
        yield tracker
    finally:
        duration_ms = (time.perf_counter() - t_start) * 1000
        step_trace = StepTrace(
            step_name=step_name,
            duration_ms=round(duration_ms, 2),
            input_count=input_count,
            output_count=tracker["output_count"],
            metadata=tracker["extra"],
        )
        context.pipeline_trace.steps.append(step_trace)
