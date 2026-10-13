"""
core/structured_llm.py
======================
Production-grade structured output validation and automatic retry for LLMs.

Eliminates fragile regex parsing by enforcing Pydantic model validation.
If the LLM returns invalid JSON or schema violations, it automatically
retries with targeted corrective feedback.
"""

import json
import re
from typing import Type, TypeVar, Optional, List
from pydantic import BaseModel, ValidationError
from core.interfaces import Generator

T = TypeVar("T", bound=BaseModel)


def extract_json_from_text(text: str) -> Optional[dict]:
    """
    Robustly extracts a JSON object from raw LLM text output,
    handling markdown code fences and surrounding conversational filler.
    """
    if not text:
        return None

    cleaned = text.strip()

    # Strip code block wrappers
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]

    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]

    cleaned = cleaned.strip()

    # 1. Direct parse attempt
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # 2. Regex fallback to extract outermost curly braces
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass

    return None


def generate_structured(
    generator: Generator,
    prompt: str,
    response_model: Type[T],
    max_retries: int = 2,
    context: Optional[List] = None,
) -> T:
    """
    Executes an LLM generation call, parsing and validating the response against
    a Pydantic BaseModel. If validation fails, retries with schema feedback.
    """
    current_prompt = prompt
    last_error = None

    for attempt in range(max_retries + 1):
        result = generator.generate(prompt=current_prompt, context=context or [])
        data = extract_json_from_text(result.content)

        if data is not None:
            try:
                validated = response_model.model_validate(data)
                return validated
            except ValidationError as ve:
                last_error = f"Schema validation error: {ve}"
        else:
            last_error = f"Could not parse valid JSON from output: {result.content[:150]}..."

        # If not the final attempt, formulate corrective prompt
        if attempt < max_retries:
            current_prompt = (
                f"{prompt}\n\n"
                f"[SYSTEM CORRECTION]: Your previous response was invalid.\n"
                f"Error: {last_error}\n"
                f"Required schema fields: {list(response_model.model_fields.keys())}\n"
                f"Output strictly raw, valid JSON only."
            )

    raise ValueError(f"Failed to generate valid {response_model.__name__} after {max_retries + 1} attempts. Last error: {last_error}")
