from typing import List
try:
    from google import genai
    from google.genai import types
    import time
except ImportError:
    raise ImportError("Please install the Gemini SDK: pip install google-genai")

from core.interfaces import Generator, GenerationResult
from core.schemas import RetrievedChunk

class GeminiGenerator(Generator):
    """
    Concrete implementation of the Generator interface using the Gemini API.
    Uses the new google-genai SDK (replaces deprecated google-generativeai).
    This acts as our placeholder until the local Qwen QLoRA model is merged.
    """
    def __init__(self, api_key: str, model_name: str = "gemini-3.5-flash"):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def generate(self, prompt: str, context: List[RetrievedChunk]) -> GenerationResult:
        """
        Executes the LLM call. The pipeline handles formatting the chunks into the prompt,
        so this just wraps the API call.
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(response_mime_type="application/json")
                )
                # The pipeline parses the JSON content itself, so we just return the raw text
                return GenerationResult(
                    content=response.text,
                    citations=[]
                )
            except Exception as e:
                print(f"Gemini API Error (Attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt) # Exponential backoff
                else:
                    return GenerationResult(
                        content='{"content": "Error communicating with LLM.", "citations": []}',
                        citations=[]
                    )

class LocalGenerator(Generator):
    """
    Stub for the future local Qwen2.5-1.5B QLoRA model.
    To be implemented when fine-tuning is complete.
    """
    def __init__(self, model_path: str):
        raise NotImplementedError("Local LLM integration pending QLoRA merge.")

    def generate(self, prompt: str, context: List[RetrievedChunk]) -> GenerationResult:
        raise NotImplementedError()
