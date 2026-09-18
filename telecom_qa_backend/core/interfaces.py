from abc import ABC, abstractmethod
from typing import List
from pydantic import BaseModel

from core.schemas import RetrievedChunk

# -------------------------------------------------------------------------
# Generation Interfaces
# -------------------------------------------------------------------------
class GenerationResult(BaseModel):
    """
    Standardized output from any LLM generator.
    """
    content: str
    citations: List[str]  # List of chunk_ids that the LLM decided to cite

class Generator(ABC):
    """
    Abstract Base Class for the LLM. 
    This ensures we can swap Gemini for a local fine-tuned model effortlessly.
    """
    @abstractmethod
    def generate(self, prompt: str, context: List[RetrievedChunk]) -> GenerationResult:
        """
        Generate an answer given a prompt and a set of retrieved chunks.
        """
        pass

# -------------------------------------------------------------------------
# Web Search Interfaces
# -------------------------------------------------------------------------
class WebSearchResult(BaseModel):
    url: str
    title: str
    content: str

class WebSearch(ABC):
    """
    Abstract Base Class for Web Fallback.
    """
    @abstractmethod
    def search(self, query: str) -> List[WebSearchResult]:
        """
        Execute a search and return parsed results.
        """
        pass
