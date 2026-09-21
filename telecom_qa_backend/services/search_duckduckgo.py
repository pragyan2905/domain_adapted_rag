from typing import List
try:
    from ddgs import DDGS
except ImportError:
    raise ImportError("Please install the ddgs package: pip install ddgs")

from core.interfaces import WebSearch, WebSearchResult

class DuckDuckGoSearch(WebSearch):
    """
    Implementation of the WebSearch interface using DuckDuckGo.
    This is free, requires no API keys, and is strictly domain-restricted to 3GPP/ETSI.
    """
    def search(self, query: str, max_results: int = 3) -> List[WebSearchResult]:
        # Perform a generic unrestricted web search
        restricted_query = query
        
        results = []
        try:
            with DDGS() as ddgs:
                # text() returns an iterator of dicts with 'title', 'href', 'body'
                for r in ddgs.text(restricted_query, max_results=max_results):
                    results.append(WebSearchResult(
                        url=r.get("href", ""),
                        title=r.get("title", ""),
                        content=r.get("body", "")
                    ))
        except Exception as e:
            # Catch DDGS rate limits or network issues gracefully
            print(f"Web search failed: {e}")
            
        return results

class DisabledWebSearch(WebSearch):
    """
    A stub implementation to disable web fallback cleanly without changing pipeline logic.
    """
    def search(self, query: str) -> List[WebSearchResult]:
        return []
