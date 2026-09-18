import re
import uuid
from typing import List, Dict, Any, Callable
from pydantic import BaseModel

from core.schemas import ChunkMetadata, SourceType

class RawClause(BaseModel):
    """Temporary dataclass used during parsing before final ChunkMetadata is built."""
    clause_id: str
    title: str
    content: str
    token_count: int = 0
    breadcrumb: str = ""
    parent_clause_id: str | None = None

class ClauseAwareChunker:
    """
    Custom chunker designed strictly for 3GPP hierarchical documents.
    It respects the logical boundaries (e.g., 5.3.1.2) rather than splitting
    blindly at fixed character counts.
    """
    def __init__(
        self, 
        max_tokens: int = 400, 
        min_tokens: int = 30, 
        token_counter: Callable[[str], int] = None
    ):
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        
        # If no strict tokenizer is provided, use a simple heuristic (1 token ~= 4 chars)
        self.count_tokens = token_counter if token_counter else lambda text: len(text) // 4

    def chunk_document(self, text: str, ts_number: str, series: str, release_version: str) -> List[Dict[str, Any]]:
        """
        Main entry point. Takes raw text of a 3GPP document and returns final chunks.
        Returns a list of dictionaries containing 'text' and 'metadata' (matching ChunkMetadata).
        """
        # 1. Parse into raw clauses
        raw_clauses = self._extract_clauses(text)
        
        # 2. Build hierarchical breadcrumbs
        raw_clauses = self._build_breadcrumbs(raw_clauses)
        
        # 3. Merge undersized clauses (signal-to-noise ratio)
        raw_clauses = self._merge_undersized(raw_clauses)
        
        # 4. Split oversized clauses (context window budget)
        final_clauses = self._split_oversized(raw_clauses)
        
        # 5. Format into final Output
        return self._format_output(final_clauses, ts_number, series, release_version)

    def _extract_clauses(self, text: str) -> List[RawClause]:
        """
        Regex parser finding 3GPP headings like "5.3.1.2 Conditions".
        Matches a line starting with numbers separated by dots, followed by a space and text.
        """
        # Regex breakdown: 
        # ^(\d+(?:\.\d+)*) -> Captures the clause number (e.g., 5, 5.3, 5.3.1.2) at the start of a line
        # \s+([^\n]+) -> Captures the title (everything else on the line)
        heading_pattern = re.compile(r"^(\d+(?:\.\d+)*)\s+([^\n]+)", re.MULTILINE)
        
        matches = list(heading_pattern.finditer(text))
        clauses = []
        
        for i, match in enumerate(matches):
            clause_id = match.group(1).strip()
            title = match.group(2).strip()
            
            # The content is everything from the end of this heading to the start of the next
            start_idx = match.end()
            end_idx = matches[i+1].start() if i + 1 < len(matches) else len(text)
            
            content = text[start_idx:end_idx].strip()
            
            clauses.append(RawClause(
                clause_id=clause_id,
                title=title,
                content=content,
                token_count=self.count_tokens(content)
            ))
            
        return clauses

    def _build_breadcrumbs(self, clauses: List[RawClause]) -> List[RawClause]:
        """
        Derives the hierarchical path and parent ID for every clause.
        Example breadcrumb: "5.3 Mobility Management > 5.3.1 Handover > 5.3.1.2 Conditions"
        """
        # Dictionary to keep track of the most recent title for any given clause depth
        # e.g., "5.3": "Mobility Management"
        history = {}
        
        for clause in clauses:
            history[clause.clause_id] = clause.title
            
            parts = clause.clause_id.split('.')
            breadcrumb_parts = []
            
            # Reconstruct the path from the root down to the current clause
            for i in range(1, len(parts) + 1):
                ancestor_id = ".".join(parts[:i])
                ancestor_title = history.get(ancestor_id, "")
                if ancestor_title:
                    breadcrumb_parts.append(f"{ancestor_id} {ancestor_title}")
                    
            clause.breadcrumb = " > ".join(breadcrumb_parts)
            
            # Assign parent ID (if it's 5.3.1, parent is 5.3)
            if len(parts) > 1:
                clause.parent_clause_id = ".".join(parts[:-1])
                
        return clauses

    def _merge_undersized(self, clauses: List[RawClause]) -> List[RawClause]:
        """
        Merges clauses that have too few tokens (e.g. empty parent headers) into the next clause.
        """
        merged = []
        skip_next = False
        
        for i in range(len(clauses)):
            if skip_next:
                skip_next = False
                continue
                
            current = clauses[i]
            
            # If undersized and not the last element, merge it forward
            if current.token_count < self.min_tokens and i + 1 < len(clauses):
                next_clause = clauses[i+1]
                
                # We prepend the current title and content to the next clause's content
                combined_content = f"[{current.clause_id} {current.title}]\n{current.content}\n\n{next_clause.content}"
                next_clause.content = combined_content.strip()
                next_clause.token_count = self.count_tokens(next_clause.content)
                
                # Note: We do NOT append current to merged; it gets absorbed by next_clause
            else:
                merged.append(current)
                
        return merged

    def _split_oversized(self, clauses: List[RawClause]) -> List[RawClause]:
        """
        Splits clauses that exceed the token budget into smaller chunks.
        Every sub-chunk retains the original metadata and breadcrumb.
        """
        final_chunks = []
        
        for clause in clauses:
            if clause.token_count <= self.max_tokens:
                final_chunks.append(clause)
                continue
                
            # Naive paragraph/sentence splitting fallback for oversized chunks
            # In a production scenario, you would use a RecursiveCharacterTextSplitter logic here
            paragraphs = clause.content.split('\n\n')
            
            current_sub_content = ""
            for p in paragraphs:
                if self.count_tokens(current_sub_content) + self.count_tokens(p) > self.max_tokens:
                    # Save current and start fresh
                    if current_sub_content:
                        sub_clause = clause.model_copy()
                        sub_clause.content = current_sub_content.strip()
                        sub_clause.token_count = self.count_tokens(sub_clause.content)
                        final_chunks.append(sub_clause)
                    current_sub_content = p
                else:
                    current_sub_content += f"\n\n{p}" if current_sub_content else p
                    
            # Catch the remainder
            if current_sub_content:
                sub_clause = clause.model_copy()
                sub_clause.content = current_sub_content.strip()
                sub_clause.token_count = self.count_tokens(sub_clause.content)
                final_chunks.append(sub_clause)
                
        return final_chunks

    def _format_output(
        self, 
        clauses: List[RawClause], 
        ts_number: str, 
        series: str, 
        release_version: str
    ) -> List[Dict[str, Any]]:
        """
        Converts internal RawClauses into the final text + ChunkMetadata schema required by the pipeline.
        Prepend the breadcrumb to the text before embedding.
        """
        results = []
        for clause in clauses:
            metadata = ChunkMetadata(
                chunk_id=str(uuid.uuid4()),
                ts_number=ts_number,
                series=series,
                release_version=release_version,
                clause_id=clause.clause_id,
                clause_title=clause.title,
                breadcrumb=clause.breadcrumb,
                parent_clause_id=clause.parent_clause_id,
                token_count=clause.token_count,
                source_type=SourceType.CORPUS
            )
            
            # CRITICAL 3GPP FIX: Bake the breadcrumb directly into the text that gets embedded
            embedded_text = f"CONTEXT PATH: {clause.breadcrumb}\n---\n{clause.content}"
            
            results.append({
                "text": embedded_text,
                "metadata": metadata
            })
            
        return results
