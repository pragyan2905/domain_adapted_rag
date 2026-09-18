from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field

# -------------------------------------------------------------------------
# 1. Chunk and Metadata Models
# -------------------------------------------------------------------------
class SourceType(str, Enum):
    CORPUS = "corpus"
    WEB = "web"

class ChunkMetadata(BaseModel):
    """
    Exact schema as mandated for 3GPP hierarchical documents.
    """
    chunk_id: str = Field(..., description="Primary key, used by the citation-check guardrail")
    ts_number: str = Field(..., description="Filter by document (e.g. '38.331')")
    series: str = Field(..., description="Coarse filter (RAN vs Core) (e.g. '38-series')")
    release_version: str = Field(..., description="Enables the release-conflict guardrail (e.g. 'Rel-17')")
    clause_id: str = Field(..., description="Precise citation target (e.g. '5.3.1.2')")
    clause_title: str = Field(..., description="Human-readable citation display")
    breadcrumb: str = Field(..., description="Full hierarchy path (e.g. '5.3 Mobility Management > ...')")
    parent_clause_id: Optional[str] = Field(None, description="For parent-child chunk merging")
    token_count: int = Field(..., description="Enforces context-window budget")
    source_type: SourceType = Field(default=SourceType.CORPUS, description="Distinguishes DB vs Web chunks")

class RetrievedChunk(BaseModel):
    """
    Represents a chunk retrieved from either Qdrant or Web Fallback.
    """
    text: str
    metadata: ChunkMetadata
    score: float = 0.0  # Dense/Sparse/Fusion/Reranker score depending on pipeline stage

# -------------------------------------------------------------------------
# 2. Pipeline State Models (The Context)
# -------------------------------------------------------------------------
class AnswerabilityType(str, Enum):
    FACTUAL_LOOKUP = "factual_lookup"
    MULTI_STEP = "multi_step_procedural"
    OUT_OF_DOMAIN = "out_of_domain"

class QueryUnderstanding(BaseModel):
    normalized_query: str
    expanded_acronyms: dict[str, str] = Field(default_factory=dict)
    detected_intent: str
    answerability: AnswerabilityType

class EvidenceValidationResult(BaseModel):
    is_sufficient: bool
    contradiction_found: bool
    contradiction_details: Optional[str] = None
    reasoning: str

class GuardrailFlags(BaseModel):
    unresolved_citations: List[str] = Field(default_factory=list)
    ungrounded_numbers: List[str] = Field(default_factory=list)
    version_conflict_detected: bool = False
    web_fallback_used: bool = False

class QueryContext(BaseModel):
    """
    The central state object passed through the entire 12-step pipeline.
    It starts with just the 'original_query' and gets populated sequentially.
    """
    original_query: str
    
    # Step 1: Understanding
    understanding: Optional[QueryUnderstanding] = None
    
    # Step 2: Representation (Vectors are tiny, safe to keep in context for the lifespan of the query)
    target_ts_numbers: List[str] = Field(default_factory=list)
    target_releases: List[str] = Field(default_factory=list)
    query_dense_vector: Optional[List[float]] = None
    query_sparse_vector: Optional[dict[str, float]] = None
    
    # Step 3-7: Candidates
    dense_candidates: List[RetrievedChunk] = Field(default_factory=list)
    sparse_candidates: List[RetrievedChunk] = Field(default_factory=list)
    fused_candidates: List[RetrievedChunk] = Field(default_factory=list)
    filtered_candidates: List[RetrievedChunk] = Field(default_factory=list)
    reranked_candidates: List[RetrievedChunk] = Field(default_factory=list)
    selected_evidence: List[RetrievedChunk] = Field(default_factory=list)
    
    # Step 8-10: Validation & Abstention
    evidence_validation: Optional[EvidenceValidationResult] = None
    is_abstained: bool = False
    abstention_reason: Optional[str] = None
    
    # Step 11-12: Output
    final_answer: Optional[str] = None
    citations: List[str] = Field(default_factory=list)  # List of chunk_ids
    guardrails: GuardrailFlags = Field(default_factory=GuardrailFlags)
    confidence_score: float = 0.0
