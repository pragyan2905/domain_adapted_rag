from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import time

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
    sub_queries: List[str] = Field(default_factory=list)

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
    answered_from_general_knowledge: bool = False

# -------------------------------------------------------------------------
# 3. Pipeline Observability (Tracing)
# -------------------------------------------------------------------------
class StepTrace(BaseModel):
    """
    Captures timing and throughput metrics for a single pipeline step.
    """
    step_name: str
    duration_ms: float
    input_count: int = 0   # Candidates entering this step
    output_count: int = 0  # Candidates surviving this step
    metadata: Dict[str, Any] = Field(default_factory=dict)

class PipelineTrace(BaseModel):
    """
    Full trace of the entire pipeline execution, returned alongside the answer.
    Enables the frontend to render a live funnel/step-by-step visualization.
    """
    total_duration_ms: float = 0.0
    cache_hit: bool = False
    steps: List[StepTrace] = Field(default_factory=list)

# -------------------------------------------------------------------------
# 4. Conversation Memory
# -------------------------------------------------------------------------
class ConversationTurn(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class QueryContext(BaseModel):
    """
    The central state object passed through the entire 12-step pipeline.
    It starts with just the 'original_query' and gets populated sequentially.
    """
    original_query: str
    session_id: Optional[str] = None

    # Step 1: Understanding
    understanding: Optional[QueryUnderstanding] = None

    # Step 2: Representation (Vectors are tiny, safe to keep in context for the lifespan of the query)
    hypothetical_document: Optional[str] = None
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
    needs_web_search: bool = False

    # Conversation history injected from Redis before pipeline runs
    conversation_history: List[ConversationTurn] = Field(default_factory=list)

    # Pipeline trace (populated by each step, returned to client)
    pipeline_trace: PipelineTrace = Field(default_factory=PipelineTrace)


# -------------------------------------------------------------------------
# Generation & Evaluation Schemas
# -------------------------------------------------------------------------
class GenerationResponse(BaseModel):
    content: str = ""
    citations: List[str] = Field(default_factory=list)
    needs_web_search: bool = False
    answered_from_general_knowledge: bool = False


class CompareConfigResult(BaseModel):
    config_name: str
    answer: str
    confidence_score: float
    citations_count: int
    duration_ms: float
    retrieval_count: int
    guardrails_passed: bool


class CompareResponse(BaseModel):
    query: str
    advanced_rag: CompareConfigResult
    baseline_naive_rag: CompareConfigResult


# -------------------------------------------------------------------------
# API Contract Schemas
# -------------------------------------------------------------------------
class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None  # For multi-turn conversation


class QueryResponse(BaseModel):
    answer: str
    citations: List[str]
    is_abstained: bool
    abstention_reason: Optional[str] = None
    guardrails: GuardrailFlags
    confidence_score: float
    session_id: str
    pipeline_trace: PipelineTrace

