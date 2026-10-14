# TelecomRAG: 3GPP Standards Intelligence System
### Domain-Adapted Retrieval-Augmented Generation for 5G/NR Technical Specifications

[![Web Console](https://img.shields.io/badge/Web%20Console-Live%20Demo-10b981?style=for-the-badge&logo=render)](https://episteme-6tt4.onrender.com)
[![API Documentation](https://img.shields.io/badge/Swagger%20Docs-Interactive%20API-3b82f6?style=for-the-badge&logo=fastapi)](https://domain-adapted-rag.onrender.com/docs)
[![Backend Health](https://img.shields.io/badge/API%20Status-Healthy-22c55e?style=for-the-badge)](https://domain-adapted-rag.onrender.com/health)

---

## 1. Overview

Telecommunications technical specifications published by the **3rd Generation Partnership Project (3GPP)** contain thousands of densely interconnected standards (TS 38.331, TS 38.211, TS 38.214, TS 38.300) characterized by:
- **Colliding Technical Acronyms**: UE, AMF, SMF, UPF, RRC, BWP, PRACH, CSI-RS.
- **Cross-Release Protocol Divergence**: Enhancements in Release 17/18 superseding or altering Release 15/16 baseline behaviors.
- **Mission-Critical Parameter Sensitivity**: Strict numerical tolerances for timers, slot offsets, frequency ranges (FR1/FR2), and ciphering algorithms.

**TelecomRAG** is a production-grade, domain-adapted RAG platform built to query 3GPP specifications with zero hallucinations. It pairs hybrid vector-lexical indexing with a deterministic **12-layered pipeline**, an in-memory **semantic vector cache**, **real-time SSE execution tracing**, and strict post-generation **deterministic guardrails**.

- **Live Web Console**: [https://episteme-6tt4.onrender.com](https://episteme-6tt4.onrender.com)
- **Interactive Swagger API Docs**: [https://domain-adapted-rag.onrender.com/docs](https://domain-adapted-rag.onrender.com/docs)
- **ReDoc API Reference**: [https://domain-adapted-rag.onrender.com/redoc](https://domain-adapted-rag.onrender.com/redoc)

---

## 2. System Architecture

```mermaid
flowchart TD
    User([Client / Web Console]) --> Gateway[FastAPI Gateway :8001]

    subgraph StateAndCache ["State, Quota & Semantic Cache"]
        Redis[("Upstash Redis Stack\n• Semantic Vector Cache (KNN Cosine)\n• Guest Quotas & IP Rate Limiting\n• Multi-Turn Conversation Memory")]
    end

    Gateway <--> Redis
    Gateway -- "Semantic Cache Hit (<25ms)" --> QuickReturn([Return Cached Answer])

    subgraph CoreEngine ["Domain Execution Engine"]
        Gateway -- "Cache Miss" --> Pipeline["12-Layered Domain-Adapted Pipeline\n(Understanding ➔ Hybrid Retrieval ➔ RRF ➔ Cross-Encoder ➔ Validation ➔ Web Fallback ➔ Abstention Gate ➔ Synthesis ➔ Guardrails)"]
        
        Qdrant[("Qdrant Cloud\n1,748 3GPP Rel-17 Chunks\nDense: 1024-dim | Sparse BM25")]
        WebEngine["External Search Engine\n(DuckDuckGo Live Search)"]
        LLM["Domain LLM\n(Google Gemini 2.5 Flash)"]
        Postgres[("Neon PostgreSQL Pool\nAudit Logs & Metrics")]

        Pipeline <--> Qdrant
        Pipeline <--> WebEngine
        Pipeline <--> LLM
        Pipeline --> Postgres
    end

    Pipeline --> Stream([Real-Time SSE Stream / JSON Response])
    Stream --> User
```

---

## 3. The 12-Layered Pipeline

Every query passes through a sequential state machine implemented in [`pipeline/orchestrator.py`](telecom_qa_backend/pipeline/orchestrator.py):

| Step | Layer | Functionality | Implementation |
|:---:|---|---|---|
| **01** | **Understanding** | Normalizes acronyms, detects intent, and generates multi-hop sub-queries. | Structured schema extraction with `google-genai` |
| **02** | **Representation** | Generates 1024-dim dense vectors and sparse lexical weights with HyDE. | `BAAI/bge-m3` + Hypothetical Document Embeddings |
| **03** | **Hybrid Retrieval** | Runs dual dense-cosine and sparse dot-product vector search. | Qdrant Cloud HNSW vector index |
| **04** | **Candidate Fusion** | Merges dense and sparse candidates using reciprocal rankings. | Reciprocal Rank Fusion (RRF, $k=60$) |
| **05** | **Filtering** | Prunes low-confidence candidates before computationally heavy reranking. | Score cutoff filter (`MIN_SCORE = 0.005`) |
| **06** | **Reranking** | Cross-encoder joint scoring over $(Query, Passage)$ token pairs. | `BAAI/bge-reranker-v2-m3` (FP16 optimized) |
| **07** | **Evidence Selection** | Selects top-$K$ chunks within context budget; calibrates confidence. | Hierarchical Clause sorting + Sigmoid calibration |
| **08** | **Validation** | LLM-as-a-Judge inspects evidence sufficiency and checks for contradictions. | Structured evaluation protocol |
| **09** | **Web Fallback** | Supplementary search triggered when local specification chunks are missing. | Live DuckDuckGo search integration |
| **10** | **Abstention Gate** | Halts non-telecom out-of-domain queries; permits technical synthesis. | Domain answerability classification gate |
| **11** | **Generation** | Synthesizes answers using retrieved specifications and domain principles. | Gemini 2.5 Flash with strict bracketed citations |
| **12** | **Guardrails** | Deterministic post-generation verification and citation integrity checks. | Verbatim numerical checks, release congruency |

---

## 4. Key Infrastructure & Capabilities

### Upstash Redis Semantic Cache
- Pre-retrieval vector similarity index (`qa_cache_idx`) over 1024-dim BGE-M3 query embeddings.
- When an incoming query matches a cached entry with **Cosine similarity $\ge 0.90$**, the complete verified response is returned in **under 25 ms**, bypassing the pipeline.
- Dynamic TTL: **30 days** for official 3GPP specifications; **24 hours** for web-fallback queries.

### Multi-Turn Conversation Memory
- Tracks conversational state using sliding windows of the last 10 messages (5 user-assistant turns) in Redis (`session:{id}:turns`) with a rolling 1-hour expiration.

### Rate Limiting & Guest Quotas
- Protects cloud infrastructure and API budgets:
  - **Unauthenticated Guests**: Capped at 3 free queries before prompting Google Sign-in.
  - **Burst Protection**: Max 5 queries / minute per client IP.
  - **Daily Quota**: Max 30 queries / day per client IP.

### Deterministic Python Guardrails
1. **Citation Resolution**: Asserts every citation tag `[chunk_id]` strictly maps to retrieved specification chunks. Unresolved citations are automatically pruned.
2. **Verbatim Numerical Grounding**: Scans all numeric values (timers, frequencies, bandwidths) against reference source text via regex.
3. **Cross-Release Conflict Warning**: Flags when evidence simultaneously cites disparate releases (e.g., Rel-15 vs Rel-17).
4. **Transparent Disclaimers**: Discloses when answers incorporate live web fallback evidence or broad telecommunications domain principles.

### Dual-Engine Persistence Layer
- **Neon PostgreSQL (Production)**: Threaded connection pool (capped at 8 connections) for query auditing, user sessions, latency percentiles, and cache metrics.
- **SQLite (Local Dev)**: Embedded zero-config fallback (`telecom_qa.db`).

---

## 5. Evaluation & Benchmark Suite

The platform includes a 14-question benchmark suite in [`evaluation/benchmark.py`](telecom_qa_backend/evaluation/benchmark.py) evaluating:
- **Radio Resource Control (RRC)**: Connection establishment, state transitions, timers.
- **5G Core Network (5GC)**: AMF mobility management, SMF PDU sessions, network slicing.
- **Physical Layer / RAN**: FR1 vs FR2 frequency bands, antenna configurations, duplexing modes.
- **Security & Authentication**: 5G-AKA authentication vectors, SUPI/SUCI privacy.
- **Out-of-Domain Negative Controls**: General chit-chat, weather, non-telecom tasks (verifies clean abstention).

Run benchmark suite:
```bash
cd telecom_qa_backend
python -m evaluation.benchmark
```

---

## 6. Quickstart

### 1. Clone & Configure Environment
```bash
git clone https://github.com/pragyan2905/domain_adapted_rag.git
cd domain_adapted_rag/telecom_qa_backend

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create `.env` file:
```env
GEMINI_API_KEY="your_gemini_api_key"
QDRANT_URL="your_qdrant_url"
QDRANT_API_KEY="your_qdrant_key"
REDIS_URL="rediss://default:your_redis_url:6379"
DATABASE_URL="postgresql://user:pass@ep-hostname.neon.tech/neondb?sslmode=require"
```

### 2. Start Backend API
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8001
```
- API Docs: `http://localhost:8001/docs`
- Health Check: `http://localhost:8001/health`

### 3. Launch Frontend Console
```bash
cd ../telecom_qa_frontend
python3 -m http.server 5500
```
Open `http://localhost:5500` in your browser.

---

## 7. Technology Stack

- **Vector Database**: Qdrant Cloud (HNSW indexed, 1024-dim dense + sparse vectors)
- **Semantic Cache & State**: Upstash Redis (RediSearch vector KNN, session memory, rate limiting)
- **Relational Storage**: Neon PostgreSQL (connection pooled) / SQLite
- **Embedding & Reranker**: `BAAI/bge-m3` (FP16) & `BAAI/bge-reranker-v2-m3` (Cross-encoder)
- **LLM Synthesis**: Google Gemini 2.5 Flash via `google-genai` SDK
- **Backend**: FastAPI, Pydantic v2, Uvicorn (ASGI)
- **Frontend**: Native Vanilla JavaScript (SSE streaming), Semantic HTML5, CSS Design System
- **Hosting**: Render (Static Web Service & Web API Service)
