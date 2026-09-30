# Domain Adapted RAG: 3GPP Telecom Quality Assurance System

## 1. Project Overview
This project implements an advanced, domain-adapted Retrieval-Augmented Generation (RAG) system specialized for the 3GPP Telecommunications domain. It is designed to accurately answer complex technical queries regarding telecommunication standards (e.g., 5G technologies, RAN, Core Networks) by relying on a highly structured 12-step pipeline, strict guardrails, and hierarchical chunk metadata matching. It effectively minimizes hallucinations and utilizes abstention-aware answering (refusing to answer when evidence is insufficient or out of domain).

## 2. System Architecture
The system follows a microservices-inspired architecture containing a frontend web application, a backend API server, a vector database, and an in-memory caching layer.

### Core Technologies
*   **Backend Framework**: FastAPI, Pydantic, Uvicorn
*   **Vector Database**: Qdrant (deployed via Docker)
*   **Caching Layer**: Redis Stack (Semantic caching using dense vectors, deployed via Docker)
*   **LLM Providers / Generators**: Google Gemini (via `google-genai`). *Note: Local models (e.g., Qwen2.5-1.5B-Instruct fine-tuned via QLoRA) are not currently being used in the active pipeline but will be integrated in future phases.*
*   **Embedding & Reranking**: BGE Models (`FlagEmbedding` with FP16 support)
*   **Web Fallback / Search**: DuckDuckGo Search API (`ddgs`)
*   **Frontend**: Vanilla HTML5, CSS3, JavaScript

## 3. The 12-Step RAG Assembly Line
The backend execution is driven by a linear `QueryContext` state machine that traverses a highly deterministic 12-step pipeline:

1.  **Step 01 - Understanding**: Normalizes the user query, expands domain-specific acronyms, and detects query intent (Factual Lookup, Multi-step, or Out-of-Domain).
2.  **Step 02 - Representation**: Generates dense and sparse vector embeddings of the query using the BGE embedder.
3.  **Step 03 - Retrieval**: Fetches top candidate chunks from the Qdrant vector store.
4.  **Step 04 - Fusion**: Implements Reciprocal Rank Fusion (RRF) to merge and normalize scores from multiple retrieval strategies.
5.  **Step 05 - Filtering**: Prunes retrieved candidates that fall below a strict confidence threshold.
6.  **Step 06 - Reranking**: Applies a Cross-Encoder (BGE Reranker) to evaluate the exact semantic match between the query and the filtered chunks.
7.  **Step 07 - Selection**: Identifies and isolates the top-K chunks to serve as the grounded context for the generator.
8.  **Step 08 - Validation**: Verifies if the selected context is logically sufficient to answer the question, checking for contradictions.
9.  **Step 09 - Web Fallback**: Triggers dynamically if local evidence validation fails, querying DuckDuckGo to augment the context.
10. **Step 10 - Abstention**: Intercepts the pipeline if the query is out of domain or lacks grounded evidence, forcing an "I cannot answer" response rather than hallucinating.
11. **Step 11 - Generation**: Passes the validated context and query to the LLM (currently Gemini) for generating the final answer.
12. **Step 12 - Guardrails**: A strict, deterministic Python post-processing step to catch LLM hallucinations before returning the answer. It implements:
    *   **Strict Citation Check**: Validates that all citations claimed by the LLM perfectly match the `chunk_id`s in the selected evidence.
    *   **Strict Numeric Grounding Check**: Uses regex to extract all numbers from the generated answer and asserts they exist verbatim in the raw evidence text, flagging ungrounded numbers.
    *   **Version Conflict Check**: Flags answers that draw evidence from multiple 3GPP releases (e.g., mixing Rel-17 and Rel-18).
    *   **Web Fallback Disclaimer**: Appends a clear text disclaimer if DuckDuckGo search was utilized due to insufficient local knowledge.

## 4. Semantic Caching Implementation
To drastically reduce latency and API costs, the system uses **Redis Stack** as a semantic cache. 
*   Before entering the pipeline, the incoming query is converted to a 1024-dimensional dense vector (FLOAT32).
*   RediSearch performs a K-Nearest Neighbors (KNN) search (`*=>[KNN 1 @vector $vec AS score]`) using a `FLAT` index and a `COSINE` distance metric.
*   The system calculates similarity as `1.0 - distance`. If this similarity score is >= 0.90, the pipeline is entirely bypassed, and the cached JSON response is immediately returned.
*   Cache Time-To-Live (TTL) is dynamically assigned: 30 days for pure 3GPP documentation answers and 24 hours for answers relying on Web Fallback.

## 5. Domain-Specific Metadata Schema
Each document chunk ingested into Qdrant is strictly typed with 3GPP metadata to enable precision filtering and citation generation:
*   `chunk_id`: Primary key for citation tracking.
*   `ts_number`: Technical Specification number (e.g., '38.331').
*   `series`: 3GPP Series (e.g., '38-series').
*   `release_version`: 3GPP Release (e.g., 'Rel-17').
*   `clause_id` & `clause_title`: Precise targeting for user-facing citations.
*   `breadcrumb`: Full hierarchical path within the standard.

## 6. Model Fine-Tuning (QLoRA)
A dedicated training pipeline (`qlora_finetune.ipynb`) was developed to adapt an open-weights model for local, privacy-preserving execution:
*   **Base Model**: `Qwen/Qwen2.5-1.5B-Instruct`
*   **Dataset**: `netop/TeleQnA` (Test split repartitioned for training/eval)
*   **Methodology**: 4-bit NF4 Quantized Low-Rank Adaptation (QLoRA) using `Unsloth` (r=16, alpha=16).
*   **Negative Sampling**: The model is heavily trained on "hard negatives" and "answer-contamination filters". If the provided context does not contain the answer, the model is explicitly trained to output a hardcoded abstention string ("I cannot answer this from the provided context.") rather than hallucinating.

## 7. Setup and Execution

### Prerequisites
*   Docker & Docker Compose
*   Python 3.9+
*   Environment Variables: `GEMINI_API_KEY` stored in `.env`

### Running the Services
1.  **Launch Databases**:
    ```bash
    cd telecom_qa_backend
    docker-compose up -d
    ```
    This spins up Qdrant (Port 6333) and Redis Stack (Port 6380).
2.  **Install Python Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Start the Backend Server**:
    ```bash
    uvicorn api.main:app --host 127.0.0.1 --port 8001
    ```
4.  **Run the Frontend**:
    Serve the `telecom_qa_frontend` directory using any local web server (e.g., `python -m http.server 3000`).

## 8. Current Status & Future Roadmap
### Accomplished
*   Completed the 12-step modular pipeline architecture.
*   Integrated Qdrant vector storage and Redis semantic caching.
*   Implemented strict 3GPP metadata schemas and post-generation guardrails.
*   Successfully fine-tuned Qwen2.5-1.5B for abstention-aware context extraction.
*   Developed vanilla frontend UI and API endpoints.

### Future Work
*   **Local LLM Integration**: Replace the Gemini API endpoint in `api/main.py` with the locally fine-tuned Qwen2.5 model using vLLM or HuggingFace pipelines for completely offline generation.
*   **Corpus Ingestion Automation**: Enhance the scripts utilizing `python-docx` to continuously parse, chunk (`utils/chunking.py`), and ingest new 3GPP releases into Qdrant automatically.
*   **Advanced Reranking**: Upgrade the BGE cross-encoder layer to utilize a domain-specific fine-tuned reranker.
*   **UI Enhancements**: Transition the frontend to a robust framework (e.g., React or Next.js) to handle complex citation highlighting and nested breadcrumb navigation.
*   **Telemetry & Brier Score Analysis**: Implement tracking for confidence calibration to evaluate the abstention mechanism mathematically.
