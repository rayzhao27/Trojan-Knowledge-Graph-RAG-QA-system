import os
import sys
import time
import json
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from contextlib import asynccontextmanager
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response, StreamingResponse, HTMLResponse

logger = logging.getLogger(__name__)

# --- Runtime settings to avoid deadlocks / segfaults ---

# 1. Disable HuggingFace tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 2. Keep BLAS / OpenMP threads under control on macOS
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

try:
    import faiss

    faiss.omp_set_num_threads(1)
except ImportError:
    logger.warning("Failed to import faiss")
    faiss = None

# --- end runtime settings ---
# Load environment variables from .env file
from dotenv import load_dotenv

load_dotenv()

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.llm_generator import LLMGenerator

# --- Prometheus Metrics ---
# 1. REQUEST METRICS (with basic labels)
REQUEST_COUNT = Counter(
    'api_requests_total',
    'Total API requests',
    ['endpoint', 'status']  # endpoint: /search or /ask, status: 200 or 500
)

REQUEST_DURATION = Histogram(
    'api_request_duration_seconds',
    'Request duration in seconds',
    ['endpoint'],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)  # Tuned for your app
)

# 2. LLM METRICS (track model usage)
LLM_REQUEST_COUNT = Counter(
    'llm_requests_total',
    'Total LLM requests',
    ['model', 'status']  # Track which model and if it succeeded
)

LLM_REQUEST_DURATION = Histogram(
    'llm_request_duration_seconds',
    'LLM request duration',
    ['model'],
    buckets=(0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0)  # LLM is slower
)

LLM_TOKENS_USED = Counter(
    'llm_tokens_total',
    'Total tokens consumed',
    ['model']
)

# 3. ERROR TRACKING (important for production)
ERROR_COUNT = Counter(
    'errors_total',
    'Total errors',
    ['endpoint', 'error_type']  # Know what's failing and why
)


# --- Global Services ---
retriever: Optional[HybridRetriever] = None
llm_generator: Optional[LLMGenerator] = None
admin_control = {"use_rerank": True, "use_graph": True, "use_llm": True}

DUMMY_CHUNKS = [
    {
        'chunk_id': 'test_doc_p1_c1',
        'text': 'RAG stands for Retrieval Augmented Generation. It is a technique that combines information retrieval with language generation.',
        'source': 'test_document.pdf',
        'page_num': 1
    },
    {
        'chunk_id': 'test_doc_p1_c2',
        'text': 'Vector databases store embeddings of text chunks and enable semantic search using similarity metrics.',
        'source': 'test_document.pdf',
        'page_num': 1
    },
    {
        'chunk_id': 'test_doc_p2_c1',
        'text': 'BM25 is a ranking function used for keyword-based search. It works well for exact term matching.',
        'source': 'test_document.pdf',
        'page_num': 2
    },
    {
        'chunk_id': 'test_doc_p2_c2',
        'text': 'Hybrid retrieval combines vector search and keyword search to get the best of both approaches.',
        'source': 'test_document.pdf',
        'page_num': 2
    }
]


def process_pdf_data(retriever_instance):
    try:
        from src.ingestion.pdf_processor import PDFProcessor
        from src.ingestion.chunker import Chunker
        import glob

        processor = PDFProcessor()
        chunker = Chunker()

        pdf_files = glob.glob("data/documents/*.pdf")
        all_chunks = []

        # Check cache first
        import pickle
        cache_file = "data/processed_data/processed_chunks.pkl"

        if os.path.exists(cache_file):
            print(f"[INFO] Loading from cache: {cache_file}")
            with open(cache_file, 'rb') as f:
                all_chunks = pickle.load(f)
            retriever_instance.build_index(all_chunks, index_file="data/faiss.index")
            print(f"[OK] Loaded {len(all_chunks)} chunks from cache")
        elif pdf_files:
            print(f"[INFO] Processing {len(pdf_files)} PDF files...")

            for pdf_file in pdf_files:
                try:
                    print(f"  - Processing: {os.path.basename(pdf_file)}")
                    text = processor.extract_text(pdf_file)
                    chunks = chunker.chunk_text(text, source=pdf_file)
                    all_chunks.extend(chunks)
                    print(f"    - Extracted {len(chunks)} chunks")
                except Exception as e:
                    print(f"[WARN] Failed to process {pdf_file}: {e}")

            if all_chunks:
                retriever_instance.build_index(all_chunks, index_file="data/processed_data/faiss.index")
                with open(cache_file, 'wb') as f:
                    pickle.dump(all_chunks, f)
                print(f"[OK] Processed and cached {len(all_chunks)} chunks")
            else:
                print("[WARN] No chunks extracted from PDFs, using dummy data")
                raise Exception("No PDF content processed")
        else:
            print("[WARN] No PDF files found in data/documents/, using dummy data")
            raise Exception("No PDF files found")

    except ImportError as e:
        print(f"[WARN] PDF processing not available: {e}")
        print("   Using dummy data instead")
        raise Exception("PDF processing unavailable")

    except Exception as e:
        print(f"Falling back to dummy data: {e}")
        retriever_instance.build_index(DUMMY_CHUNKS)
        print("[OK] Retriever initialized with dummy data")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever, llm_generator

    # Initializing retriever and LLM generator
    try:
        retriever = HybridRetriever(use_rerank=True)
        process_pdf_data(retriever)
    except Exception as e:
        print(f"[WARN] HybridRetriever initialization failed: {e}")

    try:
        llm_generator = LLMGenerator()
        print("[OK] LLM Generator initialized")
    except Exception as e:
        print(f"[WARN] LLM Generator initialization failed: {e}")
        admin_control["use_llm"] = False

    # If server is running
    yield

    print("[SHUTDOWN] Shutting down the system...")


app = FastAPI(
    title="Graph-RAG Q&A API",
    version="1.0",
    lifespan=lifespan
)


class QueryRequest(BaseModel):
    query: str
    k: int = 10
    use_graph: bool = True
    use_rerank: bool = True
    alpha: float = 0.5
    bypass_cache: bool = False


class QueryResponse(BaseModel):
    query: str
    results: List[Dict]
    total_results: int
    search_time_ms: float


class AskRequest(BaseModel):
    query: str
    k: int = 10
    use_graph: bool = True
    use_rerank: bool = True
    max_context_chunks: int = 3
    model: Optional[str] = "gpt-5-nano"
    temperature: Optional[float] = 1
    stream: bool = True


class AskResponse(BaseModel):
    query: str
    answer: str
    sources: List[Dict]
    total_sources: int
    retrieval_time_ms: float
    generation_time_ms: float
    total_time_ms: float
    model: str
    tokens_used: int


@app.post("/search", response_model=QueryResponse)
def search(request: QueryRequest):
    """Search endpoint with SLO monitoring"""
    start_time = time.time()

    if not retriever:
        raise HTTPException(status_code=503, detail="Retriever not initialized")

    try:
        # Make sure the admin allows
        use_rerank = request.use_rerank and admin_control.get("use_rerank", True)
        use_graph = request.use_graph and admin_control.get("use_graph", True)

        results = retriever.search(
            request.query, 
            k=request.k, 
            use_rerank=use_rerank, 
            use_graph=use_graph,
            bypass_cache=request.bypass_cache
        )

        formatted_results = []
        for chunk, score in results:
            formatted_results.append({
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "source": chunk["source"],
                "page_num": chunk["page_num"],
                "score": float(score)
            })

        search_time_ms = (time.time() - start_time) * 1000

        # Record metrics with labels
        REQUEST_COUNT.labels(endpoint='search', status='200').inc()
        REQUEST_DURATION.labels(endpoint='search').observe(search_time_ms / 1000)

        return QueryResponse(
            query=request.query,
            results=formatted_results,
            total_results=len(formatted_results),
            search_time_ms=search_time_ms
        )

    except Exception as e:
        REQUEST_COUNT.labels(endpoint='search', status='500').inc()
        ERROR_COUNT.labels(endpoint='search', error_type=type(e).__name__).inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    """Health check endpoint"""
    status = {
        "status": "healthy",
        "retriever": retriever is not None,
        "llm_generator": llm_generator is not None,
        "admin_control": admin_control
    }
    return status


@app.get("/config")
def get_config():
    """Get current model configuration"""
    return {
        "model": "gpt-5-nano",
        "temperature": 1,
        "max_tokens": llm_generator.max_tokens if llm_generator else 3000
    }


@app.get("/metrics")
def metrics():
    """Prometheus metrics endpoint"""
    return Response(content=generate_latest(), media_type="text/plain")


@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the home page"""
    html_path = os.path.join(os.path.dirname(__file__), "..", "html/homepage.html")
    try:
        with open(html_path, "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>index.html not found</h1>", status_code=404)


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """Ask endpoint with LLM-generated answers"""
    total_start = time.time()

    if not retriever:
        raise HTTPException(status_code=503, detail="Retriever not initialized")

    if not llm_generator or not admin_control.get("use_llm", True):
        raise HTTPException(status_code=503, detail="LLM Generator not available")

    try:
        # Step 1: Retrieve relevant chunks
        retrieval_start = time.time()
        use_rerank = request.use_rerank and admin_control.get("use_rerank", True)
        use_graph = request.use_graph and admin_control.get("use_graph", True)

        retrieved_chunks = retriever.search(request.query, k=request.k, use_rerank=use_rerank, use_graph=use_graph)
        retrieval_time_ms = (time.time() - retrieval_start) * 1000

        # Step 2: Generate answer with LLM
        generation_start = time.time()

        result = llm_generator.generate_answer(
            query=request.query,
            retrieved_chunks=retrieved_chunks,
            model=request.model,
            temperature=request.temperature,
            max_context_chunks=request.max_context_chunks
        )

        generation_time_ms = (time.time() - generation_start) * 1000
        total_time_ms = (time.time() - total_start) * 1000

        # Record metrics with labels
        LLM_REQUEST_COUNT.labels(model=request.model, status='success').inc()
        LLM_REQUEST_DURATION.labels(model=request.model).observe(total_time_ms / 1000)
        LLM_TOKENS_USED.labels(model=request.model).inc(result["tokens_used"])

        REQUEST_COUNT.labels(endpoint='ask', status='200').inc()
        REQUEST_DURATION.labels(endpoint='ask').observe(total_time_ms / 1000)

        return AskResponse(
            query=request.query,
            answer=result["answer"],
            sources=result["sources"],
            total_sources=len(result["sources"]),
            retrieval_time_ms=retrieval_time_ms,
            generation_time_ms=generation_time_ms,
            total_time_ms=total_time_ms,
            model=result["model"],
            tokens_used=result["tokens_used"]
        )

    except Exception as e:
        REQUEST_COUNT.labels(endpoint='ask', status='500').inc()
        ERROR_COUNT.labels(endpoint='ask', error_type=type(e).__name__).inc()
        LLM_REQUEST_COUNT.labels(model=request.model, status='error').inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask/stream")
def ask_stream(request: AskRequest):
    """Streaming ask endpoint with metadata at the end"""
    if not retriever or not llm_generator:
        raise HTTPException(status_code=503, detail="Services not initialized")

    async def stream_with_metadata():
        try:
            # Track timing
            total_start = time.time()

            # Retrieve chunks
            retrieval_start = time.time()
            use_rerank = request.use_rerank and admin_control.get("use_rerank", True)
            use_graph = request.use_graph and admin_control.get("use_graph", True)
            retrieved_chunks = retriever.search(request.query, k=request.k, use_rerank=use_rerank, use_graph=use_graph)
            retrieval_time_ms = (time.time() - retrieval_start) * 1000

            # Track generation time
            generation_start = time.time()
            token_count = 0

            # Stream the answer text
            for chunk in llm_generator.generate_streaming(
                    query=request.query,
                    retrieved_chunks=retrieved_chunks,
                    model=request.model,
                    temperature=request.temperature,
                    max_context_chunks=request.max_context_chunks
            ):
                yield chunk
                token_count += len(chunk) // 4

            generation_time_ms = (time.time() - generation_start) * 1000
            total_time_ms = (time.time() - total_start) * 1000

            # Record metrics
            LLM_REQUEST_COUNT.labels(model=request.model, status='success').inc()
            LLM_REQUEST_DURATION.labels(model=request.model).observe(total_time_ms / 1000)
            LLM_TOKENS_USED.labels(model=request.model).inc(token_count)

            REQUEST_COUNT.labels(endpoint='ask_stream', status='200').inc()
            REQUEST_DURATION.labels(endpoint='ask_stream').observe(total_time_ms / 1000)

            # After streaming completes, send metadata as JSON
            # Use a special delimiter so frontend knows this is metadata
            metadata = {
                "type": "metadata",
                "sources": [
                    {
                        "source": doc_chunk['source'],
                        "page_num": doc_chunk['page_num'],
                        "chunk_id": doc_chunk['chunk_id'],
                        "score": float(score),
                        "text_preview": doc_chunk['text'][:200] + "..." if len(doc_chunk['text']) > 200 else doc_chunk[
                            'text']
                    }
                    for doc_chunk, score in retrieved_chunks[:request.max_context_chunks]
                ],
                "retrieval_time_ms": retrieval_time_ms,
                "generation_time_ms": generation_time_ms,
                "total_time_ms": total_time_ms,
                "model": request.model,
                "tokens_used": token_count
            }

            # Ensure no extra whitespace
            metadata_json = json.dumps(metadata, ensure_ascii=False)
            yield f"\n\n__METADATA__\n{metadata_json}"

        except Exception as e:
            REQUEST_COUNT.labels(endpoint='ask_stream', status='500').inc()
            ERROR_COUNT.labels(endpoint='ask_stream', error_type=type(e).__name__).inc()
            LLM_REQUEST_COUNT.labels(model=request.model, status='error').inc()
            yield f"\n\nError: {str(e)}"

    return StreamingResponse(
        stream_with_metadata(),
        media_type="text/plain"
    )


@app.post("/feature-flags/{flag_name}")
def toggle_feature_flag(flag_name: str, enabled: bool):
    """Toggle feature flags for A/B testing"""
    if flag_name in admin_control:
        admin_control[flag_name] = enabled
        return {"flag": flag_name, "enabled": enabled}
    raise HTTPException(status_code=404, detail="Feature flag not found")


if __name__ == "__main__":
    import uvicorn
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        print("[TEST] Testing FastAPI app initialization...")

        import asyncio

        async def test_lifespan():
            async with lifespan(app):
                # Test a simple search
                if retriever:
                    results = retriever.search("What is this document about?", k=2)
                    print(f"[OK] Search test: found {len(results)} results")
                    for chunk, score in results:
                        print(f"   - {chunk['text'][:50]}... (score: {score:.3f})")

                print("[OK] FastAPI app test completed - exiting")


        asyncio.run(test_lifespan())
        sys.exit(0)
    else:
        uvicorn.run(app, host="0.0.0.0", port=8000)
