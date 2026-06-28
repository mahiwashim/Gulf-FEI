import csv
import io
import logging
import os
import re
import time
from collections.abc import Iterable
from contextlib import asynccontextmanager
from typing import Any, List

# ── macOS multiprocessing / deadlock guards ─────────────────────────────────
# MUST run before importing torch / transformers / sentence-transformers / sklearn.
# On macOS (spawn start method) the HuggingFace `tokenizers` library deadlocks if
# its parallelism is active when the process forks — the request silently hangs
# with no error. loky/joblib (sklearn) can hang the same way. Disabling these is
# the standard fix and costs ~nothing for our small per-query workloads.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")

import matplotlib
matplotlib.use("Agg")

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from FCM.extractor import STOPWORDS, _USELESS_MODIFIERS, _VAGUE_SINGLES
from FCM.simulator import FCMSimulator
from src.FCM import FCMGenerator
from src.rag import create_rag_chain
from src.vector_loader import VectorStoreLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not set in .env")

vector_store = None
_fcm_generator = None
_last_fcm: dict = {"concepts": [], "matrix": []}

# Tokens that don't carry research meaning. Used to reject queries that are
# pure filler ("how is the", "what about it", "yes please") so we don't burn
# the RAG + FCM pipeline producing an empty graph.
_QUERY_NOISE_TOKENS = STOPWORDS | _USELESS_MODIFIERS | _VAGUE_SINGLES | {
    'please', 'thanks', 'hi', 'hello', 'hey', 'ok', 'okay', 'test', 'testing',
}
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]+")


def _is_meaningful_query(q: str) -> bool:
    """Return False if the query is empty, too short, or carries no real terms.

    A meaningful query must contain at least one alphabetic token of length >= 3
    that is not a stopword / filler / interrogative. This blocks junk like
    "asdf", "the the", "yes please", "how is it" before they hit the pipeline.
    """
    if not q or not q.strip():
        return False
    tokens = [t.lower() for t in _WORD_RE.findall(q)]
    if not tokens:
        return False
    meaningful = [t for t in tokens if len(t) >= 3 and t not in _QUERY_NOISE_TOKENS]
    return bool(meaningful)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_store, _fcm_generator
    logger.info("Starting up application...")
    try:
        # The vector store must already be downloaded. Fetch it once with:
        #     python download_vector_db.py
        from download_vector_db import _required_ready, VECTOR_DB_DIR
        if not _required_ready():
            logger.error(
                "Vector store not found in %s.\n"
                "    Download it first (one time):  python download_vector_db.py",
                VECTOR_DB_DIR,
            )
        loader = VectorStoreLoader()
        vector_store = loader.load()
        logger.info("Vector store loaded successfully")
    except Exception as exc:
        logger.error("Failed to load vector store: %s", exc, exc_info=True)
        vector_store = None
    # Pre-build FCMGenerator once so pattern compilation + imports happen at startup
    # edge_threshold = 0.20 keeps the graph rich (typical pattern edges land
    # at 0.55-0.75, co-occurrence at 0.30-0.70) while still filtering the
    # bottom-tier noise; max_concepts = 25 lets the substantive-content gate
    # surface the full set of legitimate concepts a document mentions.
    _fcm_generator = FCMGenerator(max_concepts=25, edge_threshold=0.20, directed=True)
    logger.info("FCMGenerator initialized")
    yield


app = FastAPI(
    title="Gulf FEI — Perception & Discourse Intelligence",
    description=(
        "Interactive Gulf FEI interface that mines public perception of Gulf of Mexico "
        "fisheries from blogs, forums, YouTube, and podcasts — combining RAG answers with "
        "Fuzzy Cognitive Maps of the community's perceived cause-and-effect relationships."
    ),
    lifespan=lifespan,
)

# Ensure the mountable directories exist on a fresh clone before mounting.
os.makedirs("static", exist_ok=True)
os.makedirs("logo", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/logo",   StaticFiles(directory="logo"),   name="logo")

templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request, name="index.html", context={"request": request}
    )


@app.get("/health")
async def health_check():
    return {"status": "healthy", "vector_store_loaded": vector_store is not None}


# ── File parsers ──────────────────────────────────────────────────────────────

def _parse_docx(content: bytes) -> str:
    try:
        from docx import Document
        doc = Document(io.BytesIO(content))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception as exc:
        logger.warning("DOCX parse failed: %s", exc)
        return ""


def _parse_pdf(content: bytes) -> str:
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(content))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as exc:
        logger.warning("PDF parse failed: %s", exc)
        return ""


def _parse_adjacency_csv(content: bytes):
    """
    Return (concepts, matrix) if the CSV looks like an adjacency matrix
    (first header cell is blank, row count == col count).
    Otherwise return (None, None).
    """
    try:
        text   = content.decode("utf-8-sig", errors="replace")
        reader = csv.reader(io.StringIO(text))
        rows   = [r for r in reader if any(c.strip() for c in r)]
        if len(rows) < 2:
            return None, None
        header = rows[0]
        # Adjacency matrix: top-left cell is empty
        if header[0].strip():
            return None, None
        concepts = [h.strip() for h in header[1:] if h.strip()]
        if not concepts:
            return None, None
        matrix: List[List[float]] = []
        for row in rows[1: len(concepts) + 1]:
            vals: List[float] = []
            for v in row[1: len(concepts) + 1]:
                try:
                    vals.append(float(v.strip()) if v.strip() else 0.0)
                except ValueError:
                    vals.append(0.0)
            while len(vals) < len(concepts):
                vals.append(0.0)
            matrix.append(vals[: len(concepts)])
        if len(matrix) == len(concepts):
            return concepts, matrix
        return None, None
    except Exception as exc:
        logger.warning("CSV adjacency parse failed: %s", exc)
        return None, None


def _consolidate_matrices(matrices: List[tuple]) -> tuple:
    """Merge multiple (concepts, matrix) pairs into a single master matrix.

    Mirrors the R pipeline: build the union of node labels, expand each
    matrix to that master space (zero-padding missing rows/cols), then sum.
    """
    if not matrices:
        return [], []
    # Master node set = union of all concepts
    master_nodes: List[str] = []
    seen = set()
    for concepts, _ in matrices:
        for c in concepts:
            if c not in seen:
                seen.add(c)
                master_nodes.append(c)
    master_nodes.sort()
    idx = {c: i for i, c in enumerate(master_nodes)}
    n = len(master_nodes)
    master = [[0.0] * n for _ in range(n)]
    for concepts, matrix in matrices:
        for i, src in enumerate(concepts):
            for j, tgt in enumerate(concepts):
                try:
                    v = float(matrix[i][j])
                except (IndexError, TypeError, ValueError):
                    v = 0.0
                if v:
                    master[idx[src]][idx[tgt]] += v
    return master_nodes, master


def _parse_xlsx(content: bytes):
    """Parse Excel adjacency matrix (same layout as CSV version)."""
    try:
        import openpyxl
        wb = openpyxl.load_workbook(io.BytesIO(content), data_only=True)
        ws = wb.active
        rows = [[str(cell.value or "").strip() for cell in row] for row in ws.iter_rows()]
        rows = [r for r in rows if any(c for c in r)]
        if not rows:
            return None, None
        # Re-use CSV logic via a text round-trip
        csv_text = "\n".join(",".join(r) for r in rows).encode("utf-8")
        return _parse_adjacency_csv(csv_text)
    except Exception as exc:
        logger.warning("XLSX parse failed: %s", exc)
        return None, None


# ── Shared helpers ────────────────────────────────────────────────────────────

def _get_relevant_documents(store: Any, query: str):
    retriever = store.as_retriever(search_kwargs={"k": 6})
    for method_name in ("invoke", "get_relevant_documents", "retrieve"):
        method = getattr(retriever, method_name, None)
        if callable(method):
            try:
                result = method(query)
                if isinstance(result, Iterable):
                    return list(result)
                if result is not None:
                    return [result]
            except Exception:
                logger.warning("Retriever method %s failed", method_name, exc_info=True)
    return []


def _extract_sources(docs: list) -> list:
    sources = set()
    for doc in docs:
        meta = getattr(doc, "metadata", None)
        if isinstance(meta, dict):
            sources.add(meta.get("source", "unknown"))
    return sorted(sources)


def _normalize_answer(answer: Any) -> str:
    if answer is None:
        return ""
    if isinstance(answer, str):
        return answer
    content = getattr(answer, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("text"):
                parts.append(str(item["text"]))
        return "\n".join(parts)
    return str(answer)


def _cache_fcm_for_simulation(details: dict) -> None:
    """Store the most recent (nodes, matrix) so /simulate-fcm can run on it."""
    nodes = details.get("nodes") or []
    matrix = details.get("adjacency_matrix") or []
    if nodes and matrix and len(matrix) == len(nodes):
        _last_fcm["concepts"] = list(nodes)
        _last_fcm["matrix"] = [list(row) for row in matrix]


def _build_response(details, edge_results, plot_path, answer="", sources=None, max_edges=10,
                    plot_path_summary=None):
    pos_n = sum(1 for e in edge_results if e.get("polarity") == "+")
    neg_n = sum(1 for e in edge_results if e.get("polarity") == "-")
    warnings: List[str] = []
    if edge_results and neg_n == 0:
        warnings.append(
            "FCM contains no negative relationships — a credible signed causal "
            "map normally includes both reinforcing (+) and opposing (-) links. "
            "Consider whether the source text genuinely describes only positive "
            "drivers, or if the extraction missed negative phrasings."
        )
    if edge_results and pos_n == 0:
        warnings.append(
            "FCM contains no positive relationships — unusual for a balanced "
            "causal map. Verify the source content."
        )
    return {
        "answer":             answer,
        "sources":            sources or [],
        "fcm_edge_results":   edge_results,
        "fcm_details":        details,
        "edges_meta":         details.get("edges_meta", []),
        "edge_types":         details.get("stats", {}).get("edge_types", {}),
        "plot_path":          plot_path,
        "plot_path_summary":  plot_path_summary,
        "max_edges":          max_edges,
        "relation_length":    len(edge_results),
        "positive_edges":     pos_n,
        "negative_edges":     neg_n,
        "warnings":           warnings,
        "adjacency_matrix":   details.get("adjacency_matrix", []),
        "matrix_concepts":    details.get("nodes", []),
    }


# ── RAG query endpoint ────────────────────────────────────────────────────────

def prepare_query_response(query: str, max_edges: int):
    if not query or not query.strip():
        return JSONResponse(status_code=400, content={"error": "Query text is required."})
    if not _is_meaningful_query(query):
        return JSONResponse(
            status_code=400,
            content={"error": (
                "Please enter a meaningful research question. The query appears "
                "to contain only filler or stopwords (e.g. 'the', 'how is it', "
                "'yes please') with no recognizable topic terms."
            )},
        )
    if vector_store is None:
        return JSONResponse(status_code=503,
                            content={"error": "Vector store not loaded. Please wait and retry."})

    q   = query.strip()
    lim = max(1, min(int(max_edges), 60))
    logger.info("Processing query: %s", q[:80])

    docs    = _get_relevant_documents(vector_store, q)
    logger.info("Retrieved %d documents; calling Groq LLM...", len(docs))
    sources = _extract_sources(docs)
    chain   = create_rag_chain(vector_store)
    answer  = _normalize_answer(chain.invoke(q))
    logger.info("LLM answer received (%d chars); building FCM...", len(answer))

    doc_context = "\n\n".join(
        getattr(doc, "page_content", "")
        for doc in docs if getattr(doc, "page_content", None)
    )
    fcm_input = f"Question: {q}\n\nAnswer:\n{answer}\n\nContext:\n{doc_context}".strip()

    fcm = _fcm_generator
    fcm.max_edges = lim
    graph, details = fcm.build_fcm(fcm_input, q, lim)

    edge_results = [
        {"result": i + 1, "cause": src, "effect": tgt,
         "polarity": "+" if w >= 0 else "-", "weight": round(abs(w), 2)}
        for i, (src, tgt, w) in enumerate(details["edges"])
    ]

    timestamp         = int(time.time())
    plot_path         = fcm.plot_fcm(graph, f"fcm_plot_{timestamp}")
    details.pop("top10_graph", None)
    _cache_fcm_for_simulation(details)
    return _build_response(details, edge_results, plot_path, answer,
                           sources, lim)


@app.post("/query")
async def query_rag(query: str = Form(...), max_edges: int = Form(5)):
    try:
        return prepare_query_response(query, max_edges)
    except Exception as exc:
        logger.exception("Error in /query: %s", exc)
        return JSONResponse(status_code=500,
                            content={"error": f"Failed to process query: {exc}"})


@app.post("/api/query")
async def api_query_rag(payload: dict):
    try:
        query     = payload.get("question") or payload.get("query")
        max_edges = payload.get("relation_length") or payload.get("max_edges") or 5
        if not query:
            return JSONResponse(status_code=400, content={"error": "Missing query text."})
        return prepare_query_response(query, int(max_edges))
    except Exception as exc:
        logger.exception("Error in /api/query: %s", exc)
        return JSONResponse(status_code=500,
                            content={"error": f"Failed to process API query: {exc}"})


# ── File-upload endpoint ──────────────────────────────────────────────────────

@app.post("/upload-files")
async def upload_files(
    files: List[UploadFile] = File(...),
    max_edges: int = Form(10),
):
    """
    Accept up to 20 Word / PDF / TXT / CSV(adjacency) / XLSX files.
    - If any file is a valid adjacency-matrix CSV/XLSX it is used directly.
    - Otherwise all text is combined and run through FCM extraction.
    """
    if not files:
        return JSONResponse(status_code=400, content={"error": "No files uploaded."})

    files = files[:20]   # hard cap
    lim   = max(1, min(int(max_edges), 60))

    all_text:        List[str]  = []
    matrix_list:     List[tuple] = []   # list of (concepts, matrix) — one per CSV/XLSX
    file_names:      List[str]  = []
    skipped:         List[str]  = []

    for upload in files:
        fname = upload.filename or "unnamed"
        ext   = fname.rsplit(".", 1)[-1].lower() if "." in fname else ""
        content = await upload.read()
        file_names.append(fname)

        if ext == "docx":
            txt = _parse_docx(content)
            if txt:
                all_text.append(txt)
            else:
                skipped.append(fname)

        elif ext == "pdf":
            txt = _parse_pdf(content)
            if txt:
                all_text.append(txt)
            else:
                skipped.append(fname)

        elif ext in ("txt", "md"):
            all_text.append(content.decode("utf-8", errors="replace"))

        elif ext == "csv":
            concepts, matrix = _parse_adjacency_csv(content)
            if concepts and matrix:
                matrix_list.append((concepts, matrix))
                logger.info("Loaded adjacency matrix from %s (%d concepts)", fname, len(concepts))
            else:
                all_text.append(content.decode("utf-8", errors="replace"))

        elif ext in ("xlsx", "xls"):
            concepts, matrix = _parse_xlsx(content)
            if concepts and matrix:
                matrix_list.append((concepts, matrix))
                logger.info("Loaded adjacency matrix from %s (%d concepts)", fname, len(concepts))
            else:
                skipped.append(fname)

        else:
            skipped.append(fname)

    fcm = _fcm_generator
    fcm.max_concepts   = 20
    fcm.edge_threshold = 0.10
    fcm.max_edges      = lim

    # ── Build from adjacency matrix if available ──────────────────────
    if matrix_list:
        # Consolidate ALL uploaded matrices (master-node expansion + sum)
        concepts, matrix = _consolidate_matrices(matrix_list)
        graph, details   = fcm.build_fcm_from_matrix(
            concepts, matrix,
            query=f"Consolidated from {len(matrix_list)} uploaded matri"
                  f"{'x' if len(matrix_list) == 1 else 'ces'}"
        )
        edge_results = [
            {"result": i + 1, "cause": src, "effect": tgt,
             "polarity": "+" if w >= 0 else "-", "weight": round(abs(w), 2)}
            for i, (src, tgt, w) in enumerate(details["edges"])
        ]
        answer = (
            f"FCM consolidated from {len(matrix_list)} uploaded adjacency "
            f"matri{'x' if len(matrix_list) == 1 else 'ces'} — "
            f"{len(concepts)} master nodes, {len(edge_results)} active edges."
        )
    elif all_text:
        combined = "\n\n---\n\n".join(all_text)
        graph, details = fcm.build_fcm(combined, "Uploaded document analysis", lim)
        edge_results = [
            {"result": i + 1, "cause": src, "effect": tgt,
             "polarity": "+" if w >= 0 else "-", "weight": round(abs(w), 2)}
            for i, (src, tgt, w) in enumerate(details["edges"])
        ]
        answer = (
            f"FCM extracted from {len(all_text)} uploaded file(s). "
            f"Found {len(edge_results)} causal relations across "
            f"{details['stats']['num_nodes']} concepts."
        )
    else:
        return JSONResponse(
            status_code=422,
            content={"error": "No readable content found in uploaded files.",
                     "skipped": skipped},
        )

    timestamp         = int(time.time())
    plot_path         = fcm.plot_fcm(graph, f"fcm_upload_{timestamp}")
    details.pop("top10_graph", None)
    _cache_fcm_for_simulation(details)

    response = _build_response(details, edge_results, plot_path, answer, [], lim)
    response["files_processed"] = [f for f in file_names if f not in skipped]
    response["files_skipped"]   = skipped
    return response


# ── FCM scenario simulation ──────────────────────────────────────────────────

@app.get("/fcm-concepts")
async def fcm_concepts():
    """Return the concept list of the most recently built FCM (for UI selector)."""
    return {
        "concepts": list(_last_fcm.get("concepts") or []),
        "has_fcm": bool(_last_fcm.get("concepts")),
    }


@app.post("/simulate-fcm")
async def simulate_fcm(payload: dict):
    """Run Kosko activation propagation on the cached FCM.

    Payload: {
        "activations": {"Concept A": 1.0, "Concept B": -0.5},
        "steps":        30,              # optional
        "squash":       "sigmoid",       # or "tanh" | "none"
        "lam":          1.0,             # sigmoid/tanh steepness
        "clamp_drivers": true            # keep drivers pinned each step
    }
    """
    concepts = list(_last_fcm.get("concepts") or [])
    matrix   = _last_fcm.get("matrix") or []
    if not concepts or not matrix:
        return JSONResponse(
            status_code=400,
            content={"error": "No FCM available. Build one by querying or uploading files first."},
        )

    activations = payload.get("activations") or {}
    if not activations:
        return JSONResponse(
            status_code=400,
            content={"error": "activations dict required, e.g. {'Overfishing': 1.0}."},
        )

    try:
        sim = FCMSimulator(concepts, matrix)
        result = sim.simulate(
            activation    = {str(k): float(v) for k, v in activations.items()},
            steps         = int(payload.get("steps", 30)),
            squash        = str(payload.get("squash", "sigmoid")),
            lam           = float(payload.get("lam", 1.0)),
            clamp_drivers = bool(payload.get("clamp_drivers", True)),
        )
    except Exception as exc:
        logger.exception("Simulation failed: %s", exc)
        return JSONResponse(status_code=500, content={"error": f"Simulation failed: {exc}"})

    return result


# ── Entry point ───────────────────────────────────────────────────────────────
# Allows `python main.py` to launch the server directly. On first run this will
# download the vector store from Google Drive (see download_vector_db.py).
if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    logger.info("Starting Gulf FEI on http://localhost:%d", port)
    uvicorn.run("main:app", host=host, port=port, reload=False)
