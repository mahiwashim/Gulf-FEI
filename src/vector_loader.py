import os
import json
import pickle
import logging
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
import faiss
import pandas as pd
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.docstore.in_memory import InMemoryDocstore

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load .env
load_dotenv()

# Config
BASE_DIR = Path(__file__).resolve().parent.parent
VECTOR_DB_DIR = BASE_DIR / "vector_db"

# Native build format (produced by Research/build_faiss_rag.ipynb)
NATIVE_INDEX_PATH   = VECTOR_DB_DIR / "index.faiss"
NATIVE_PARQUET_PATH = VECTOR_DB_DIR / "vector_df.parquet"
BUILD_CONFIG_PATH   = VECTOR_DB_DIR / "build_config.json"

# Legacy LangChain-serialized format
LEGACY_INDEX_PATH = VECTOR_DB_DIR / "index.faiss"
INDEX_DB_PATH     = VECTOR_DB_DIR / "index.pkl"

# Legacy chunk/source pickle format
IDX_PATH     = VECTOR_DB_DIR / "faiss_index.idx"
CHUNKS_PATH  = VECTOR_DB_DIR / "chunks.pkl"
SOURCES_PATH = VECTOR_DB_DIR / "sources.pkl"

# Default embedding model — overridden by build_config.json when present.
# IMPORTANT: this MUST match the model used to build the index, otherwise
# query vectors live in a different space and retrieval returns noise.
DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# Columns in vector_df.parquet carried onto each chunk as metadata.
_TEXT_COLUMN = "text"
_META_COLUMNS = [
    "chunk_id", "Document_ID", "chunk_index", "Source_Type", "Source_Name",
    "Source_Folder", "Title", "URL", "Published_Date", "Location",
]


def _resolve_model_name() -> str:
    """Read the embedding model from build_config.json so the query encoder
    always matches the one that produced the index."""
    if BUILD_CONFIG_PATH.exists():
        try:
            cfg = json.loads(BUILD_CONFIG_PATH.read_text(encoding="utf-8"))
            name = cfg.get("model_name")
            if name:
                return str(name)
        except Exception as exc:
            logger.warning("Could not read build_config.json: %s", exc)
    return DEFAULT_EMBED_MODEL


class VectorStoreLoader:
    """
    Loads the Gulf FEI FAISS vector store.

    Primary path: the native build (``index.faiss`` + ``vector_df.parquet``)
    produced by ``Research/build_faiss_rag.ipynb``. Falls back to the legacy
    LangChain-serialized store or the older chunk/source pickle layout.
    """

    def __init__(self, embeddings_model: Optional[str] = None):
        self.model_name = embeddings_model or _resolve_model_name()
        # normalize_embeddings=True → query vectors are L2-normalized, so the
        # inner-product (IndexFlatIP) index returns true cosine similarity.
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                encode_kwargs={"normalize_embeddings": True},
            )
        except Exception as exc:
            # No / flaky network: the Hugging Face online "update check" fails
            # even though the model is already cached. Retry strictly from the
            # local cache so a network drop never crashes startup.
            logger.warning(
                "Online embedding-model load failed (%s); retrying offline from cache.",
                exc,
            )
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={"local_files_only": True},
                encode_kwargs={"normalize_embeddings": True},
            )
        logger.info("Embeddings model: %s", self.model_name)

    # ------------------------------------------------------------------
    def load(self) -> FAISS:
        # 1) Native build: index.faiss + vector_df.parquet
        if NATIVE_INDEX_PATH.exists() and NATIVE_PARQUET_PATH.exists():
            return self._load_native()

        # 2) Legacy LangChain-serialized store: index.faiss + index.pkl
        if LEGACY_INDEX_PATH.exists() and INDEX_DB_PATH.exists():
            try:
                return FAISS.load_local(
                    str(VECTOR_DB_DIR), self.embeddings,
                    allow_dangerous_deserialization=True,
                )
            except Exception as exc:
                logger.warning("FAISS.load_local failed: %s — manual fallback", exc)
                return self._load_from_saved_store()

        # 3) Legacy chunk/source pickles
        if all(p.exists() for p in (IDX_PATH, CHUNKS_PATH, SOURCES_PATH)):
            return self._load_from_chunks_sources()

        raise FileNotFoundError(
            "Vector DB files missing. Expected vector_db/index.faiss + "
            "vector_db/vector_df.parquet (native build), or a legacy "
            "index.faiss + index.pkl, or faiss_index.idx + chunks.pkl + sources.pkl."
        )

    # ------------------------------------------------------------------
    def _load_native(self) -> FAISS:
        """Load the native (notebook) build: FAISS index + parquet metadata."""
        index = faiss.read_index(str(NATIVE_INDEX_PATH))
        df = pd.read_parquet(NATIVE_PARQUET_PATH)
        logger.info("Loaded vector_df.parquet: %d chunks, columns=%s",
                    len(df), list(df.columns))

        if index.ntotal != len(df):
            logger.warning(
                "Index/parquet size mismatch: index=%d, parquet=%d. "
                "Using positional alignment up to the smaller of the two.",
                index.ntotal, len(df),
            )

        text_col = _TEXT_COLUMN if _TEXT_COLUMN in df.columns else df.columns[-1]
        meta_cols = [c for c in _META_COLUMNS if c in df.columns]

        documents = {}
        index_to_docstore_id = {}
        for i, row in enumerate(df.itertuples(index=False)):
            rec = row._asdict()
            content = str(rec.get(text_col, "") or "")
            metadata = {c: rec.get(c, "") for c in meta_cols}
            # Friendly fields the RAG chain / source extractor look for.
            metadata["source"] = (
                rec.get("Title")
                or rec.get("Source_Name")
                or rec.get("Source_Type")
                or rec.get("Document_ID")
                or "unknown"
            )
            metadata["title"] = rec.get("Title", "")
            metadata["url"] = rec.get("URL", "")
            metadata["source_type"] = rec.get("Source_Type", "")
            documents[str(i)] = Document(page_content=content, metadata=metadata)
            index_to_docstore_id[i] = str(i)

        docstore = InMemoryDocstore(documents)
        vector_store = FAISS(
            embedding_function=self.embeddings,
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id,
            normalize_L2=False,   # vectors already normalized at build time
            distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,  # IndexFlatIP = cosine
        )
        logger.info("Native vector store ready: %d vectors, %d documents.",
                    index.ntotal, len(documents))
        return vector_store

    # ------------------------------------------------------------------
    def _load_from_saved_store(self) -> FAISS:
        index = faiss.read_index(str(LEGACY_INDEX_PATH))
        with open(INDEX_DB_PATH, "rb") as f:
            payload = pickle.load(f)

        if isinstance(payload, tuple) and len(payload) == 2:
            docstore, index_to_docstore_id = payload
        elif isinstance(payload, dict) and "docstore" in payload:
            docstore = payload["docstore"]
            index_to_docstore_id = payload["index_to_docstore_id"]
        else:
            raise ValueError("Unsupported index.pkl format.")

        vector_store = FAISS(
            embedding_function=self.embeddings,
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id,
        )
        logger.info("Loaded legacy store with %d vectors.", index.ntotal)
        return vector_store

    # ------------------------------------------------------------------
    def _load_from_chunks_sources(self) -> FAISS:
        index = faiss.read_index(str(IDX_PATH))
        with open(CHUNKS_PATH, "rb") as f:
            chunks: List[str] = pickle.load(f)
        with open(SOURCES_PATH, "rb") as f:
            sources: List[str] = pickle.load(f)

        if len(chunks) != len(sources) or index.ntotal != len(chunks):
            raise ValueError("Mismatch between index size, chunks, and sources.")

        documents = [
            Document(page_content=chunk, metadata={"source": src})
            for chunk, src in zip(chunks, sources)
        ]
        docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(documents)})
        index_to_docstore_id = {i: str(i) for i in range(len(documents))}

        vector_store = FAISS(
            embedding_function=self.embeddings,
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id,
        )
        logger.info("Loaded legacy chunk/source store with %d vectors.", index.ntotal)
        return vector_store
