"""
Ensure the FAISS vector store is present locally.

The vector artifacts are ~1.6 GB — far too large for GitHub — so they live on
Google Drive and are downloaded automatically on first run. Only the two files
the app actually needs at runtime are fetched:

    • index.faiss        (the FAISS index)
    • vector_df.parquet  (chunk text + metadata)

`vectors.npy` is skipped by default (it is only needed to *rebuild* the index,
not to run the app). `build_config.json` is tiny and ships in the git repo.

Default Google Drive IDs are baked in (from the shared folder
https://drive.google.com/drive/folders/1u8ggjd3rnwhXZt7rwoksj-QEsG-GXqre), so a
fresh clone downloads everything automatically with NO .env edits. You can
override any of them, or point at your own copy, via .env (see .env.example):

    GDRIVE_INDEX_FAISS_ID=...        # overrides index.faiss
    GDRIVE_VECTOR_DF_ID=...          # overrides vector_df.parquet
    GDRIVE_VECTORS_NPY_ID=...        # optional, only to also fetch vectors.npy
    GDRIVE_VECTOR_DB_FOLDER_ID=...   # optional: resolve IDs from a shared folder
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
VECTOR_DB_DIR = BASE_DIR / "vector_db"

# Shared folder (used only as a fallback to resolve any missing file IDs).
DEFAULT_FOLDER_ID = "1u8ggjd3rnwhXZt7rwoksj-QEsG-GXqre"

# Per-file artifacts:
#   (filename, env-var override, baked-in Drive file ID, min sane bytes, required?)
_ARTIFACTS = [
    ("index.faiss",       "GDRIVE_INDEX_FAISS_ID", "1VjUllghRcfRvYM00Ngg3P91BkWIImSTU", 10_000_000, True),
    ("vector_df.parquet", "GDRIVE_VECTOR_DF_ID",   "1iGCWrPDgzVKxx9XbsbCxAmjp8z_k7lzh",  1_000_000, True),
    # Optional — only fetched if you set GDRIVE_VECTORS_NPY_ID (rebuild only).
    ("vectors.npy",       "GDRIVE_VECTORS_NPY_ID", "",                                  10_000_000, False),
]


def _present(path: Path, min_bytes: int) -> bool:
    return path.exists() and path.stat().st_size >= min_bytes


def _required_ready() -> bool:
    return all(
        _present(VECTOR_DB_DIR / name, min_bytes)
        for name, _, _, min_bytes, required in _ARTIFACTS
        if required
    )


def ensure_vector_db() -> bool:
    """Make sure the runtime vector files exist, downloading them if needed.

    Returns True once the required files are present. Raises a clear error if a
    required file is missing and no Drive ID can be resolved for it.
    """
    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

    if _required_ready():
        logger.info("✓ Vector store already present in %s", VECTOR_DB_DIR)
        return True

    try:
        import gdown
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "gdown is required to auto-download the vector store. "
            "Run: pip install gdown  (it is also in requirements.txt)."
        ) from exc

    # Lazily resolve IDs from a shared folder only if some file lacks an ID.
    folder_id = os.getenv("GDRIVE_VECTOR_DB_FOLDER_ID") or DEFAULT_FOLDER_ID
    _folder_ids: dict[str, "str | None"] = {}

    def _from_folder(name: str) -> str | None:
        nonlocal _folder_ids
        if not _folder_ids and folder_id:
            try:
                items = gdown.download_folder(
                    id=folder_id, skip_download=True, quiet=True, use_cookies=False
                ) or []
                _folder_ids = {
                    Path(getattr(it, "path", "")).name: getattr(it, "id", None)
                    for it in items
                }
            except Exception as exc:  # pragma: no cover
                logger.warning("Could not list Drive folder: %s", exc)
                _folder_ids = {"__failed__": ""}
        return _folder_ids.get(name)

    for name, env, default_id, min_bytes, required in _ARTIFACTS:
        dest = VECTOR_DB_DIR / name
        if _present(dest, min_bytes):
            continue

        file_id = os.getenv(env) or default_id or _from_folder(name)
        if not file_id:
            if required:
                raise RuntimeError(
                    f"{name} is missing and no Drive ID could be resolved "
                    f"(set {env} in .env or check folder sharing)."
                )
            logger.info("• Skipping optional %s (no %s set)", name, env)
            continue

        logger.info("⬇ Downloading %s from Google Drive …", name)
        gdown.download(f"https://drive.google.com/uc?id={file_id}", str(dest), quiet=False)

    if not _required_ready():
        raise RuntimeError(
            "Vector store download finished but required files are still missing "
            "or too small. Check Drive sharing ('Anyone with the link') and the "
            "IDs in .env / download_vector_db.py."
        )

    logger.info("✓ Vector store ready in %s", VECTOR_DB_DIR)
    return True


if __name__ == "__main__":
    ensure_vector_db()
