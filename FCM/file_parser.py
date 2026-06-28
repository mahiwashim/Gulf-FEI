from __future__ import annotations

import io
from pathlib import Path

from docx import Document
from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}
MAX_FILE_SIZE_BYTES = 12 * 1024 * 1024


class FileParsingError(ValueError):
    pass


def _parse_pdf(content: bytes) -> str:
    reader = PdfReader(io.BytesIO(content))
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    return "\n\n".join(p for p in pages if p)


def _parse_docx(content: bytes) -> str:
    doc = Document(io.BytesIO(content))
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)


def _parse_txt(content: bytes) -> str:
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return content.decode(encoding).strip()
        except UnicodeDecodeError:
            continue
    raise FileParsingError("Unable to decode text file. Use UTF-8, UTF-16, or Latin-1 encoding.")


def parse_uploaded_file(filename: str, content: bytes) -> str:
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise FileParsingError(f"Unsupported file type for '{filename}'. Allowed: PDF, DOCX, TXT.")

    if not content:
        raise FileParsingError(f"File '{filename}' is empty.")

    if len(content) > MAX_FILE_SIZE_BYTES:
        raise FileParsingError(
            f"File '{filename}' exceeds the 12 MB limit. Please split large files into smaller parts."
        )

    try:
        if ext == ".pdf":
            text = _parse_pdf(content)
        elif ext == ".docx":
            text = _parse_docx(content)
        else:
            text = _parse_txt(content)
    except FileParsingError:
        raise
    except Exception as exc:
        raise FileParsingError(f"Could not read '{filename}'. The file may be corrupted or unsupported.") from exc

    cleaned = text.strip()
    if not cleaned:
        raise FileParsingError(f"No extractable text found in '{filename}'.")

    return cleaned
