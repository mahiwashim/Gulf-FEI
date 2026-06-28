from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class StoredDoc:
    doc_id: str
    text: str
    metadata: Dict[str, str] = field(default_factory=dict)


class TfidfRetriever:
    def __init__(self) -> None:
        self._docs: List[StoredDoc] = []
        self._vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        self._matrix = None

    def ingest(self, docs: List[StoredDoc]) -> None:
        cleaned_docs = [
            StoredDoc(
                doc_id=doc.doc_id,
                text=doc.text.strip(),
                metadata=doc.metadata,
            )
            for doc in docs
            if doc.text and doc.text.strip()
        ]
        if not cleaned_docs:
            raise ValueError('At least one document with non-empty text is required for ingestion.')

        self._docs.extend(cleaned_docs)
        corpus = [d.text for d in self._docs if d.text.strip()]
        if not corpus:
            raise ValueError('Unable to build a retriever index from empty documents.')
        self._matrix = self._vectorizer.fit_transform(corpus)

    def search(self, query: str, top_k: int = 5) -> List[dict]:
        cleaned_query = query.strip()
        if not cleaned_query or top_k <= 0:
            return []
        if not self._docs or self._matrix is None:
            return []
        q = self._vectorizer.transform([cleaned_query])
        scores = cosine_similarity(q, self._matrix)[0]
        ranked_idx = scores.argsort()[::-1][: min(top_k, len(self._docs))]
        results = []
        for idx in ranked_idx:
            if float(scores[idx]) <= 0:
                continue
            doc = self._docs[int(idx)]
            results.append(
                {
                    'doc_id': doc.doc_id,
                    'score': float(scores[idx]),
                    'text': doc.text,
                    'metadata': doc.metadata,
                }
            )
        return results
