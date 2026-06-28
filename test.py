import os

from dotenv import load_dotenv

from src.rag import create_rag_chain
from src.vector_loader import VectorStoreLoader

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not set in .env")


def _load_vector_store():
    loader = VectorStoreLoader()
    return loader.load()


def _get_documents(vector_store, query: str):
    retriever = vector_store.as_retriever(search_kwargs={"k": 6})
    for method_name in ("invoke", "get_relevant_documents", "retrieve"):
        method = getattr(retriever, method_name, None)
        if callable(method):
            result = method(query)
            if result is not None:
                return result
    return []


def run_perfect_rag(query: str) -> dict:
    vector_store = _load_vector_store()
    chain = create_rag_chain(vector_store)

    answer = chain.invoke(query)
    docs = _get_documents(vector_store, query)
    sources = [
        doc.metadata.get("source", "unknown")
        for doc in docs
        if hasattr(doc, "metadata") and isinstance(doc.metadata, dict)
    ]

    result = {
        "query": query,
        "answer": answer,
        "sources": sources,
    }

    print("Answer:", answer)
    print("Sources:", sources)

    return result


def test_run_perfect_rag_returns_expected_shape():
    query = "How does the Gulf fishing community perceive red snapper season limits and quotas?"
    result = run_perfect_rag(query)

    assert result["query"] == query
    assert isinstance(result["answer"], str)
    assert result["answer"].strip()
    assert isinstance(result["sources"], list)


if __name__ == "__main__":
    SAMPLE_QUERY = "Explain NOAA's ecosystem-based fisheries management framework in the Gulf of Mexico"
    run_perfect_rag(SAMPLE_QUERY)
