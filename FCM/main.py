"""
FCM package — quick smoke-test / standalone demo.

Run from the project root:
    python -m FCM.main
"""
from __future__ import annotations

from .extractor import CausalExtractor
from .fcm_graph import FCMGraphBuilder
from .schemas import FCMMap


def demo() -> None:
    sample_text = (
        "Overfishing reduces fish biomass. "
        "Habitat loss threatens marine biodiversity. "
        "Climate change increases ocean temperature. "
        "Rising ocean temperature drives coral bleaching. "
        "Coral bleaching damages reef habitat."
    )

    extractor = CausalExtractor()
    chunks = [{"text": sample_text, "score": 1.0, "doc_id": "demo"}]
    concepts, edges = extractor.extract(chunks)

    builder = FCMGraphBuilder()
    fcm_map: FCMMap = builder.build("demo query", concepts, edges)

    print(f"Concepts ({len(fcm_map.concepts)}): {fcm_map.concepts}")
    print(f"Edges    ({len(fcm_map.edges)}):")
    for e in fcm_map.edges:
        sign = "+" if e.weight >= 0 else "-"
        print(f"  {e.source} --[{sign}{abs(e.weight):.2f}]--> {e.target}")


if __name__ == "__main__":
    demo()
