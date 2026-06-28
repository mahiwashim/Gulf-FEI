from __future__ import annotations

from typing import Dict, List

from .schemas import Edge


def build_answer(query: str, edges: List[Edge], simulation: Dict[str, float]) -> str:
    lines: List[str] = [f'Query: {query}']
    if edges:
        lines.append('Main causal relations found from retrieved evidence:')
        for edge in sorted(edges, key=lambda e: abs(e.weight), reverse=True)[:8]:
            direction = 'increases' if edge.weight >= 0 else 'decreases'
            lines.append(
                f'- {edge.source} {direction} {edge.target} (weight={edge.weight}, confidence={edge.confidence})'
            )
    else:
        lines.append('No explicit causal edges were extracted. The retrieved evidence was still indexed and converted into a concept list for manual review.')

    if simulation:
        lines.append('Scenario propagation result:')
        for concept, value in sorted(simulation.items(), key=lambda kv: abs(kv[1]), reverse=True)[:8]:
            lines.append(f'- {concept}: {value}')
    return '\n'.join(lines)
