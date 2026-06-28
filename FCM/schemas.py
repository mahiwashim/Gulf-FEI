from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class DocumentIn(BaseModel):
    doc_id: str
    text: str
    metadata: Dict[str, str] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    documents: List[DocumentIn]


class QueryRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=50)
    scenario_activation: Dict[str, float] = Field(default_factory=dict)
    run_simulation: bool = True
    max_relations: int = Field(default=10, ge=1, le=100)
    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    min_abs_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    manual_relations: List['ManualRelationIn'] = Field(default_factory=list)


class RetrievedChunk(BaseModel):
    doc_id: str
    score: float
    text: str
    metadata: Dict[str, str] = Field(default_factory=dict)


class Edge(BaseModel):
    source: str
    target: str
    weight: float
    polarity: str
    confidence: float = 0.7
    evidence: str
    evidence_doc_id: Optional[str] = None
    edge_type: str = 'pattern'   # 'pattern' | 'cooccurrence' | 'transitive'
    hops: int = 1                 # 1 = direct, 2+ = transitive chain length


class ManualRelationIn(BaseModel):
    source: str
    target: str
    weight: float = Field(ge=-1.0, le=1.0)
    confidence: float = Field(default=0.95, ge=0.0, le=1.0)
    evidence: str = 'manual relation'


class FCMMap(BaseModel):
    concepts: List[str]
    edges: List[Edge]
    adjacency_matrix: List[List[float]]
    query: str


class AskResponse(BaseModel):
    query: str
    retrieved_chunks: List[RetrievedChunk]
    fcm_map: FCMMap
    simulation: Dict[str, float] = Field(default_factory=dict)
    answer: str
    artifacts: Dict[str, str] = Field(default_factory=dict)
