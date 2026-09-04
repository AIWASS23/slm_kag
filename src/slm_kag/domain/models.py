from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from enum import StrEnum
from hashlib import sha256
from typing import Any


def normalize_name(value: str) -> str:
    """Forma canônica usada para resolução determinística de entidades."""
    normalized = unicodedata.normalize("NFKD", value.strip().casefold())
    ascii_value = "".join(char for char in normalized if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", ascii_value)


def stable_id(namespace: str, *parts: str) -> str:
    payload = "\x1f".join(normalize_name(part) for part in parts)
    return f"{namespace}_{sha256(payload.encode('utf-8')).hexdigest()[:20]}"


@dataclass(frozen=True, slots=True)
class Document:
    id: str
    title: str
    content: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Chunk:
    id: str
    document_id: str
    position: int
    content: str


@dataclass(frozen=True, slots=True)
class ExtractedFact:
    subject: str
    predicate: str
    object: str
    chunk_id: str
    confidence: float = 1.0
    subject_type: str = "Entity"
    object_type: str = "Entity"


@dataclass(frozen=True, slots=True)
class Entity:
    id: str
    name: str
    normalized_name: str
    kind: str = "Entity"


@dataclass(frozen=True, slots=True)
class Relation:
    id: str
    source: Entity
    predicate: str
    target: Entity
    confidence: float
    chunk_id: str


@dataclass(frozen=True, slots=True)
class StoredChunk:
    chunk: Chunk
    document_title: str
    document_source: str
    document_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LoadedDocument:
    title: str
    content: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ChunkHit:
    item: StoredChunk
    score: float


@dataclass(frozen=True, slots=True)
class EntityHit:
    entity: Entity
    score: float


@dataclass(frozen=True, slots=True)
class GraphSnapshot:
    entities: tuple[Entity, ...]
    relations: tuple[Relation, ...]


class EvidenceKind(StrEnum):
    TEXT = "text"
    RELATION = "relation"


@dataclass(frozen=True, slots=True)
class Evidence:
    id: str
    kind: EvidenceKind
    content: str
    score: float
    chunk_id: str | None = None
    document_title: str | None = None
    document_source: str | None = None
    relation_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class LogicalOperation(StrEnum):
    LOOKUP = "lookup"
    GRAPH_EXPAND = "graph_expand"
    COUNT = "count"
    COMPARE = "compare"


@dataclass(frozen=True, slots=True)
class QueryPlan:
    question: str
    operations: tuple[LogicalOperation, ...]
    max_hops: int
    top_k: int


@dataclass(frozen=True, slots=True)
class VerificationResult:
    grounded: bool
    cited_evidence: tuple[str, ...]
    issues: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class QueryTrace:
    plan: QueryPlan
    matched_entities: tuple[str, ...]
    lexical_candidates: int
    graph_relations: int
    elapsed_ms: float
    verifier_issues: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class QueryResponse:
    answer: str
    evidence: tuple[Evidence, ...]
    grounded: bool
    confidence: float
    trace: QueryTrace


@dataclass(frozen=True, slots=True)
class IndexReport:
    document_id: str
    chunks: int
    facts: int
    entities: int
    metadata: dict[str, Any] = field(default_factory=dict)
