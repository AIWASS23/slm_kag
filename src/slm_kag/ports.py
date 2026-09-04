from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, Sequence

from slm_kag.domain.models import (
    Chunk,
    ChunkHit,
    Document,
    EntityHit,
    Evidence,
    ExtractedFact,
    GraphSnapshot,
    IndexReport,
    LoadedDocument,
    StoredChunk,
)


class KnowledgeStore(Protocol):
    def initialize(self) -> None: ...

    def replace_document(
        self,
        document: Document,
        chunks: Sequence[Chunk],
        facts: Sequence[ExtractedFact],
    ) -> IndexReport: ...

    def search_chunks(self, query: str, limit: int) -> list[ChunkHit]: ...

    def match_entities(self, query: str, limit: int) -> list[EntityHit]: ...

    def neighborhood(self, entity_ids: Sequence[str], max_hops: int) -> GraphSnapshot: ...

    def chunks_for_entities(self, entity_ids: Sequence[str], limit: int) -> list[StoredChunk]: ...

    def get_chunks(self, chunk_ids: Sequence[str]) -> dict[str, StoredChunk]: ...

    def stats(self) -> dict[str, int]: ...


class FactExtractor(Protocol):
    def extract(self, chunk: Chunk) -> list[ExtractedFact]: ...


class DocumentLoader(Protocol):
    def load(self, path: Path, *, source: str | None = None) -> LoadedDocument: ...


class AnswerGenerator(Protocol):
    def generate(self, question: str, evidence: Sequence[Evidence]) -> str: ...


class ChatModel(Protocol):
    """Contrato mínimo compartilhado pelos runtimes Ollama, HF e ModelScope."""

    def chat(
        self,
        messages: Sequence[dict[str, str]],
        *,
        format_schema: dict[str, Any] | None = None,
    ) -> str: ...
