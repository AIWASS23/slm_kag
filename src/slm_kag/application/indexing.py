from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from slm_kag.domain.models import Chunk, Document, IndexReport, stable_id
from slm_kag.ports import DocumentLoader, FactExtractor, KnowledgeStore


class TextChunker:
    """Chunker por parágrafos com janela deslizante e sobreposição controlada."""

    def __init__(self, max_chars: int = 1_200, overlap_chars: int = 160) -> None:
        if max_chars < 200:
            raise ValueError("max_chars deve ser >= 200")
        if overlap_chars < 0 or overlap_chars >= max_chars:
            raise ValueError("overlap_chars deve estar entre 0 e max_chars")
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars

    def split(self, document: Document) -> list[Chunk]:
        text = document.content.strip()
        if not text:
            return []

        chunks: list[Chunk] = []
        start = 0
        position = 0
        while start < len(text):
            end = min(start + self.max_chars, len(text))
            if end < len(text):
                paragraph_break = text.rfind("\n\n", start + self.max_chars // 2, end)
                sentence_break = max(text.rfind(". ", start, end), text.rfind("\n", start, end))
                boundary = max(paragraph_break, sentence_break)
                if boundary > start:
                    end = boundary + (2 if text[boundary : boundary + 2] == ". " else 1)

            content = text[start:end].strip()
            if content:
                chunks.append(
                    Chunk(
                        id=stable_id("chk", document.id, str(position), content),
                        document_id=document.id,
                        position=position,
                        content=content,
                    )
                )
                position += 1
            if end >= len(text):
                break
            next_start = max(start + 1, end - self.overlap_chars)
            whitespace = text.find(" ", next_start, min(end + 1, len(text)))
            start = whitespace + 1 if whitespace >= 0 else next_start
        return chunks


class IndexDocument:
    def __init__(
        self,
        store: KnowledgeStore,
        extractor: FactExtractor,
        chunker: TextChunker | None = None,
    ) -> None:
        self.store = store
        self.extractor = extractor
        self.chunker = chunker or TextChunker()

    def execute(
        self,
        *,
        title: str,
        content: str,
        source: str,
        metadata: dict[str, object] | None = None,
    ) -> IndexReport:
        if not title.strip():
            raise ValueError("title não pode ser vazio")
        if not content.strip():
            raise ValueError("content não pode ser vazio")
        if not source.strip():
            raise ValueError("source não pode ser vazio")

        document = Document(
            id=stable_id("doc", source),
            title=title.strip(),
            content=content,
            source=source.strip(),
            metadata=dict(metadata or {}),
        )
        chunks = self.chunker.split(document)
        facts = [fact for chunk in chunks for fact in self.extractor.extract(chunk)]
        return self.store.replace_document(document, chunks, facts)


class IngestFile:
    """Converte um arquivo e encaminha conteúdo + metadados ao pipeline KAG."""

    def __init__(self, loader: DocumentLoader, index_document: IndexDocument) -> None:
        self.loader = loader
        self.index_document = index_document

    def execute(
        self,
        path: Path,
        *,
        source: str | None = None,
        title: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> IndexReport:
        loaded = self.loader.load(path, source=source)
        merged_metadata = dict(loaded.metadata)
        if metadata:
            merged_metadata["user"] = dict(metadata)
        return self.index_document.execute(
            title=title or loaded.title,
            content=loaded.content,
            source=loaded.source,
            metadata=merged_metadata,
        )


def index_many(indexer: IndexDocument, documents: Sequence[dict[str, object]]) -> list[IndexReport]:
    return [
        indexer.execute(
            title=str(item["title"]),
            content=str(item["content"]),
            source=str(item["source"]),
            metadata=dict(item.get("metadata", {})),
        )
        for item in documents
    ]
