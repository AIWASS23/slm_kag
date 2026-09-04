from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from threading import RLock

from slm_kag.domain.models import (
    Chunk,
    ChunkHit,
    Document,
    Entity,
    EntityHit,
    ExtractedFact,
    GraphSnapshot,
    IndexReport,
    Relation,
    StoredChunk,
    normalize_name,
    stable_id,
)

_STOP_WORDS = {
    "a",
    "as",
    "de",
    "do",
    "da",
    "dos",
    "das",
    "e",
    "em",
    "o",
    "os",
    "para",
    "por",
    "que",
    "qual",
    "quais",
    "quem",
    "como",
    "onde",
    "the",
    "what",
    "who",
    "where",
    "is",
    "are",
}


class SQLiteKnowledgeStore:
    """Property graph leve e índice textual persistidos em um único arquivo."""

    def __init__(self, path: str) -> None:
        self.path = path
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._lock = RLock()
        self._fts_enabled = False

    def initialize(self) -> None:
        with self._lock:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source TEXT NOT NULL UNIQUE,
                    metadata_json TEXT NOT NULL,
                    indexed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    position INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    UNIQUE(document_id, position)
                );

                CREATE TABLE IF NOT EXISTS entities (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    normalized_name TEXT NOT NULL UNIQUE,
                    kind TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS entity_mentions (
                    entity_id TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
                    chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
                    role TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    PRIMARY KEY(entity_id, chunk_id, role)
                );

                CREATE TABLE IF NOT EXISTS relations (
                    id TEXT PRIMARY KEY,
                    source_entity_id TEXT NOT NULL REFERENCES entities(id),
                    predicate TEXT NOT NULL,
                    target_entity_id TEXT NOT NULL REFERENCES entities(id),
                    confidence REAL NOT NULL CHECK(confidence >= 0 AND confidence <= 1),
                    chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
                CREATE INDEX IF NOT EXISTS idx_mentions_chunk ON entity_mentions(chunk_id);
                CREATE INDEX IF NOT EXISTS idx_relations_source ON relations(source_entity_id);
                CREATE INDEX IF NOT EXISTS idx_relations_target ON relations(target_entity_id);
                CREATE INDEX IF NOT EXISTS idx_relations_chunk ON relations(chunk_id);
                """
            )
            try:
                self._connection.execute(
                    """CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts
                    USING fts5(
                        content, title, chunk_id UNINDEXED,
                        tokenize='unicode61 remove_diacritics 2'
                    )"""
                )
                self._fts_enabled = True
            except sqlite3.OperationalError:
                self._fts_enabled = False
            self._connection.commit()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    @staticmethod
    def _entity_id(name: str) -> str:
        return stable_id("ent", normalize_name(name))

    @staticmethod
    def _predicate(value: str) -> str:
        return re.sub(r"[^a-z0-9_]+", "_", normalize_name(value)).strip("_") or "related_to"

    def _upsert_entity(self, name: str, kind: str) -> str:
        normalized = normalize_name(name)
        entity_id = self._entity_id(name)
        self._connection.execute(
            """INSERT INTO entities(id, name, normalized_name, kind) VALUES (?, ?, ?, ?)
            ON CONFLICT(normalized_name) DO UPDATE SET
                name = CASE WHEN length(excluded.name) > length(entities.name)
                            THEN excluded.name ELSE entities.name END,
                kind = CASE WHEN entities.kind = 'Entity'
                            THEN excluded.kind ELSE entities.kind END""",
            (entity_id, name.strip(), normalized, kind),
        )
        row = self._connection.execute(
            "SELECT id FROM entities WHERE normalized_name = ?", (normalized,)
        ).fetchone()
        return str(row["id"])

    def replace_document(
        self,
        document: Document,
        chunks: Sequence[Chunk],
        facts: Sequence[ExtractedFact],
    ) -> IndexReport:
        with self._lock, self._connection:
            old_chunk_ids = [
                str(row["id"])
                for row in self._connection.execute(
                    "SELECT id FROM chunks WHERE document_id = ?", (document.id,)
                )
            ]
            if self._fts_enabled and old_chunk_ids:
                placeholders = ",".join("?" for _ in old_chunk_ids)
                self._connection.execute(
                    f"DELETE FROM chunk_fts WHERE chunk_id IN ({placeholders})", old_chunk_ids
                )
            self._connection.execute("DELETE FROM documents WHERE id = ?", (document.id,))
            self._connection.execute(
                """INSERT INTO documents(id, title, content, source, metadata_json)
                VALUES (?, ?, ?, ?, ?)""",
                (
                    document.id,
                    document.title,
                    document.content,
                    document.source,
                    json.dumps(document.metadata, ensure_ascii=False, sort_keys=True),
                ),
            )
            self._connection.executemany(
                "INSERT INTO chunks(id, document_id, position, content) VALUES (?, ?, ?, ?)",
                [(chunk.id, chunk.document_id, chunk.position, chunk.content) for chunk in chunks],
            )
            if self._fts_enabled:
                self._connection.executemany(
                    "INSERT INTO chunk_fts(content, title, chunk_id) VALUES (?, ?, ?)",
                    [(chunk.content, document.title, chunk.id) for chunk in chunks],
                )

            entity_ids: set[str] = set()
            valid_chunk_ids = {chunk.id for chunk in chunks}
            for fact in facts:
                if fact.chunk_id not in valid_chunk_ids:
                    raise ValueError(f"fato referencia chunk desconhecido: {fact.chunk_id}")
                source_id = self._upsert_entity(fact.subject, fact.subject_type)
                target_id = self._upsert_entity(fact.object, fact.object_type)
                entity_ids.update((source_id, target_id))
                confidence = max(0.0, min(1.0, fact.confidence))
                self._connection.executemany(
                    """INSERT OR REPLACE INTO entity_mentions(entity_id, chunk_id, role, confidence)
                    VALUES (?, ?, ?, ?)""",
                    [
                        (source_id, fact.chunk_id, "subject", confidence),
                        (target_id, fact.chunk_id, "object", confidence),
                    ],
                )
                predicate = self._predicate(fact.predicate)
                relation_id = stable_id(
                    "rel", source_id, predicate, target_id, fact.chunk_id
                )
                self._connection.execute(
                    """INSERT OR REPLACE INTO relations(
                        id, source_entity_id, predicate, target_entity_id, confidence, chunk_id
                    ) VALUES (?, ?, ?, ?, ?, ?)""",
                    (relation_id, source_id, predicate, target_id, confidence, fact.chunk_id),
                )

            self._connection.execute(
                """DELETE FROM entities
                WHERE NOT EXISTS (
                    SELECT 1 FROM entity_mentions m WHERE m.entity_id = entities.id
                ) AND NOT EXISTS (
                    SELECT 1 FROM relations r
                    WHERE r.source_entity_id = entities.id OR r.target_entity_id = entities.id
                )"""
            )
        return IndexReport(
            document.id,
            len(chunks),
            len(facts),
            len(entity_ids),
            metadata=dict(document.metadata),
        )

    @staticmethod
    def _terms(query: str) -> list[str]:
        terms = re.findall(r"[\wÀ-ÿ-]+", normalize_name(query))
        filtered = [term for term in terms if len(term) > 1 and term not in _STOP_WORDS]
        return filtered or [term for term in terms if len(term) > 1]

    @staticmethod
    def _stored_chunk(row: sqlite3.Row) -> StoredChunk:
        raw_metadata = row["document_metadata_json"]
        try:
            document_metadata = json.loads(str(raw_metadata)) if raw_metadata else {}
        except (TypeError, json.JSONDecodeError):
            document_metadata = {}
        return StoredChunk(
            chunk=Chunk(
                id=str(row["id"]),
                document_id=str(row["document_id"]),
                position=int(row["position"]),
                content=str(row["content"]),
            ),
            document_title=str(row["document_title"]),
            document_source=str(row["document_source"]),
            document_metadata=document_metadata,
        )

    def search_chunks(self, query: str, limit: int) -> list[ChunkHit]:
        terms = self._terms(query)
        if not terms:
            return []
        with self._lock:
            if self._fts_enabled:
                expression = " OR ".join(f'"{term.replace(chr(34), "")}"' for term in terms)
                rows = self._connection.execute(
                    """SELECT c.id, c.document_id, c.position, c.content,
                              d.title AS document_title, d.source AS document_source,
                              d.metadata_json AS document_metadata_json,
                              bm25(chunk_fts) AS raw_score
                       FROM chunk_fts
                       JOIN chunks c ON c.id = chunk_fts.chunk_id
                       JOIN documents d ON d.id = c.document_id
                       WHERE chunk_fts MATCH ?
                       ORDER BY raw_score ASC
                       LIMIT ?""",
                    (expression, limit),
                ).fetchall()
            else:
                clauses = " OR ".join("lower(c.content) LIKE ?" for _ in terms)
                rows = self._connection.execute(
                    f"""SELECT c.id, c.document_id, c.position, c.content,
                               d.title AS document_title, d.source AS document_source,
                               d.metadata_json AS document_metadata_json,
                               0.0 AS raw_score
                        FROM chunks c JOIN documents d ON d.id = c.document_id
                        WHERE {clauses} LIMIT ?""",
                    [*(f"%{term}%" for term in terms), limit],
                ).fetchall()
        return [
            ChunkHit(self._stored_chunk(row), 1.0 / (index + 1))
            for index, row in enumerate(rows)
        ]

    def match_entities(self, query: str, limit: int) -> list[EntityHit]:
        normalized_query = normalize_name(query)
        query_tokens = set(self._terms(query))
        with self._lock:
            rows = self._connection.execute(
                "SELECT id, name, normalized_name, kind FROM entities"
            ).fetchall()
        hits: list[EntityHit] = []
        for row in rows:
            normalized = str(row["normalized_name"])
            entity_tokens = set(normalized.split())
            overlap = len(query_tokens & entity_tokens)
            if normalized not in normalized_query and not overlap:
                continue
            exact_bonus = 1.0 if normalized in normalized_query else 0.0
            score = exact_bonus + overlap / max(1, len(entity_tokens))
            hits.append(
                EntityHit(
                    Entity(str(row["id"]), str(row["name"]), normalized, str(row["kind"])),
                    score,
                )
            )
        return sorted(hits, key=lambda hit: hit.score, reverse=True)[:limit]

    def neighborhood(self, entity_ids: Sequence[str], max_hops: int) -> GraphSnapshot:
        discovered = set(entity_ids)
        frontier = set(entity_ids)
        relation_rows: dict[str, sqlite3.Row] = {}
        with self._lock:
            for _ in range(max_hops):
                if not frontier:
                    break
                placeholders = ",".join("?" for _ in frontier)
                parameters = [*frontier, *frontier]
                rows = self._connection.execute(
                    f"""SELECT id, source_entity_id, predicate, target_entity_id,
                               confidence, chunk_id
                        FROM relations
                        WHERE source_entity_id IN ({placeholders})
                           OR target_entity_id IN ({placeholders})""",
                    parameters,
                ).fetchall()
                next_frontier: set[str] = set()
                for row in rows:
                    relation_rows[str(row["id"])] = row
                    next_frontier.update(
                        (str(row["source_entity_id"]), str(row["target_entity_id"]))
                    )
                next_frontier -= discovered
                discovered.update(next_frontier)
                frontier = next_frontier

            if not discovered:
                return GraphSnapshot((), ())
            placeholders = ",".join("?" for _ in discovered)
            entity_rows = self._connection.execute(
                f"""SELECT id, name, normalized_name, kind FROM entities
                WHERE id IN ({placeholders})""",
                list(discovered),
            ).fetchall()

        entities = {
            str(row["id"]): Entity(
                str(row["id"]),
                str(row["name"]),
                str(row["normalized_name"]),
                str(row["kind"]),
            )
            for row in entity_rows
        }
        relations = tuple(
            Relation(
                id=str(row["id"]),
                source=entities[str(row["source_entity_id"])],
                predicate=str(row["predicate"]),
                target=entities[str(row["target_entity_id"])],
                confidence=float(row["confidence"]),
                chunk_id=str(row["chunk_id"]),
            )
            for row in relation_rows.values()
            if str(row["source_entity_id"]) in entities and str(row["target_entity_id"]) in entities
        )
        return GraphSnapshot(tuple(entities.values()), relations)

    def chunks_for_entities(self, entity_ids: Sequence[str], limit: int) -> list[StoredChunk]:
        if not entity_ids:
            return []
        placeholders = ",".join("?" for _ in entity_ids)
        with self._lock:
            rows = self._connection.execute(
                f"""SELECT c.id, c.document_id, c.position, c.content,
                           d.title AS document_title, d.source AS document_source,
                           d.metadata_json AS document_metadata_json,
                           MAX(m.confidence) AS mention_score
                    FROM entity_mentions m
                    JOIN chunks c ON c.id = m.chunk_id
                    JOIN documents d ON d.id = c.document_id
                    WHERE m.entity_id IN ({placeholders})
                    GROUP BY c.id
                    ORDER BY mention_score DESC, c.position ASC
                    LIMIT ?""",
                [*entity_ids, limit],
            ).fetchall()
        return [self._stored_chunk(row) for row in rows]

    def get_chunks(self, chunk_ids: Sequence[str]) -> dict[str, StoredChunk]:
        if not chunk_ids:
            return {}
        unique_ids = list(dict.fromkeys(chunk_ids))
        placeholders = ",".join("?" for _ in unique_ids)
        with self._lock:
            rows = self._connection.execute(
                f"""SELECT c.id, c.document_id, c.position, c.content,
                           d.title AS document_title, d.source AS document_source,
                           d.metadata_json AS document_metadata_json
                    FROM chunks c JOIN documents d ON d.id = c.document_id
                    WHERE c.id IN ({placeholders})""",
                unique_ids,
            ).fetchall()
        items = [self._stored_chunk(row) for row in rows]
        return {item.chunk.id: item for item in items}

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                table: int(self._connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                for table in ("documents", "chunks", "entities", "relations", "entity_mentions")
            }
