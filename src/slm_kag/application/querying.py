from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import replace
from time import perf_counter

from slm_kag.domain.models import (
    EntityHit,
    Evidence,
    EvidenceKind,
    GraphSnapshot,
    LogicalOperation,
    QueryPlan,
    QueryResponse,
    QueryTrace,
    VerificationResult,
)
from slm_kag.ports import AnswerGenerator, KnowledgeStore


class LogicalPlanner:
    """Compila a pergunta para uma forma lógica pequena e auditável."""

    _count_markers = ("quantos", "quantas", "how many", "número de", "numero de")
    _compare_markers = ("compare", "comparar", "diferença", "diferenca", "versus", " vs ")

    def plan(self, question: str, *, max_hops: int, top_k: int) -> QueryPlan:
        normalized = question.casefold()
        operations = [LogicalOperation.LOOKUP, LogicalOperation.GRAPH_EXPAND]
        if any(marker in normalized for marker in self._count_markers):
            operations.append(LogicalOperation.COUNT)
        if any(marker in normalized for marker in self._compare_markers):
            operations.append(LogicalOperation.COMPARE)
        return QueryPlan(question, tuple(operations), max_hops=max_hops, top_k=top_k)


class GroundingVerifier:
    """Impede que uma resposta seja marcada como fundamentada sem citações válidas."""

    _citation = re.compile(r"\[(E\d+)\]")

    def verify(self, answer: str, evidence: Sequence[Evidence]) -> VerificationResult:
        available = {item.id for item in evidence}
        citations = tuple(dict.fromkeys(self._citation.findall(answer)))
        issues: list[str] = []
        if evidence and not citations:
            issues.append("resposta sem citação")
        invalid = [citation for citation in citations if citation not in available]
        if invalid:
            issues.append(f"citações inexistentes: {', '.join(invalid)}")
        if not evidence:
            issues.append("nenhuma evidência recuperada")
        return VerificationResult(not issues, citations, tuple(issues))


class DeterministicAnswerGenerator:
    """Fallback seguro: sumariza somente evidências recuperadas, sem inferir fatos novos."""

    def generate(self, question: str, evidence: Sequence[Evidence]) -> str:
        del question
        if not evidence:
            return "Não encontrei evidências suficientes na base de conhecimento para responder."
        statements = [f"{item.content.strip()} [{item.id}]" for item in evidence[:4]]
        return "Com base na base de conhecimento: " + " ".join(statements)


class HybridRetriever:
    """Funde busca textual, indexação mútua e Personalized PageRank local."""

    def __init__(self, store: KnowledgeStore, evidence_budget: int = 6_000) -> None:
        self.store = store
        self.evidence_budget = evidence_budget

    @staticmethod
    def _page_rank(
        snapshot: GraphSnapshot,
        seeds: Sequence[EntityHit],
        damping: float = 0.85,
        iterations: int = 20,
    ) -> dict[str, float]:
        node_ids = {entity.id for entity in snapshot.entities}
        if not node_ids:
            return {}
        adjacency: dict[str, set[str]] = defaultdict(set)
        for relation in snapshot.relations:
            adjacency[relation.source.id].add(relation.target.id)
            adjacency[relation.target.id].add(relation.source.id)

        seed_scores = {hit.entity.id: hit.score for hit in seeds if hit.entity.id in node_ids}
        total_seed = sum(seed_scores.values())
        if total_seed <= 0:
            personalization = {node_id: 1.0 / len(node_ids) for node_id in node_ids}
        else:
            personalization = {
                node_id: seed_scores.get(node_id, 0.0) / total_seed for node_id in node_ids
            }
        rank = dict(personalization)
        for _ in range(iterations):
            updated = {node_id: (1.0 - damping) * personalization[node_id] for node_id in node_ids}
            dangling = sum(rank[node_id] for node_id in node_ids if not adjacency[node_id])
            for node_id in node_ids:
                updated[node_id] += damping * dangling * personalization[node_id]
            for source, neighbors in adjacency.items():
                if not neighbors:
                    continue
                contribution = damping * rank[source] / len(neighbors)
                for target in neighbors:
                    updated[target] += contribution
            rank = updated
        return rank

    def retrieve(
        self, plan: QueryPlan
    ) -> tuple[list[Evidence], list[EntityHit], int, int]:
        lexical = self.store.search_chunks(plan.question, max(plan.top_k * 2, 10))
        seeds = self.store.match_entities(plan.question, max(plan.top_k, 8))
        snapshot = self.store.neighborhood(
            [hit.entity.id for hit in seeds],
            max_hops=plan.max_hops,
        )
        ranks = self._page_rank(snapshot, seeds)

        candidates: dict[tuple[str, str], Evidence] = {}
        for position, hit in enumerate(lexical):
            score = 1.0 / (60.0 + position + 1)
            item = hit.item
            key = (EvidenceKind.TEXT.value, item.chunk.id)
            candidates[key] = Evidence(
                id="",
                kind=EvidenceKind.TEXT,
                content=item.chunk.content,
                score=score,
                chunk_id=item.chunk.id,
                document_title=item.document_title,
                document_source=item.document_source,
                metadata=item.document_metadata,
            )

        ranked_entities = sorted(ranks, key=ranks.get, reverse=True)
        graph_chunks = self.store.chunks_for_entities(
            ranked_entities[: plan.top_k * 2], plan.top_k * 2
        )
        for position, item in enumerate(graph_chunks):
            key = (EvidenceKind.TEXT.value, item.chunk.id)
            graph_score = 1.0 / (60.0 + position + 1)
            previous = candidates.get(key)
            if previous:
                candidates[key] = replace(previous, score=previous.score + graph_score)
            else:
                candidates[key] = Evidence(
                    id="",
                    kind=EvidenceKind.TEXT,
                    content=item.chunk.content,
                    score=graph_score,
                    chunk_id=item.chunk.id,
                    document_title=item.document_title,
                    document_source=item.document_source,
                    metadata=item.document_metadata,
                )

        chunk_lookup = self.store.get_chunks(
            [relation.chunk_id for relation in snapshot.relations if relation.chunk_id]
        )
        ranked_relations = sorted(
            snapshot.relations,
            key=lambda relation: max(
                ranks.get(relation.source.id, 0), ranks.get(relation.target.id, 0)
            )
            * relation.confidence,
            reverse=True,
        )
        for position, relation in enumerate(ranked_relations):
            graph_score = relation.confidence / (60.0 + position + 1)
            source_chunk = chunk_lookup.get(relation.chunk_id)
            key = (EvidenceKind.RELATION.value, relation.id)
            candidates[key] = Evidence(
                id="",
                kind=EvidenceKind.RELATION,
                content=f"{relation.source.name} --{relation.predicate}--> {relation.target.name}",
                score=graph_score,
                chunk_id=relation.chunk_id,
                document_title=source_chunk.document_title if source_chunk else None,
                document_source=source_chunk.document_source if source_chunk else None,
                relation_id=relation.id,
                metadata=source_chunk.document_metadata if source_chunk else {},
            )

        ordered = sorted(candidates.values(), key=lambda item: item.score, reverse=True)
        selected: list[Evidence] = []
        consumed = 0
        for item in ordered:
            if len(selected) >= plan.top_k:
                break
            size = len(item.content)
            if selected and consumed + size > self.evidence_budget:
                continue
            selected.append(replace(item, id=f"E{len(selected) + 1}"))
            consumed += size
        return selected, seeds, len(lexical), len(snapshot.relations)


class AnswerQuestion:
    def __init__(
        self,
        store: KnowledgeStore,
        generator: AnswerGenerator,
        *,
        max_hops: int = 2,
        top_k: int = 8,
        evidence_budget: int = 6_000,
    ) -> None:
        self.planner = LogicalPlanner()
        self.retriever = HybridRetriever(store, evidence_budget=evidence_budget)
        self.generator = generator
        self.fallback = DeterministicAnswerGenerator()
        self.verifier = GroundingVerifier()
        self.max_hops = max_hops
        self.top_k = top_k

    def execute(self, question: str) -> QueryResponse:
        if not question.strip():
            raise ValueError("question não pode ser vazia")
        started = perf_counter()
        plan = self.planner.plan(question.strip(), max_hops=self.max_hops, top_k=self.top_k)
        evidence, entities, lexical_count, graph_count = self.retriever.retrieve(plan)
        answer = self.generator.generate(plan.question, evidence)
        verification = self.verifier.verify(answer, evidence)
        if evidence and not verification.grounded:
            answer = self.fallback.generate(plan.question, evidence)
            verification = self.verifier.verify(answer, evidence)

        mean_score = sum(item.score for item in evidence) / len(evidence) if evidence else 0.0
        confidence = min(1.0, mean_score * 12.0) if verification.grounded else 0.0
        trace = QueryTrace(
            plan=plan,
            matched_entities=tuple(hit.entity.name for hit in entities),
            lexical_candidates=lexical_count,
            graph_relations=graph_count,
            elapsed_ms=round((perf_counter() - started) * 1_000, 3),
            verifier_issues=verification.issues,
        )
        return QueryResponse(
            answer=answer,
            evidence=tuple(evidence),
            grounded=verification.grounded,
            confidence=round(confidence, 4),
            trace=trace,
        )
