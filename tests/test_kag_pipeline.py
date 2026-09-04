from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from slm_kag.application.indexing import IndexDocument, IngestFile, TextChunker
from slm_kag.application.querying import AnswerQuestion, DeterministicAnswerGenerator
from slm_kag.bootstrap import build_runtime
from slm_kag.config import Settings
from slm_kag.domain.models import Document
from slm_kag.infrastructure.document_loaders import DoclingDocumentLoader
from slm_kag.infrastructure.extractors import RuleBasedFactExtractor, StructuredFactExtractor
from slm_kag.infrastructure.sqlite_store import SQLiteKnowledgeStore


class KAGPipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.store = SQLiteKnowledgeStore(str(Path(self.temporary.name) / "test.db"))
        self.store.initialize()
        self.indexer = IndexDocument(
            self.store,
            RuleBasedFactExtractor(),
            TextChunker(max_chars=400, overlap_chars=40),
        )
        self.answer = AnswerQuestion(
            self.store,
            DeterministicAnswerGenerator(),
            top_k=5,
            max_hops=2,
        )

    def tearDown(self) -> None:
        self.store.close()
        self.temporary.cleanup()

    def test_mutual_index_graph_retrieval_and_grounding(self) -> None:
        report = self.indexer.execute(
            title="Equipe",
            source="memory://team",
            content="Ana | trabalha_em | Acme\nAcme | localizada_em | Fortaleza",
        )

        response = self.answer.execute("Onde trabalha Ana?")

        self.assertEqual(report.facts, 2)
        self.assertEqual(report.entities, 3)
        self.assertTrue(response.grounded)
        self.assertIn("Ana --trabalha_em--> Acme", response.answer)
        self.assertIn("E1", response.answer)
        self.assertIn("Ana", response.trace.matched_entities)
        self.assertGreaterEqual(response.trace.graph_relations, 2)

    def test_reindex_same_source_removes_stale_graph(self) -> None:
        self.indexer.execute(
            title="Cadastro",
            source="memory://profile",
            content="Bia | trabalha_em | Antiga SA",
        )
        self.indexer.execute(
            title="Cadastro",
            source="memory://profile",
            content="Bia | trabalha_em | Nova SA",
        )

        stats = self.store.stats()
        response = self.answer.execute("Onde Bia trabalha?")

        self.assertEqual(stats["documents"], 1)
        self.assertEqual(stats["relations"], 1)
        self.assertIn("Nova SA", response.answer)
        self.assertNotIn("Antiga SA", response.answer)

    def test_unknown_question_is_not_claimed_as_grounded(self) -> None:
        response = self.answer.execute("Qual é a população de Marte?")

        self.assertFalse(response.grounded)
        self.assertEqual(response.evidence, ())
        self.assertIn("Não encontrei", response.answer)

    def test_structured_extractor_accepts_any_chat_backend(self) -> None:
        class FakeChatModel:
            def chat(self, messages, *, format_schema=None):  # type: ignore[no-untyped-def]
                self.messages = messages
                self.schema = format_schema
                return (
                    '```json\n{"facts":[{"subject":"Cris","predicate":"trabalha_em",'
                    '"object":"Gamma","confidence":0.91}]}\n```'
                )

        model = FakeChatModel()
        extractor = StructuredFactExtractor(model)
        chunks = TextChunker(max_chars=400, overlap_chars=40).split(
            Document(
                id="doc_test",
                title="Teste",
                content="Cris trabalha em Gamma.",
                source="memory://structured",
            )
        )

        facts = extractor.extract(chunks[0])

        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0].subject, "Cris")
        self.assertEqual(facts[0].object, "Gamma")
        self.assertIsNotNone(model.schema)

    def test_structured_extractor_rejects_unanchored_entities(self) -> None:
        class HallucinatingChatModel:
            def chat(self, messages, *, format_schema=None):  # type: ignore[no-untyped-def]
                return (
                    '{"facts":[{"subject":"Pessoa inexistente",'
                    '"predicate":"trabalha_em","object":"Empresa imaginária",'
                    '"confidence":0.99}]}'
                )

        chunk = TextChunker(max_chars=400, overlap_chars=40).split(
            Document(
                id="doc_grounding",
                title="Grounding",
                content="Cris trabalha em Gamma.",
                source="memory://grounding",
            )
        )[0]

        facts = StructuredFactExtractor(HallucinatingChatModel()).extract(chunk)

        self.assertEqual(facts, [])

    def test_huggingface_backend_is_lazy(self) -> None:
        runtime = build_runtime(
            Settings(
                db_path=str(Path(self.temporary.name) / "lazy.db"),
                generator="huggingface",
            )
        )
        try:
            response = runtime.answer_question.execute("Pergunta sem corpus")
            self.assertFalse(response.grounded)
        finally:
            runtime.close()

    def test_docling_content_and_file_metadata_reach_evidence(self) -> None:
        class FakeOrigin:
            def model_dump(self, **kwargs):  # type: ignore[no-untyped-def]
                return {"filename": "relatorio.pdf", "mimetype": "application/pdf"}

        class FakeDocument:
            name = "Relatório Docling"
            schema_name = "DoclingDocument"
            version = "1.0.0"
            origin = FakeOrigin()
            pages = {1: object(), 2: object()}

            def export_to_markdown(self, **kwargs):  # type: ignore[no-untyped-def]
                self.export_options = kwargs
                return "Lia | trabalha_em | Delta"

        class FakeResult:
            document = FakeDocument()
            status = "success"
            errors = []

        class FakeConverter:
            def convert(self, path, **kwargs):  # type: ignore[no-untyped-def]
                self.path = path
                self.options = kwargs
                return FakeResult()

        source_file = Path(self.temporary.name) / "relatorio.pdf"
        source_file.write_bytes(b"fake-pdf-content")
        converter = FakeConverter()
        loader = DoclingDocumentLoader(converter=converter)
        ingest = IngestFile(loader, self.indexer)

        report = ingest.execute(source_file)
        response = self.answer.execute("Onde Lia trabalha?")

        self.assertEqual(report.facts, 1)
        self.assertEqual(report.metadata["file"]["name"], "relatorio.pdf")
        self.assertEqual(converter.options["max_num_pages"], 500)
        self.assertEqual(response.evidence[0].metadata["file"]["name"], "relatorio.pdf")
        self.assertEqual(response.evidence[0].metadata["docling"]["page_count"], 2)
        self.assertEqual(len(response.evidence[0].metadata["file"]["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
