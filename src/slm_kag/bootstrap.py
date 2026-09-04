from __future__ import annotations

from dataclasses import dataclass

from slm_kag.application.indexing import IndexDocument, IngestFile
from slm_kag.application.querying import AnswerQuestion, DeterministicAnswerGenerator
from slm_kag.config import Settings
from slm_kag.infrastructure.document_loaders import (
    DoclingDocumentLoader,
    PlainTextDocumentLoader,
)
from slm_kag.infrastructure.extractors import RuleBasedFactExtractor, StructuredFactExtractor
from slm_kag.infrastructure.ollama import GroundedChatAnswerGenerator, OllamaClient
from slm_kag.infrastructure.sqlite_store import SQLiteKnowledgeStore
from slm_kag.infrastructure.transformers_client import (
    HuggingFaceChatClient,
    ModelScopeChatClient,
)
from slm_kag.ports import ChatModel


@dataclass(slots=True)
class Runtime:
    store: SQLiteKnowledgeStore
    index_document: IndexDocument
    ingest_file: IngestFile
    answer_question: AnswerQuestion

    def close(self) -> None:
        self.store.close()


def build_runtime(settings: Settings | None = None) -> Runtime:
    settings = settings or Settings.from_env()
    store = SQLiteKnowledgeStore(settings.db_path)
    store.initialize()

    if settings.document_loader == "docling":
        document_loader = DoclingDocumentLoader(
            max_file_size=settings.max_file_size,
            max_pages=settings.max_pages,
        )
    elif settings.document_loader == "text":
        document_loader = PlainTextDocumentLoader(max_file_size=settings.max_file_size)
    else:
        store.close()
        raise ValueError("SLM_KAG_DOCUMENT_LOADER deve ser 'docling' ou 'text'")

    clients: dict[str, ChatModel] = {}

    def chat_model(backend: str) -> ChatModel:
        if backend in clients:
            return clients[backend]
        common = {
            "device": settings.model_device,
            "cache_dir": settings.model_cache,
            "max_new_tokens": settings.model_max_new_tokens,
            "trust_remote_code": settings.trust_remote_code,
        }
        if backend == "ollama":
            client: ChatModel = OllamaClient(settings.ollama_url, settings.ollama_model)
        elif backend == "huggingface":
            client = HuggingFaceChatClient(settings.hf_model, **common)
        elif backend == "modelscope":
            client = ModelScopeChatClient(settings.modelscope_model, **common)
        else:
            raise ValueError(
                "backend de modelo deve ser 'ollama', 'huggingface' ou 'modelscope'"
            )
        clients[backend] = client
        return client

    if settings.extractor == "rules":
        extractor = RuleBasedFactExtractor()
    elif settings.extractor in {"ollama", "huggingface", "modelscope"}:
        extractor = StructuredFactExtractor(chat_model(settings.extractor))
    else:
        store.close()
        raise ValueError(
            "SLM_KAG_EXTRACTOR deve ser 'rules', 'ollama', 'huggingface' ou 'modelscope'"
        )

    if settings.generator == "deterministic":
        generator = DeterministicAnswerGenerator()
    elif settings.generator in {"ollama", "huggingface", "modelscope"}:
        generator = GroundedChatAnswerGenerator(chat_model(settings.generator))
    else:
        store.close()
        raise ValueError(
            "SLM_KAG_GENERATOR deve ser 'deterministic', 'ollama', 'huggingface' ou 'modelscope'"
        )

    index_document = IndexDocument(store, extractor)
    return Runtime(
        store=store,
        index_document=index_document,
        ingest_file=IngestFile(document_loader, index_document),
        answer_question=AnswerQuestion(
            store,
            generator,
            max_hops=settings.max_hops,
            top_k=settings.top_k,
            evidence_budget=settings.evidence_budget,
        ),
    )
