from __future__ import annotations

import os
from dataclasses import dataclass


def _positive_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} deve ser maior que zero")
    return value


def _boolean(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().casefold()
    if normalized in {"1", "true", "yes", "sim", "on"}:
        return True
    if normalized in {"0", "false", "no", "nao", "não", "off"}:
        return False
    raise ValueError(f"{name} deve ser true ou false")


@dataclass(frozen=True, slots=True)
class Settings:
    db_path: str = "data/slm_kag.db"
    document_loader: str = "docling"
    max_file_size: int = 50 * 1024 * 1024
    max_pages: int = 500
    extractor: str = "rules"
    generator: str = "deterministic"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:3b"
    hf_model: str = "Qwen/Qwen2.5-3B-Instruct"
    modelscope_model: str = "Qwen/Qwen2.5-3B-Instruct"
    model_device: str = "auto"
    model_cache: str | None = "data/models"
    model_max_new_tokens: int = 512
    trust_remote_code: bool = False
    top_k: int = 8
    max_hops: int = 2
    evidence_budget: int = 6000

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            db_path=os.getenv("SLM_KAG_DB_PATH", "data/slm_kag.db"),
            document_loader=os.getenv("SLM_KAG_DOCUMENT_LOADER", "docling").casefold(),
            max_file_size=_positive_int("SLM_KAG_MAX_FILE_SIZE", 50 * 1024 * 1024),
            max_pages=_positive_int("SLM_KAG_MAX_PAGES", 500),
            extractor=os.getenv("SLM_KAG_EXTRACTOR", "rules").casefold(),
            generator=os.getenv("SLM_KAG_GENERATOR", "deterministic").casefold(),
            ollama_url=os.getenv("SLM_KAG_OLLAMA_URL", "http://localhost:11434"),
            ollama_model=os.getenv("SLM_KAG_OLLAMA_MODEL", "qwen2.5:3b"),
            hf_model=os.getenv("SLM_KAG_HF_MODEL", "Qwen/Qwen2.5-3B-Instruct"),
            modelscope_model=os.getenv(
                "SLM_KAG_MODELSCOPE_MODEL", "Qwen/Qwen2.5-3B-Instruct"
            ),
            model_device=os.getenv("SLM_KAG_MODEL_DEVICE", "auto"),
            model_cache=os.getenv("SLM_KAG_MODEL_CACHE", "data/models") or None,
            model_max_new_tokens=_positive_int("SLM_KAG_MODEL_MAX_NEW_TOKENS", 512),
            trust_remote_code=_boolean("SLM_KAG_TRUST_REMOTE_CODE", False),
            top_k=_positive_int("SLM_KAG_TOP_K", 8),
            max_hops=_positive_int("SLM_KAG_MAX_HOPS", 2),
            evidence_budget=_positive_int("SLM_KAG_EVIDENCE_BUDGET", 6000),
        )
