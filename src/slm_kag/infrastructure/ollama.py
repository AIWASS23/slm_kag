from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from slm_kag.domain.models import Evidence
from slm_kag.ports import ChatModel


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, base_url: str, model: str, timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def chat(
        self,
        messages: Sequence[dict[str, str]],
        *,
        format_schema: dict[str, Any] | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "stream": False,
            "options": {"temperature": 0},
        }
        if format_schema:
            payload["format"] = format_schema
        request = Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise OllamaError(f"falha ao consultar Ollama em {self.base_url}: {exc}") from exc
        try:
            return str(result["message"]["content"])
        except (KeyError, TypeError) as exc:
            raise OllamaError("resposta inesperada do Ollama") from exc


class GroundedChatAnswerGenerator:
    """Gerador fundamentado compartilhado por todos os backends de chat."""

    def __init__(self, client: ChatModel) -> None:
        self.client = client

    def generate(self, question: str, evidence: Sequence[Evidence]) -> str:
        if not evidence:
            return "Não encontrei evidências suficientes na base de conhecimento para responder."
        context = "\n".join(f"[{item.id}] {item.content}" for item in evidence)
        return self.client.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "Responda somente com fatos sustentados pelas evidências. "
                        "Cite toda afirmação factual com identificadores como [E1]. "
                        "Se faltar suporte, declare a limitação."
                    ),
                },
                {"role": "user", "content": f"Pergunta: {question}\n\nEvidências:\n{context}"},
            ]
        ).strip()


# Compatibilidade com a API pública da primeira versão.
OllamaAnswerGenerator = GroundedChatAnswerGenerator
