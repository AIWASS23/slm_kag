from __future__ import annotations

import json
import re
import unicodedata

from slm_kag.domain.models import Chunk, ExtractedFact, normalize_name
from slm_kag.ports import ChatModel


def _clean(value: str) -> str:
    return value.strip(" \t\n\r.,;:!?()[]{}\"'")


def _parse_json_object(content: str) -> dict[str, object]:
    """Aceita JSON puro ou cercado por Markdown, sem avaliar texto/código arbitrário."""
    candidate = content.strip()
    if candidate.startswith("```"):
        first_newline = candidate.find("\n")
        if first_newline >= 0:
            candidate = candidate[first_newline + 1 :]
        if candidate.endswith("```"):
            candidate = candidate[:-3].rstrip()
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(candidate[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("a extração estruturada deve retornar um objeto JSON")
    return payload


def _grounded_verbatim(label: str, content: str) -> bool:
    """Compara texto ignorando acentos, Markdown e pontuação, mas não aceita paráfrases."""

    def canonical(value: str) -> str:
        normalized = unicodedata.normalize("NFKD", value.casefold())
        without_accents = "".join(
            char for char in normalized if not unicodedata.combining(char)
        )
        return " ".join(re.findall(r"[a-z0-9]+", without_accents))

    entity = canonical(label)
    return bool(entity) and entity in canonical(content)


class RuleBasedFactExtractor:
    """Extrator previsível para bootstrap, testes e documentos sem modelo local."""

    _patterns = (
        (re.compile(r"^(?P<s>.+?)\s+trabalha\s+em\s+(?P<o>.+)$", re.I), "trabalha_em"),
        (re.compile(r"^(?P<s>.+?)\s+works\s+at\s+(?P<o>.+)$", re.I), "works_at"),
        (re.compile(r"^(?P<s>.+?)\s+nasceu\s+em\s+(?P<o>.+)$", re.I), "nasceu_em"),
        (re.compile(r"^(?P<s>.+?)\s+pertence\s+(?:a|ao|à)\s+(?P<o>.+)$", re.I), "pertence_a"),
        (
            re.compile(r"^(?P<s>.+?)\s+está\s+localizad[ao]\s+em\s+(?P<o>.+)$", re.I),
            "localizado_em",
        ),
        (re.compile(r"^(?P<s>.+?)\s+is\s+(?P<o>.+)$", re.I), "is"),
        (re.compile(r"^(?P<s>.+?)\s+é\s+(?P<o>.+)$", re.I), "e"),
    )

    def extract(self, chunk: Chunk) -> list[ExtractedFact]:
        facts: list[ExtractedFact] = []
        seen: set[tuple[str, str, str]] = set()
        lines = [line.strip() for line in chunk.content.splitlines() if line.strip()]
        for line in lines:
            explicit = [part.strip() for part in line.strip("() ").split("|")]
            if len(explicit) == 3 and all(explicit):
                self._append(facts, seen, chunk.id, *explicit, confidence=1.0)

        sentences = re.split(r"(?<=[.!?])\s+|\n+", chunk.content)
        for sentence in sentences:
            candidate = _clean(sentence)
            if not candidate or "|" in candidate:
                continue
            for pattern, predicate in self._patterns:
                match = pattern.match(candidate)
                if not match:
                    continue
                subject, obj = _clean(match.group("s")), _clean(match.group("o"))
                if subject and obj and len(subject) <= 160 and len(obj) <= 160:
                    self._append(
                        facts,
                        seen,
                        chunk.id,
                        subject,
                        predicate,
                        obj,
                        confidence=0.82,
                    )
                break
        return facts

    @staticmethod
    def _append(
        facts: list[ExtractedFact],
        seen: set[tuple[str, str, str]],
        chunk_id: str,
        subject: str,
        predicate: str,
        obj: str,
        *,
        confidence: float,
    ) -> None:
        key = (subject.casefold(), predicate.casefold(), obj.casefold())
        if key in seen:
            return
        seen.add(key)
        facts.append(ExtractedFact(subject, predicate, obj, chunk_id, confidence))


class StructuredFactExtractor:
    """Extração JSON comum a qualquer backend de chat local."""

    def __init__(self, client: ChatModel) -> None:
        self.client = client

    def extract(self, chunk: Chunk) -> list[ExtractedFact]:
        schema = {
            "type": "object",
            "properties": {
                "facts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "subject": {"type": "string"},
                            "predicate": {"type": "string"},
                            "object": {"type": "string"},
                            "subject_type": {"type": "string"},
                            "object_type": {"type": "string"},
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        },
                        "required": ["subject", "predicate", "object", "confidence"],
                    },
                }
            },
            "required": ["facts"],
        }
        content = self.client.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "Extraia somente fatos explicitamente afirmados. Use entidades curtas, "
                        "predicados canônicos em snake_case e não complete informação ausente."
                    ),
                },
                {"role": "user", "content": chunk.content},
            ],
            format_schema=schema,
        )
        payload = _parse_json_object(content)
        facts: list[ExtractedFact] = []
        seen: set[tuple[str, str, str]] = set()
        for item in payload.get("facts", []):
            subject = _clean(str(item.get("subject", "")))
            predicate = _clean(str(item.get("predicate", "")))
            obj = _clean(str(item.get("object", "")))
            if not subject or not predicate or not obj:
                continue
            if not _grounded_verbatim(subject, chunk.content) or not _grounded_verbatim(
                obj, chunk.content
            ):
                continue
            key = (normalize_name(subject), normalize_name(predicate), normalize_name(obj))
            if key in seen:
                continue
            seen.add(key)
            facts.append(
                ExtractedFact(
                    subject=subject,
                    predicate=predicate,
                    object=obj,
                    chunk_id=chunk.id,
                    confidence=max(0.0, min(1.0, float(item.get("confidence", 0.5)))),
                    subject_type=str(item.get("subject_type", "Entity")),
                    object_type=str(item.get("object_type", "Entity")),
                )
            )
        return facts


# Compatibilidade com a API pública da primeira versão.
OllamaFactExtractor = StructuredFactExtractor
