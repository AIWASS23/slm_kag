from __future__ import annotations

import mimetypes
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from threading import RLock
from typing import Any

from slm_kag.domain.models import LoadedDocument


class DocumentLoadError(RuntimeError):
    pass


def _timestamp(value: float) -> str:
    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()


def _file_metadata(path: Path) -> dict[str, Any]:
    stat = path.stat()
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    mime_type, encoding = mimetypes.guess_type(path.name)
    return {
        "name": path.name,
        "stem": path.stem,
        "extension": path.suffix.casefold(),
        "mime_type": mime_type or "application/octet-stream",
        "content_encoding": encoding,
        "size_bytes": stat.st_size,
        "sha256": digest.hexdigest(),
        "modified_at": _timestamp(stat.st_mtime),
        "changed_at": _timestamp(stat.st_ctime),
    }


def _jsonable(value: Any) -> Any:
    if hasattr(value, "value"):
        return _jsonable(value.value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return str(value)


class DoclingDocumentLoader:
    """Normaliza arquivos com Docling e preserva metadados de origem e conversão."""

    def __init__(
        self,
        *,
        max_file_size: int = 50 * 1024 * 1024,
        max_pages: int = 500,
        converter: Any | None = None,
    ) -> None:
        self.max_file_size = max_file_size
        self.max_pages = max_pages
        self._converter = converter
        self._lock = RLock()

    def _get_converter(self) -> Any:
        if self._converter is not None:
            return self._converter
        with self._lock:
            if self._converter is not None:
                return self._converter
            try:
                from docling.document_converter import DocumentConverter
            except ImportError as exc:
                raise DocumentLoadError(
                    "ingestão de arquivos requer Docling: pip install -e '.[documents]'"
                ) from exc
            self._converter = DocumentConverter()
            return self._converter

    def load(self, path: Path, *, source: str | None = None) -> LoadedDocument:
        resolved = path.expanduser().resolve()
        if not resolved.is_file():
            raise DocumentLoadError(f"arquivo não encontrado: {path}")
        file_metadata = _file_metadata(resolved)
        if file_metadata["size_bytes"] > self.max_file_size:
            raise DocumentLoadError(
                f"arquivo excede o limite de {self.max_file_size} bytes: {resolved.name}"
            )

        converter = self._get_converter()
        try:
            with self._lock:
                result = converter.convert(
                    resolved,
                    raises_on_error=True,
                    max_num_pages=self.max_pages,
                    max_file_size=self.max_file_size,
                )
            document = result.document
            content = document.export_to_markdown(
                page_break_placeholder="\n\n<!-- page-break -->\n\n"
            ).strip()
        except Exception as exc:
            raise DocumentLoadError(
                f"Docling não conseguiu converter {resolved.name}: {exc}"
            ) from exc
        if not content:
            raise DocumentLoadError(f"Docling produziu conteúdo vazio para {resolved.name}")

        origin = _jsonable(getattr(document, "origin", None))
        pages = getattr(document, "pages", {}) or {}
        page_numbers = sorted(int(number) for number in pages)
        errors = [_jsonable(error) for error in (getattr(result, "errors", None) or [])]
        status = _jsonable(getattr(result, "status", None))
        docling_metadata = {
            "schema_name": getattr(document, "schema_name", None),
            "schema_version": getattr(document, "version", None),
            "document_name": getattr(document, "name", None),
            "origin": origin,
            "page_count": len(page_numbers),
            "page_numbers": page_numbers,
            "conversion_status": status,
            "errors": errors,
        }
        return LoadedDocument(
            title=str(getattr(document, "name", None) or resolved.stem),
            content=content,
            source=source or resolved.as_uri(),
            metadata={"file": file_metadata, "docling": docling_metadata},
        )


class PlainTextDocumentLoader:
    """Fallback explícito para ambientes mínimos; não substitui o Docling silenciosamente."""

    def __init__(self, *, max_file_size: int = 50 * 1024 * 1024) -> None:
        self.max_file_size = max_file_size

    def load(self, path: Path, *, source: str | None = None) -> LoadedDocument:
        resolved = path.expanduser().resolve()
        if not resolved.is_file():
            raise DocumentLoadError(f"arquivo não encontrado: {path}")
        file_metadata = _file_metadata(resolved)
        if file_metadata["size_bytes"] > self.max_file_size:
            raise DocumentLoadError(
                f"arquivo excede o limite de {self.max_file_size} bytes: {resolved.name}"
            )
        try:
            content = resolved.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise DocumentLoadError(
                "o loader text aceita somente UTF-8; use o loader docling para este arquivo"
            ) from exc
        return LoadedDocument(
            title=resolved.stem,
            content=content,
            source=source or resolved.as_uri(),
            metadata={"file": file_metadata, "loader": {"name": "text"}},
        )
