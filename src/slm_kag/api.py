import json
import tempfile
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, AsyncIterator

from slm_kag.bootstrap import Runtime, build_runtime
from slm_kag.config import Settings


def create_app(settings: Settings | None = None) -> Any:
    try:
        from fastapi import FastAPI, File, Form, HTTPException, UploadFile
        from pydantic import BaseModel, Field
        from starlette.concurrency import run_in_threadpool
    except ImportError as exc:  # pragma: no cover - depende do extra api
        raise RuntimeError("instale o extra da API: pip install -e '.[api]'") from exc

    effective_settings = settings or Settings.from_env()
    runtime: Runtime = build_runtime(effective_settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        runtime.close()

    app = FastAPI(
        title="SLM-KAG",
        version="0.1.0",
        description="API local-first de geração aumentada por grafo de conhecimento",
        lifespan=lifespan,
    )

    class DocumentInput(BaseModel):
        title: str = Field(min_length=1)
        content: str = Field(min_length=1)
        source: str = Field(min_length=1)
        metadata: dict[str, Any] = Field(default_factory=dict)

    class QuestionInput(BaseModel):
        question: str = Field(min_length=1)

    @app.get("/health")
    def health() -> dict[str, object]:
        return {"status": "ok", "store": runtime.store.stats()}

    @app.post("/v1/documents")
    def ingest(payload: DocumentInput) -> dict[str, object]:
        try:
            report = runtime.index_document.execute(**payload.model_dump())
            return asdict(report)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/v1/files")
    async def ingest_file(
        file: UploadFile = File(...),
        source: str | None = Form(default=None),
        metadata_json: str | None = Form(default=None),
    ) -> dict[str, object]:
        filename = Path(file.filename or "upload.bin").name
        user_metadata: dict[str, object] = {"upload_filename": filename}
        if metadata_json:
            try:
                parsed_metadata = json.loads(metadata_json)
            except json.JSONDecodeError as exc:
                raise HTTPException(status_code=422, detail="metadata_json inválido") from exc
            if not isinstance(parsed_metadata, dict):
                raise HTTPException(status_code=422, detail="metadata_json deve ser um objeto")
            user_metadata.update(parsed_metadata)

        try:
            with tempfile.TemporaryDirectory(prefix="slm-kag-upload-") as temp_dir:
                temporary_path = Path(temp_dir) / filename
                size = 0
                with temporary_path.open("wb") as target:
                    while block := await file.read(1024 * 1024):
                        size += len(block)
                        if size > effective_settings.max_file_size:
                            raise HTTPException(status_code=413, detail="arquivo excede o limite")
                        target.write(block)
                report = await run_in_threadpool(
                    runtime.ingest_file.execute,
                    temporary_path,
                    source=source or f"upload://{filename}",
                    metadata=user_metadata,
                )
                return asdict(report)
        except HTTPException:
            raise
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        finally:
            await file.close()

    @app.post("/v1/query")
    def query(payload: QuestionInput) -> dict[str, object]:
        try:
            return asdict(runtime.answer_question.execute(payload.question))
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app
