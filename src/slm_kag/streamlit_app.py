from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import asdict, replace
from hashlib import sha256
from pathlib import Path
from typing import Any

import streamlit as st

from slm_kag.bootstrap import Runtime, build_runtime
from slm_kag.config import Settings

SUPPORTED_EXTENSIONS = (
    "pdf",
    "docx",
    "pptx",
    "xlsx",
    "html",
    "htm",
    "md",
    "txt",
    "csv",
    "png",
    "jpg",
    "jpeg",
    "tif",
    "tiff",
    "bmp",
)


@st.cache_resource(scope="session", show_spinner=False)
def _runtime(settings: Settings) -> Runtime:
    return build_runtime(settings)


def _select_index(options: tuple[str, ...], value: str) -> int:
    try:
        return options.index(value)
    except ValueError:
        return 0


def _configuration() -> Settings:
    if "active_settings" not in st.session_state:
        st.session_state.active_settings = Settings.from_env()
    active: Settings = st.session_state.active_settings
    loader_options = ("docling", "text")
    extractor_options = ("rules", "huggingface", "modelscope", "ollama")
    generator_options = ("deterministic", "huggingface", "modelscope", "ollama")

    with st.sidebar:
        st.header("Configuração")
        with st.form("runtime_configuration"):
            db_path = st.text_input("Banco SQLite", value=active.db_path)
            loader = st.selectbox(
                "Leitura de documentos",
                loader_options,
                index=_select_index(loader_options, active.document_loader),
            )
            extractor = st.selectbox(
                "Extração de fatos",
                extractor_options,
                index=_select_index(extractor_options, active.extractor),
            )
            generator = st.selectbox(
                "Geração de respostas",
                generator_options,
                index=_select_index(generator_options, active.generator),
            )
            with st.expander("Modelos locais"):
                hf_model = st.text_input("Hugging Face", value=active.hf_model)
                modelscope_model = st.text_input("ModelScope", value=active.modelscope_model)
                ollama_model = st.text_input("Ollama", value=active.ollama_model)
                ollama_url = st.text_input("URL Ollama", value=active.ollama_url)
                model_device = st.selectbox(
                    "Dispositivo",
                    ("auto", "cpu", "cuda", "mps"),
                    index=_select_index(
                        ("auto", "cpu", "cuda", "mps"), active.model_device
                    ),
                )
                model_cache = st.text_input(
                    "Cache dos modelos", value=active.model_cache or ""
                )
                max_new_tokens = st.number_input(
                    "Tokens de saída",
                    min_value=64,
                    max_value=4096,
                    value=active.model_max_new_tokens,
                )
            with st.expander("Limites e recuperação"):
                max_file_mb = st.number_input(
                    "Máximo por arquivo (MB)",
                    min_value=1,
                    max_value=2048,
                    value=max(1, active.max_file_size // (1024 * 1024)),
                )
                max_pages = st.number_input(
                    "Máximo de páginas",
                    min_value=1,
                    max_value=10000,
                    value=active.max_pages,
                )
                top_k = st.number_input(
                    "Evidências por resposta",
                    min_value=1,
                    max_value=50,
                    value=active.top_k,
                )
                max_hops = st.number_input(
                    "Saltos no grafo", min_value=1, max_value=10, value=active.max_hops
                )
            apply_settings = st.form_submit_button(
                "Aplicar configuração", type="primary", width="stretch"
            )

        if apply_settings:
            updated = replace(
                active,
                db_path=db_path.strip(),
                document_loader=loader,
                extractor=extractor,
                generator=generator,
                hf_model=hf_model.strip(),
                modelscope_model=modelscope_model.strip(),
                ollama_model=ollama_model.strip(),
                ollama_url=ollama_url.strip(),
                model_device=model_device,
                model_cache=model_cache.strip() or None,
                model_max_new_tokens=int(max_new_tokens),
                max_file_size=int(max_file_mb) * 1024 * 1024,
                max_pages=int(max_pages),
                top_k=int(top_k),
                max_hops=int(max_hops),
            )
            if updated != active:
                st.session_state.active_settings = updated
                st.session_state.chat_history = []
                st.rerun()

        st.caption(
            "A configuração é aplicada por sessão. Alterar o banco ou modelo "
            "inicia outro runtime."
        )
    return active


def _parse_metadata(raw: str) -> dict[str, object]:
    if not raw.strip():
        return {}
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("Os metadados adicionais devem formar um objeto JSON.")
    return parsed


def _result_row(
    name: str, *, report: Any | None = None, error: Exception | None = None
) -> dict[str, object]:
    if error is not None:
        return {
            "arquivo": name,
            "status": "erro",
            "chunks": 0,
            "fatos": 0,
            "entidades": 0,
            "páginas": 0,
            "detalhe": str(error),
        }
    metadata = report.metadata.get("docling", {})
    return {
        "arquivo": name,
        "status": metadata.get("conversion_status", "indexado"),
        "chunks": report.chunks,
        "fatos": report.facts,
        "entidades": report.entities,
        "páginas": metadata.get("page_count", 0),
        "detalhe": "",
    }


def _index_uploads(
    runtime: Runtime,
    uploaded_files: list[Any],
    metadata: dict[str, object],
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    progress = st.progress(0, text="Preparando documentos")
    with st.status("Indexando documentos", expanded=True) as status:
        with tempfile.TemporaryDirectory(prefix="slm-kag-streamlit-") as temp_dir:
            for index, uploaded in enumerate(uploaded_files, start=1):
                display_name = str(uploaded.name)
                safe_name = Path(display_name).name or f"documento-{index}"
                content = uploaded.getvalue()
                digest = sha256(content).hexdigest()
                file_dir = Path(temp_dir) / str(index)
                file_dir.mkdir()
                temporary_path = file_dir / safe_name
                temporary_path.write_bytes(content)
                status.write(f"Convertendo e indexando: `{display_name}`")
                try:
                    report = runtime.ingest_file.execute(
                        temporary_path,
                        source=f"streamlit://{digest}/{display_name}",
                        metadata={
                            **metadata,
                            "uploaded_via": "streamlit",
                            "original_filename": display_name,
                            "upload_mime_type": uploaded.type,
                        },
                    )
                    results.append(_result_row(display_name, report=report))
                except Exception as exc:  # cada arquivo falha de forma independente
                    results.append(_result_row(display_name, error=exc))
                progress.progress(
                    index / len(uploaded_files), text=f"{index}/{len(uploaded_files)}"
                )
        errors = sum(item["status"] == "erro" for item in results)
        status.update(
            label=f"Indexação concluída: {len(results) - errors} sucesso(s), {errors} erro(s)",
            state="error" if errors else "complete",
            expanded=bool(errors),
        )
    progress.empty()
    return results


def _index_local_files(
    runtime: Runtime,
    paths: list[Path],
    metadata: dict[str, object],
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    progress = st.progress(0, text="Preparando arquivos locais")
    with st.status("Indexando pasta data", expanded=True) as status:
        for index, path in enumerate(paths, start=1):
            status.write(f"Convertendo e indexando: `{path.name}`")
            try:
                report = runtime.ingest_file.execute(
                    path,
                    metadata={**metadata, "indexed_via": "streamlit-data-directory"},
                )
                results.append(_result_row(path.name, report=report))
            except Exception as exc:
                results.append(_result_row(path.name, error=exc))
            progress.progress(index / len(paths), text=f"{index}/{len(paths)}")
        errors = sum(item["status"] == "erro" for item in results)
        status.update(
            label=f"Indexação concluída: {len(results) - errors} sucesso(s), {errors} erro(s)",
            state="error" if errors else "complete",
            expanded=bool(errors),
        )
    progress.empty()
    return results


def _render_metrics(stats: dict[str, int]) -> None:
    labels = (
        ("Documentos", "documents"),
        ("Chunks", "chunks"),
        ("Entidades", "entities"),
        ("Relações", "relations"),
        ("Menções", "entity_mentions"),
    )
    columns = st.columns(len(labels))
    for column, (label, key) in zip(columns, labels, strict=True):
        column.metric(label, stats.get(key, 0))


def _render_answer(message: dict[str, Any]) -> None:
    st.markdown(message["answer"])
    grounded = message.get("grounded", False)
    confidence = float(message.get("confidence", 0))
    st.caption(
        f"{'✓ Fundamentada' if grounded else '⚠ Sem fundamentação suficiente'} · "
        f"confiança {confidence:.0%}"
    )
    evidence = message.get("evidence", [])
    if evidence:
        with st.expander(f"Evidências ({len(evidence)})"):
            for item in evidence:
                metadata = item.get("metadata", {})
                filename = metadata.get("file", {}).get("name") or item.get("document_title")
                st.markdown(
                    f"**[{item['id']}] {filename or 'Fonte'}** · "
                    f"{item['kind']} · score {float(item['score']):.4f}"
                )
                st.write(item["content"])
                with st.popover("Metadados"):
                    st.json(metadata)
                st.divider()
    with st.expander("Plano e rastreamento"):
        st.json(message.get("trace", {}))


def _indexing_panel(runtime: Runtime, settings: Settings) -> bool:
    st.subheader("Monte a base de conhecimento")
    st.write(
        "Selecione arquivos quantas vezes quiser. As perguntas usam todos os documentos já "
        "persistidos no banco configurado."
    )
    mode = st.radio("Origem", ("Upload de arquivos", "Pasta inteira"), horizontal=True)
    multiple: bool | str = True if mode == "Upload de arquivos" else "directory"
    generation = st.session_state.get("upload_generation", 0)
    uploaded_files = st.file_uploader(
        "Documentos",
        type=SUPPORTED_EXTENSIONS,
        accept_multiple_files=multiple,
        max_upload_size=max(1, settings.max_file_size // (1024 * 1024)),
        key=f"document_uploads_{generation}",
        help="É possível selecionar novos arquivos de forma cumulativa antes de indexar.",
    )
    metadata_raw = st.text_area(
        "Metadados adicionais do lote (JSON)",
        value="{}",
        height=90,
        placeholder='{"projeto":"KAG","classificação":"interno"}',
    )
    if uploaded_files:
        total_bytes = sum(len(item.getvalue()) for item in uploaded_files)
        st.caption(
            f"{len(uploaded_files)} arquivo(s) selecionado(s) · "
            f"{total_bytes / 1024:.1f} KB"
        )
    if st.button(
        f"Indexar {len(uploaded_files)} documento(s)",
        type="primary",
        disabled=not uploaded_files,
        width="stretch",
    ):
        try:
            metadata = _parse_metadata(metadata_raw)
        except (ValueError, json.JSONDecodeError) as exc:
            st.error(f"Metadados inválidos: {exc}")
        else:
            st.session_state.last_index_results = _index_uploads(
                runtime, list(uploaded_files), metadata
            )
            st.session_state.upload_generation = generation + 1
            st.rerun()

    local_paths = sorted(
        path
        for path in Path("data").glob("*")
        if path.is_file() and path.suffix.casefold().lstrip(".") in SUPPORTED_EXTENSIONS
    )
    with st.expander(f"Arquivos existentes em data/ ({len(local_paths)})"):
        for path in local_paths:
            st.write(f"• `{path.name}`")
        if st.button(
            "Indexar todos os arquivos de data/",
            disabled=not local_paths,
            width="stretch",
        ):
            try:
                metadata = _parse_metadata(metadata_raw)
            except (ValueError, json.JSONDecodeError) as exc:
                st.error(f"Metadados inválidos: {exc}")
            else:
                st.session_state.last_index_results = _index_local_files(
                    runtime, local_paths, metadata
                )
                st.rerun()

    results = st.session_state.get("last_index_results", [])
    if results:
        st.subheader("Último lote")
        st.dataframe(results, hide_index=True, width="stretch")
    return bool(uploaded_files)


def _question_panel(runtime: Runtime, stats: dict[str, int], indexing: bool) -> None:
    st.subheader("Converse com a base")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if stats.get("documents", 0) == 0:
        st.info("Indexe ao menos um documento antes de fazer perguntas.")
        return
    if indexing:
        st.info("Clique em **Indexar documentos** antes de perguntar sobre os novos arquivos.")

    if st.button("Limpar conversa", disabled=not st.session_state.chat_history):
        st.session_state.chat_history = []
        st.rerun()
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                _render_answer(message)
            else:
                st.markdown(message["content"])

    prompt = st.chat_input("Faça uma pergunta sobre os documentos")
    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Recuperando e verificando evidências..."):
                response = asdict(runtime.answer_question.execute(prompt))
            message = {"role": "assistant", **response}
            _render_answer(message)
        st.session_state.chat_history.append(message)


def run_app() -> None:
    st.set_page_config(
        page_title="SLM-KAG",
        page_icon="◈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("◈ SLM-KAG")
    st.caption("Conhecimento local, rastreável e apoiado por documentos")
    settings = _configuration()
    try:
        runtime = _runtime(settings)
        stats = runtime.store.stats()
    except Exception as exc:
        st.error(f"Não foi possível iniciar o sistema: {exc}")
        st.stop()

    _render_metrics(stats)
    index_tab, question_tab = st.tabs(("1 · Indexar documentos", "2 · Fazer perguntas"))
    with index_tab:
        pending_uploads = _indexing_panel(runtime, settings)
        stats = runtime.store.stats()
    with question_tab:
        _question_panel(runtime, stats, pending_uploads)


def main() -> None:
    """Inicia o servidor ao executar o entry point `slm-kag-gui`."""
    from streamlit.web import cli as streamlit_cli

    sys.argv = ["streamlit", "run", str(Path(__file__).resolve())]
    raise SystemExit(streamlit_cli.main())


if __name__ == "__main__":
    run_app()
