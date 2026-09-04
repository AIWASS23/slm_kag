from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from slm_kag.api import create_app
from slm_kag.bootstrap import build_runtime
from slm_kag.config import Settings


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def _settings(args: argparse.Namespace) -> Settings:
    settings = Settings.from_env()
    if getattr(args, "db", None):
        settings = replace(settings, db_path=args.db)
    if getattr(args, "loader", None):
        settings = replace(settings, document_loader=args.loader)
    return settings


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="slm-kag", description="KAG local-first para SLMs")
    parser.add_argument("--db", help="caminho do banco SQLite")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("init", help="inicializa o banco")

    ingest = commands.add_parser("ingest", help="indexa um arquivo de texto ou Markdown")
    ingest.add_argument("path", type=Path)
    ingest.add_argument("--title", help="título do documento")
    ingest.add_argument("--source", help="identificador estável da origem")
    ingest.add_argument(
        "--loader",
        choices=("docling", "text"),
        help="conversor documental (padrão: SLM_KAG_DOCUMENT_LOADER)",
    )

    query = commands.add_parser("query", help="consulta a base")
    query.add_argument("question")

    serve = commands.add_parser("serve", help="inicia a API HTTP")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", default=8000, type=int)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    settings = _settings(args)

    if args.command == "serve":
        try:
            import uvicorn
        except ImportError as exc:
            parser.error("o comando serve requer: pip install -e '.[api]'")
            raise AssertionError from exc
        uvicorn.run(create_app(settings), host=args.host, port=args.port)
        return

    runtime = build_runtime(settings)
    try:
        if args.command == "init":
            _print_json({"database": settings.db_path, "stats": runtime.store.stats()})
        elif args.command == "ingest":
            path: Path = args.path
            if not path.is_file():
                parser.error(f"arquivo não encontrado: {path}")
            report = runtime.ingest_file.execute(
                path,
                title=args.title,
                source=args.source,
            )
            _print_json(asdict(report))
        elif args.command == "query":
            _print_json(asdict(runtime.answer_question.execute(args.question)))
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
