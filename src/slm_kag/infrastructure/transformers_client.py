from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from threading import RLock
from typing import Any


class LocalModelError(RuntimeError):
    pass


class HuggingFaceChatClient:
    """Inferência local de modelos de chat compatíveis com Hugging Face Transformers.

    Modelo e tokenizer são carregados somente na primeira chamada. Isso permite iniciar a API,
    executar health checks e usar o pipeline determinístico sem alocar CPU/GPU desnecessariamente.
    """

    provider = "huggingface"

    def __init__(
        self,
        model_id: str,
        *,
        device: str = "auto",
        cache_dir: str | None = None,
        max_new_tokens: int = 512,
        trust_remote_code: bool = False,
    ) -> None:
        if not model_id.strip():
            raise ValueError("model_id não pode ser vazio")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens deve ser maior que zero")
        self.model_id = model_id
        self.device = device
        self.cache_dir = cache_dir
        self.max_new_tokens = max_new_tokens
        self.trust_remote_code = trust_remote_code
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._lock = RLock()

    def _resolve_model_source(self) -> str:
        return self.model_id

    def _load(self) -> tuple[Any, Any]:
        if self._tokenizer is not None and self._model is not None:
            return self._tokenizer, self._model
        with self._lock:
            if self._tokenizer is not None and self._model is not None:
                return self._tokenizer, self._model
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as exc:
                raise LocalModelError(
                    f"backend {self.provider} requer: pip install -e '.[{self.provider}]'"
                ) from exc

            source = self._resolve_model_source()
            common: dict[str, Any] = {
                "trust_remote_code": self.trust_remote_code,
            }
            if self.cache_dir:
                Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
                common["cache_dir"] = self.cache_dir
            try:
                tokenizer = AutoTokenizer.from_pretrained(source, **common)
                model_options = {**common, "torch_dtype": "auto"}
                if self.device == "auto":
                    model_options["device_map"] = "auto"
                model = AutoModelForCausalLM.from_pretrained(source, **model_options)
                if self.device != "auto":
                    model = model.to(self.device)
                model.eval()
            except Exception as exc:
                raise LocalModelError(
                    f"não foi possível carregar {self.provider}:{self.model_id}: {exc}"
                ) from exc
            self._tokenizer, self._model = tokenizer, model
            return tokenizer, model

    @staticmethod
    def _with_schema_instruction(
        messages: Sequence[dict[str, str]], format_schema: dict[str, Any] | None
    ) -> list[dict[str, str]]:
        prepared = [dict(message) for message in messages]
        if not format_schema:
            return prepared
        instruction = (
            "\nRetorne somente JSON válido, sem Markdown, obedecendo exatamente "
            "a este JSON Schema: "
            + json.dumps(format_schema, ensure_ascii=False, separators=(",", ":"))
        )
        for message in prepared:
            if message.get("role") == "system":
                message["content"] = message.get("content", "") + instruction
                break
        else:
            prepared.insert(0, {"role": "system", "content": instruction.strip()})
        return prepared

    @staticmethod
    def _manual_prompt(messages: Sequence[dict[str, str]]) -> str:
        body = "\n".join(
            f"{message.get('role', 'user').upper()}: {message.get('content', '')}"
            for message in messages
        )
        return f"{body}\nASSISTANT:"

    def chat(
        self,
        messages: Sequence[dict[str, str]],
        *,
        format_schema: dict[str, Any] | None = None,
    ) -> str:
        tokenizer, model = self._load()
        prepared = self._with_schema_instruction(messages, format_schema)
        try:
            if getattr(tokenizer, "chat_template", None):
                try:
                    encoded = tokenizer.apply_chat_template(
                        prepared,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_tensors="pt",
                        return_dict=True,
                    )
                except TypeError:
                    encoded = tokenizer.apply_chat_template(
                        prepared,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_tensors="pt",
                    )
            else:
                encoded = tokenizer(
                    self._manual_prompt(prepared), return_tensors="pt", add_special_tokens=True
                )
            if hasattr(encoded, "to"):
                encoded = encoded.to(model.device)
            if isinstance(encoded, Mapping):
                generation_inputs = dict(encoded)
                input_ids = generation_inputs["input_ids"]
            else:
                input_ids = encoded.to(model.device)
                generation_inputs = {"input_ids": input_ids}
            with self._lock:
                output = model.generate(
                    **generation_inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=(
                        tokenizer.pad_token_id
                        if tokenizer.pad_token_id is not None
                        else tokenizer.eos_token_id
                    ),
                )
            generated = output[0, input_ids.shape[-1] :]
            return str(tokenizer.decode(generated, skip_special_tokens=True)).strip()
        except Exception as exc:
            raise LocalModelError(
                f"falha de inferência em {self.provider}:{self.model_id}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc


class ModelScopeChatClient(HuggingFaceChatClient):
    """Baixa o snapshot do ModelScope e executa-o localmente via Transformers."""

    provider = "modelscope"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._resolved_source: str | None = None

    def _resolve_model_source(self) -> str:
        if self._resolved_source:
            return self._resolved_source
        try:
            from modelscope import snapshot_download
        except ImportError as exc:
            raise LocalModelError(
                "backend modelscope requer: pip install -e '.[modelscope]'"
            ) from exc
        options: dict[str, Any] = {}
        if self.cache_dir:
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
            options["cache_dir"] = self.cache_dir
        try:
            self._resolved_source = str(snapshot_download(self.model_id, **options))
        except Exception as exc:
            raise LocalModelError(
                f"não foi possível baixar modelscope:{self.model_id}: {exc}"
            ) from exc
        return self._resolved_source


# Nome genérico mantido para consumidores que tratam o runtime pelo motor subjacente.
TransformersChatClient = HuggingFaceChatClient
