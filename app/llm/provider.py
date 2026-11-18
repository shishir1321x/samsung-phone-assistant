"""LLM backends behind one tiny interface.

Open-source models are the default:

* ``huggingface``   - Hugging Face Inference Providers (OpenAI-compatible router),
                      e.g. Qwen/Qwen3-235B-A22B-Instruct-2507, Qwen/Qwen2.5-72B-Instruct.
                      Needs Inference Provider credits on the Hugging Face account.
* ``ollama``        - any model pulled into a local Ollama server (llama3.1, qwen2.5, ...).
* ``transformers``  - run a Hugging Face model in-process (CPU is fine for 0.5B-1.5B models).
* ``openai`` / ``openai_compatible`` - OpenAI, vLLM, LM Studio, Groq, ...

``auto`` picks the first one that is configured; with nothing configured the
system still works and answers with grounded, template-based text.
"""

from __future__ import annotations

import logging
import re
import threading
from functools import lru_cache
from typing import Protocol

import requests

from app.config import Settings, get_settings

log = logging.getLogger(__name__)

HF_ROUTER_URL = "https://router.huggingface.co/v1"
# Served by several mainstream providers (deepinfra, scaleway); ungated, Apache-2.0.
DEFAULT_HF_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"
DEFAULT_LOCAL_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_OLLAMA_MODEL = "llama3.1:8b"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


class LLMClient(Protocol):
    name: str

    def generate(self, system: str, prompt: str, max_tokens: int | None = None) -> str: ...


_THINK = re.compile(r"<think>.*?</think>\s*", re.S)
_ODD_CHARS = str.maketrans({" ": " ", " ": " ", " ": " ", "‑": "-", "‐": "-"})


def clean_output(text: str) -> str:
    """Drop reasoning blocks some open models emit and normalise unusual spaces/hyphens."""
    return _THINK.sub("", text).translate(_ODD_CHARS).strip()


class ChatModelClient:
    """Adapter for any LangChain chat model."""

    def __init__(self, model, name: str):
        self.model = model
        self.name = name

    def generate(self, system: str, prompt: str, max_tokens: int | None = None) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage

        model = self.model.bind(max_tokens=max_tokens) if max_tokens else self.model
        result = model.invoke([SystemMessage(content=system), HumanMessage(content=prompt)])
        content = result.content
        if isinstance(content, list):  # some providers return content parts
            content = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
        return clean_output(str(content))


class TransformersClient:
    """Runs a Hugging Face chat model locally with ``transformers`` (greedy decoding)."""

    def __init__(self, model_id: str, default_max_tokens: int = 700):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = f"transformers:{model_id}"
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
        self.model.eval()
        self.default_max_tokens = default_max_tokens
        self._lock = threading.Lock()

    def generate(self, system: str, prompt: str, max_tokens: int | None = None) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
        with self._lock, self._torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens or self.default_max_tokens,
                do_sample=False,
                repetition_penalty=1.05,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        return clean_output(self.tokenizer.decode(new_tokens, skip_special_tokens=True))


def _ollama_models(base_url: str) -> list[str] | None:
    try:
        response = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=0.8)
        response.raise_for_status()
        return [m["name"] for m in response.json().get("models", [])]
    except Exception:
        return None


def _openai_client(model: str, base_url: str | None, api_key: str, settings: Settings, label: str) -> ChatModelClient:
    from langchain_openai import ChatOpenAI

    chat = ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout=settings.llm_timeout,
        max_retries=1,
    )
    return ChatModelClient(chat, f"{label}:{model}")


def build_llm(settings: Settings) -> LLMClient | None:
    provider = settings.llm_provider
    if provider == "none":
        return None

    if provider == "auto":
        if settings.hf_token:
            provider = "huggingface"
        elif _ollama_models(settings.ollama_base_url):
            provider = "ollama"
        elif settings.openai_api_key:
            provider = "openai"
        elif settings.llm_base_url and settings.llm_model:
            provider = "openai_compatible"
        else:
            log.info("No LLM configured; answers will use grounded templates.")
            return None

    if provider == "huggingface":
        if not settings.hf_token:
            raise ValueError("LLM_PROVIDER=huggingface needs HF_TOKEN")
        return _openai_client(settings.llm_model or DEFAULT_HF_MODEL, settings.llm_base_url or HF_ROUTER_URL,
                              settings.hf_token, settings, "huggingface")
    if provider == "ollama":
        available = _ollama_models(settings.ollama_base_url) or []
        model = settings.llm_model or (available[0] if available else DEFAULT_OLLAMA_MODEL)
        return _openai_client(model, f"{settings.ollama_base_url.rstrip('/')}/v1", "ollama", settings, "ollama")
    if provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("LLM_PROVIDER=openai needs OPENAI_API_KEY")
        return _openai_client(settings.llm_model or DEFAULT_OPENAI_MODEL, settings.llm_base_url,
                              settings.openai_api_key, settings, "openai")
    if provider == "openai_compatible":
        if not (settings.llm_base_url and settings.llm_model):
            raise ValueError("LLM_PROVIDER=openai_compatible needs LLM_BASE_URL and LLM_MODEL")
        return _openai_client(settings.llm_model, settings.llm_base_url, settings.llm_api_key or "not-needed",
                              settings, "openai-compatible")
    if provider == "transformers":
        return TransformersClient(settings.llm_model or DEFAULT_LOCAL_MODEL, settings.llm_max_tokens)
    raise ValueError(f"Unknown LLM_PROVIDER {provider!r}")


@lru_cache(maxsize=1)
def get_llm() -> LLMClient | None:
    try:
        return build_llm(get_settings())
    except Exception as exc:
        log.error("Could not initialise the LLM (%s); falling back to grounded templates.", exc)
        return None


def reset_llm() -> None:
    get_llm.cache_clear()
