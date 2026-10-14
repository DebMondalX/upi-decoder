"""Model backends behind one interface.

    backend = make_backend("tinker:Qwen/Qwen3.5-4B")                       # base model on Tinker
    backend = make_backend("tinker:Qwen/Qwen3.5-4B@tinker://.../sampler_weights/final")  # fine-tuned
    backend = make_backend("ollama:upi-decoder")                           # local, after export
    backend = make_backend("openai:gpt-4.1-mini")                          # optional closed baseline
    results = asyncio.run(backend.generate_many(list_of_message_lists))

Each result has .text, .latency_s, .input_tokens, .output_tokens, .error
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass


@dataclass
class GenResult:
    text: str
    latency_s: float
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None


class Backend:
    name: str = "backend"
    concurrency: int = 16

    async def generate(self, messages: list[dict]) -> GenResult:  # pragma: no cover
        raise NotImplementedError

    async def generate_many(self, batch: list[list[dict]], progress: bool = True) -> list[GenResult]:
        sem = asyncio.Semaphore(self.concurrency)
        done = 0

        async def one(msgs):
            nonlocal done
            async with sem:
                try:
                    r = await self.generate(msgs)
                except Exception as e:  # keep going; failures count as invalid output
                    r = GenResult("", 0.0, error=f"{type(e).__name__}: {e}")
                done += 1
                if progress and (done % 50 == 0 or done == len(batch)):
                    print(f"  [{self.name}] {done}/{len(batch)}", flush=True)
                return r

        return await asyncio.gather(*(one(m) for m in batch))


# ---------------------------------------------------------------------------
# Tinker (base models and fine-tuned checkpoints)
# ---------------------------------------------------------------------------


def default_renderer(model_name: str) -> str:
    """Prefer the no-thinking renderer: we want fast, direct JSON."""
    from tinker_cookbook import model_info

    names = model_info.get_recommended_renderer_names(model_name)
    for n in names:
        if "disable_thinking" in n:
            return n
    return names[0]


class TinkerBackend(Backend):
    def __init__(
        self,
        base_model: str,
        model_path: str | None = None,
        renderer_name: str | None = None,
        max_tokens: int = 160,
        temperature: float = 0.0,
        concurrency: int = 32,
    ):
        self.base_model = base_model
        self.model_path = model_path
        self.renderer_name = renderer_name or default_renderer(base_model)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.concurrency = concurrency
        short = base_model.split("/")[-1]
        self.name = f"{short}+lora" if model_path else short
        self._client = None
        self._renderer = None
        self._lock = asyncio.Lock()

    async def _setup(self):
        async with self._lock:
            if self._client is not None:
                return
            import tinker
            from tinker_cookbook import renderers
            from tinker_cookbook.tokenizer_utils import get_tokenizer

            sc = tinker.ServiceClient()
            if self.model_path:
                self._client = await sc.create_sampling_client_async(model_path=self.model_path)
            else:
                self._client = await sc.create_sampling_client_async(base_model=self.base_model)
            tok = get_tokenizer(self.base_model)
            self._renderer = renderers.get_renderer(self.renderer_name, tok, model_name=self.base_model)

    async def generate(self, messages: list[dict]) -> GenResult:
        await self._setup()
        from tinker import types
        from tinker_cookbook.renderers import get_text_content

        prompt = self._renderer.build_generation_prompt(messages)
        params = types.SamplingParams(
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stop=self._renderer.get_stop_sequences(),
        )
        t0 = time.perf_counter()
        res = await self._client.sample_async(prompt=prompt, num_samples=1, sampling_params=params)
        dt = time.perf_counter() - t0
        toks = list(res.sequences[0].tokens)
        msg, _term = self._renderer.parse_response(toks)
        n_in = prompt.length if isinstance(prompt.length, int) else prompt.length()
        return GenResult(get_text_content(msg), dt, n_in, len(toks))


# ---------------------------------------------------------------------------
# Ollama (local inference after export) - native API so we can turn thinking off
# ---------------------------------------------------------------------------


class OllamaBackend(Backend):
    def __init__(self, model: str, host: str | None = None, concurrency: int = 1):
        self.model = model
        self.host = (host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.concurrency = concurrency  # a laptop serves one request at a time anyway
        self.name = f"local:{model}"

    async def generate(self, messages: list[dict]) -> GenResult:
        import httpx

        body = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": False,
            "format": "json",
            "options": {"temperature": 0, "num_predict": 160},
        }
        t0 = time.perf_counter()
        async with httpx.AsyncClient(timeout=120) as c:
            r = await c.post(f"{self.host}/api/chat", json=body)
            r.raise_for_status()
            d = r.json()
        dt = time.perf_counter() - t0
        return GenResult(
            d.get("message", {}).get("content", ""), dt, d.get("prompt_eval_count", 0), d.get("eval_count", 0)
        )


# ---------------------------------------------------------------------------
# Any OpenAI-compatible endpoint (optional closed-model baseline)
# ---------------------------------------------------------------------------


class OpenAIBackend(Backend):
    def __init__(self, model: str, concurrency: int = 8):
        from openai import AsyncOpenAI

        self.model = model
        self.concurrency = concurrency
        self.name = model
        self._client = AsyncOpenAI(base_url=os.environ.get("OPENAI_BASE_URL"))  # uses OPENAI_API_KEY

    async def generate(self, messages: list[dict]) -> GenResult:
        t0 = time.perf_counter()
        r = await self._client.chat.completions.create(
            model=self.model, messages=messages, temperature=0, max_tokens=160
        )
        dt = time.perf_counter() - t0
        u = r.usage
        return GenResult(r.choices[0].message.content or "", dt, u.prompt_tokens if u else 0, u.completion_tokens if u else 0)


def make_backend(spec: str, **kw) -> Backend:
    """spec: 'tinker:<base_model>[@<tinker://sampler_path>]' | 'ollama:<model>' | 'openai:<model>'"""
    kind, _, rest = spec.partition(":")
    if kind == "tinker":
        base, _, path = rest.partition("@")
        return TinkerBackend(base, model_path=path or None, **kw)
    if kind == "ollama":
        return OllamaBackend(rest, **kw)
    if kind == "openai":
        return OpenAIBackend(rest, **kw)
    raise ValueError(f"unknown backend spec: {spec}")


# $ per 1M tokens (input, output) for cost-per-1k-rows estimates.
# Tinker sampling prices from the models & pricing page (Oct 2026); edit if they change.
PRICES: dict[str, tuple[float, float]] = {
    "Qwen3.5-4B": (0.33, 1.005),
    "Qwen3.5-4B+lora": (0.33, 1.005),
    "Qwen3-8B": (0.195, 0.60),
    "Qwen3-8B+lora": (0.195, 0.60),
    "Qwen3.5-9B": (0.30, 0.807),
    "Qwen3.5-397B-A17B": (1.65, 7.50),
    "Kimi-K2.6": (2.205, 5.49),
}
