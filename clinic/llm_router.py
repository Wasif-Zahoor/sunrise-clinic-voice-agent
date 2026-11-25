"""LLM provider router: ordered fallback across OpenAI-compatible providers, built on LiveKit's own fallback classes.

Why not LiveKit's stock FallbackAdapter: after a failure it re-sends the whole conversation to the failed provider as a
"recovery check" (wasting tokens on an already limited key), ignores Retry-After, and cannot spot a malformed tool call.

What this adds (and nothing else is re-implemented, so streaming and tool calling work on every provider):
  * ordered candidates; providers whose API key is missing are skipped
  * a cooldown per candidate after a failure, taken from the provider's "try again in ..." text (429) or fixed defaults,
    with NO background re-probing: a cooled-down candidate is simply not asked until its time is up
  * first-token timeout (default 4 s): a slow candidate is abandoned for the next one
  * a malformed tool call (an unknown tool name) counts as a failure while nothing has been spoken yet
  * one INFO line per request: provider, model, prompt/completion/total tokens, time to first token (no conversation text)
  * if every candidate fails: APIConnectionError, which the agent turns into a short spoken apology plus a staff/callback offer

Configuration (names only; values stay in .env): LLM_ORDER (comma-separated candidate names), LLM_FIRST_TOKEN_TIMEOUT.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from livekit.agents import APIConnectionError, APIError, APIStatusError, llm
from livekit.agents.llm.fallback_adapter import (DEFAULT_FALLBACK_API_CONNECT_OPTIONS, FallbackAdapter,
                                                 FallbackLLMStream)
from livekit.agents.types import NOT_GIVEN

logger = logging.getLogger("clinic.llm")

DEFAULT_ORDER = "groq-gpt-oss-120b,groq-gpt-oss-20b,groq-qwen,sambanova,cloudflare"
SAMBANOVA_BASE_URL = "https://api.sambanova.ai/v1"          # verified against the live API (2026-11-24)
MIN_COOLDOWN, MAX_COOLDOWN = 1.0, 3600.0


class MalformedToolCall(APIError):
    """The model called a tool that does not exist (or similar): treat the candidate as failed."""

    def __init__(self, message: str) -> None:
        super().__init__(message, body=None, retryable=False)
        self.message = message


# ---- parsing the provider's "try again in 3.0375s / 52m24s / 329ms" ----------------------------------------------
_UNIT = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}


def parse_retry_after(message: str | None) -> float | None:
    m = re.search(r"try again in ([0-9hms.]+)", message or "", flags=re.I)
    if not m:
        return None
    parts = re.findall(r"(\d+(?:\.\d+)?)(ms|h|m|s)", m.group(1))
    return sum(float(n) * _UNIT[u] for n, u in parts) if parts else None


def cooldown_for(exc: BaseException) -> float:
    """How long to leave a candidate alone after this failure."""
    if isinstance(exc, MalformedToolCall):
        return 120.0
    if isinstance(exc, APIStatusError):
        if exc.status_code == 429:
            wait = parse_retry_after(f"{exc.message} {exc.body}")
            return min(max((wait + 0.5) if wait is not None else 30.0, MIN_COOLDOWN), MAX_COOLDOWN)
        if exc.status_code == 402:
            return 3600.0                                   # payment method required: needs a human, not a retry
        if exc.status_code in (400, 401, 403, 404):
            return 600.0                                    # misconfigured model or key: do not keep trying
        return 30.0                                         # 5xx and other
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return 20.0
    return 30.0


# ---- candidates --------------------------------------------------------------------------------------------------------
@dataclass
class Candidate:
    name: str
    provider: str
    model: str
    llm: llm.LLM


def _env_key(env: dict[str, str], *names: str) -> str:
    lowered = {k.lower(): v for k, v in env.items()}
    for n in names:
        value = env.get(n) or lowered.get(n.lower())
        if value and value.strip():
            return value.strip()
    return ""


def _make_groq(model: str, key: str, **kw: Any) -> llm.LLM:
    from livekit.plugins import groq
    return groq.LLM(model=model, api_key=key, temperature=0.4, **kw)


def _make_openai_compatible(model: str, key: str, base_url: str) -> llm.LLM:
    from livekit.plugins import openai
    return openai.LLM(model=model, api_key=key, base_url=base_url, temperature=0.4)


def _registry(env: dict[str, str]) -> dict[str, tuple[str, str, tuple[str, ...], Callable[[str], llm.LLM]]]:
    """name -> (provider, model, key env names, factory(key)). Model IDs verified against each provider's live model list."""
    cf_account = _env_key(env, "CLOUDFLARE_ACCOUNT_ID")
    return {
        "groq-gpt-oss-120b": ("groq", "openai/gpt-oss-120b", ("GROQ_API_KEY",),
                              lambda k: _make_groq("openai/gpt-oss-120b", k)),
        "groq-gpt-oss-20b": ("groq", "openai/gpt-oss-20b", ("GROQ_API_KEY",),
                             lambda k: _make_groq("openai/gpt-oss-20b", k)),
        "groq-qwen": ("groq", "qwen/qwen3.8-27b", ("GROQ_API_KEY",),
                      lambda k: _make_groq("qwen/qwen3.8-27b", k)),
        "sambanova": ("sambanova", "gpt-oss-120b", ("SAMBANOVA_API_KEY",),
                      lambda k: _make_openai_compatible("gpt-oss-120b", k, SAMBANOVA_BASE_URL)),
        # Last resort. Verified 2026-11-25 against this account: the model is in Cloudflare's catalog and the OpenAI-compatible
        # endpoint answered HTTP 200 (about 4 s; in one probe it replied in text instead of calling the tool).
        # Needs CLOUDFLARE_ACCOUNT_ID too; skipped when either is missing.
        "cloudflare": ("cloudflare", env.get("CLOUDFLARE_LLM_MODEL") or "@cf/openai/gpt-oss-120b",
                       ("CLOUDFLARE_API_TOKEN", "CLOUDFLARE_API_KEY"),
                       lambda k: _make_openai_compatible(env.get("CLOUDFLARE_LLM_MODEL") or "@cf/openai/gpt-oss-120b", k,
                                                         f"https://api.cloudflare.com/client/v4/accounts/{cf_account}/ai/v1")),
    }


def build_candidates(env: dict[str, str] | None = None, order: str | None = None) -> list[Candidate]:
    env = dict(os.environ if env is None else env)
    registry = _registry(env)
    names = [n.strip() for n in (order or env.get("LLM_ORDER") or DEFAULT_ORDER).split(",") if n.strip()]
    out: list[Candidate] = []
    for name in names:
        if name not in registry:
            logger.warning("LLM_ORDER names an unknown candidate %r (known: %s); skipped", name, ", ".join(registry))
            continue
        provider, model, key_names, factory = registry[name]
        key = _env_key(env, *key_names)
        if not key or (name == "cloudflare" and not _env_key(env, "CLOUDFLARE_ACCOUNT_ID")):
            missing = [key_names[0]] if not key else ["CLOUDFLARE_ACCOUNT_ID"]
            logger.info("LLM candidate %s skipped: %s is not set", name, " / ".join(missing))
            continue
        out.append(Candidate(name, provider, model, factory(key)))
    return out


# ---- the router ----------------------------------------------------------------------------------------------------------
class RouterLLM(FallbackAdapter):
    def __init__(self, candidates: list[Candidate], *, first_token_timeout: float | None = None,
                 max_total_seconds: float = 10.0, now: Callable[[], float] = time.monotonic) -> None:
        if not candidates:
            raise ValueError("no LLM candidate has an API key; check LLM_ORDER and the keys in .env")
        super().__init__([c.llm for c in candidates], attempt_timeout=max_total_seconds, max_retry_per_llm=0)
        self.candidates = candidates
        self.first_token_timeout = first_token_timeout or float(os.getenv("LLM_FIRST_TOKEN_TIMEOUT") or 4.0)
        self.max_total_seconds = max_total_seconds
        self.now = now
        self.cooldown_until: dict[str, float] = {}
        self.usage_log: list[dict[str, Any]] = []

    # which candidate would serve next (used for .model / .provider shown in metrics)
    def available(self) -> list[Candidate]:
        t = self.now()
        return [c for c in self.candidates if self.cooldown_until.get(c.name, 0.0) <= t]

    @property
    def model(self) -> str:
        return (self.available() or self.candidates)[0].model

    @property
    def provider(self) -> str:
        return (self.available() or self.candidates)[0].provider

    def note_failure(self, cand: Candidate, exc: BaseException) -> None:
        wait = cooldown_for(exc)
        self.cooldown_until[cand.name] = self.now() + wait
        logger.warning("LLM %s/%s failed (%s); cooling down %.0f s, trying the next provider", cand.provider, cand.model,
                       type(exc).__name__ + (f" {exc.status_code}" if isinstance(exc, APIStatusError) else ""), wait)

    def note_success(self, cand: Candidate, usage: Any, first_token_ms: float, total_ms: float) -> None:
        entry = {"provider": cand.provider, "model": cand.model, "name": cand.name, "first_token_ms": round(first_token_ms),
                 "total_ms": round(total_ms), "prompt_tokens": getattr(usage, "prompt_tokens", None),
                 "completion_tokens": getattr(usage, "completion_tokens", None),
                 "total_tokens": getattr(usage, "total_tokens", None), "at": self.now()}
        self.usage_log.append(entry)
        logger.info("LLM usage provider=%s model=%s prompt=%s completion=%s total=%s first_token_ms=%d total_ms=%d",
                    cand.provider, cand.model, entry["prompt_tokens"], entry["completion_tokens"], entry["total_tokens"],
                    entry["first_token_ms"], entry["total_ms"])

    def chat(self, *, chat_ctx, tools=None, conn_options=DEFAULT_FALLBACK_API_CONNECT_OPTIONS,
             parallel_tool_calls=NOT_GIVEN, tool_choice=NOT_GIVEN, extra_kwargs=NOT_GIVEN):  # type: ignore[override]
        return RouterStream(llm=self, conn_options=conn_options, chat_ctx=chat_ctx, tools=tools or [],
                            parallel_tool_calls=parallel_tool_calls, tool_choice=tool_choice, extra_kwargs=extra_kwargs)


class RouterStream(FallbackLLMStream):
    def _tool_names(self) -> set[str]:
        return {n for t in self._tools if (n := getattr(t, "id", None) or getattr(t, "name", None))}

    def _check(self, chunk: llm.ChatChunk, names: set[str]) -> None:
        if chunk.delta:
            for tc in chunk.delta.tool_calls:
                if tc.name and tc.name not in names:
                    raise MalformedToolCall(f"model called unknown tool {tc.name!r}")

    async def _run(self) -> None:
        router: RouterLLM = self._fallback_adapter  # type: ignore[assignment]
        started = router.now()
        names = self._tool_names()
        last_error: BaseException | None = None
        tried: list[str] = []
        for cand in router.available():
            if router.now() - started > router.max_total_seconds:
                break
            tried.append(cand.name)
            t0 = time.monotonic()
            gen = self._try_generate(llm=cand.llm, check_recovery=False)
            it = gen.__aiter__()
            held: list[llm.ChatChunk] = []              # chunks that carry nothing yet: held back until we know it is usable
            decided, first_ms, usage = False, 0.0, None
            try:
                while True:
                    timeout = router.first_token_timeout if not decided else None
                    try:
                        chunk = await asyncio.wait_for(it.__anext__(), timeout)
                    except StopAsyncIteration:
                        break
                    if chunk.usage is not None:
                        usage = chunk.usage
                    if not decided:
                        self._check(chunk, names)
                        has_payload = bool(chunk.delta and (chunk.delta.content or chunk.delta.tool_calls))
                        held.append(chunk)
                        if not has_payload:
                            continue
                        decided, first_ms = True, (time.monotonic() - t0) * 1000
                        for h in held:
                            self._event_ch.send_nowait(h)
                        held = []
                    else:
                        self._event_ch.send_nowait(chunk)
                for h in held:                           # a reply that never had text/tool payload still ends cleanly
                    self._event_ch.send_nowait(h)
                router.note_success(cand, usage, first_ms, (time.monotonic() - t0) * 1000)
                return
            except Exception as e:                       # noqa: BLE001
                router.note_failure(cand, e)
                last_error = e
                if decided:                              # something was already sent to the caller: cannot switch now
                    raise
            finally:
                await gen.aclose()
        raise APIConnectionError(f"all LLM providers failed or are cooling down (tried: {', '.join(tried) or 'none'})") \
            from last_error


def build_router() -> RouterLLM:
    return RouterLLM(build_candidates())
