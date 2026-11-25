"""LLM router: ordered fallback, cooldowns, first-token timeout, malformed tool calls, token logging, total failure.
Uses fake providers (no network, no keys). Real-provider checks live in test_scenarios_live.py."""

from __future__ import annotations

import asyncio
import logging

import pytest
from livekit.agents import APIConnectionError, APIStatusError, llm
from livekit.agents.llm import ChatChunk, ChoiceDelta, CompletionUsage, FunctionToolCall
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS

from clinic import llm_router as r


# ---- fakes ---------------------------------------------------------------------------------------------------
class FakeStream(llm.LLMStream):
    async def _run(self) -> None:
        script = self._llm.script.pop(0) if self._llm.script else ("ok", "Hello there.")
        self._llm.calls += 1
        kind = script[0]
        if kind == "429":
            raise APIStatusError(message=script[1], status_code=429, request_id="r", body=None, retryable=True)
        if kind == "500":
            raise APIStatusError(message="server error", status_code=500, request_id="r", body=None, retryable=True)
        if kind == "402":
            raise APIStatusError(message="payment method required", status_code=402, request_id="r", body=None, retryable=False)
        if kind == "slow":
            await asyncio.sleep(script[1])
        if kind == "tool":
            self._event_ch.send_nowait(ChatChunk(id="c", delta=ChoiceDelta(role="assistant", tool_calls=[
                FunctionToolCall(name=script[1], arguments=script[2] if len(script) > 2 else "{}", call_id="t1")])))
        elif kind == "fail_midway":
            self._event_ch.send_nowait(ChatChunk(id="c", delta=ChoiceDelta(role="assistant", content="Part of an ans")))
            raise APIStatusError(message="dropped", status_code=500, request_id="r", body=None, retryable=True)
        else:
            self._event_ch.send_nowait(ChatChunk(id="c", delta=ChoiceDelta(role="assistant", content="")))
            self._event_ch.send_nowait(ChatChunk(id="c", delta=ChoiceDelta(role="assistant", content=script[1] if len(script) > 1 else "Hello there.")))
        self._event_ch.send_nowait(ChatChunk(id="c", usage=CompletionUsage(completion_tokens=5, prompt_tokens=100, total_tokens=105)))


class FakeLLM(llm.LLM):
    def __init__(self, model: str, provider: str, script: list | None = None) -> None:
        super().__init__()
        self._m, self._p, self.script, self.calls = model, provider, script or [], 0

    @property
    def model(self) -> str:
        return self._m

    @property
    def provider(self) -> str:
        return self._p

    def chat(self, *, chat_ctx, tools=None, conn_options=DEFAULT_API_CONNECT_OPTIONS, **kw):
        return FakeStream(self, chat_ctx=chat_ctx, tools=tools or [], conn_options=conn_options)


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


@llm.function_tool
async def get_balance_due() -> str:
    """Get the balance."""
    return "x"


def make_router(*scripts, **kw):
    clock = Clock()
    fakes = [FakeLLM(f"model-{i}", "prov", list(s)) for i, s in enumerate(scripts)]
    cands = [r.Candidate(f"c{i}", "prov", f.model, f) for i, f in enumerate(fakes)]
    return r.RouterLLM(cands, now=clock, **kw), fakes, clock


async def ask(router, tools=None):
    ctx = llm.ChatContext()
    ctx.add_message(role="user", content="hi")
    text, calls = "", []
    async with router.chat(chat_ctx=ctx, tools=tools or []) as stream:
        async for ch in stream:
            if ch.delta:
                text += ch.delta.content or ""
                calls += [tc.name for tc in ch.delta.tool_calls]
    return text, calls


# ---- parsing --------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("msg,expected", [("Please try again in 3.0375s.", 3.0375), ("try again in 52m24s", 3144.0),
                                          ("try again in 329.99ms", 0.32999), ("try again in 1h2m3s", 3723.0), ("nope", None)])
def test_retry_after_parsing(msg, expected):
    got = r.parse_retry_after(msg)
    assert (got is None and expected is None) or abs(got - expected) < 1e-3


def test_cooldowns_follow_the_provider_and_have_sane_bounds():
    e429 = lambda m: APIStatusError(message=m, status_code=429, request_id="r", body=None)  # noqa: E731
    assert abs(r.cooldown_for(e429("try again in 3.0375s")) - 3.5375) < 1e-3
    assert r.cooldown_for(e429("try again in 52m24s")) == 3144.5
    assert r.cooldown_for(e429("try again in 9999h")) == 3600.0               # capped
    assert r.cooldown_for(e429("try again in 1ms")) == 1.0                    # floored
    assert r.cooldown_for(e429("no hint")) == 30.0
    assert r.cooldown_for(APIStatusError(message="x", status_code=500, request_id="r", body=None)) == 30.0
    assert r.cooldown_for(APIStatusError(message="x", status_code=402, request_id="r", body=None)) == 3600.0
    assert r.cooldown_for(r.MalformedToolCall("bad")) == 120.0
    assert r.cooldown_for(asyncio.TimeoutError()) == 20.0


# ---- fallback order ----------------------------------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_primary_is_used_when_healthy():
    router, (a, b), _ = make_router([("ok", "from a")], [("ok", "from b")])
    text, _ = await ask(router)
    assert text == "from a" and (a.calls, b.calls) == (1, 0)


@pytest.mark.asyncio
async def test_429_falls_back_to_the_next_provider_in_order():
    router, (a, b, c), _ = make_router([("429", "try again in 5s")], [("ok", "from b")], [("ok", "from c")])
    text, _ = await ask(router)
    assert text == "from b" and (a.calls, b.calls, c.calls) == (1, 1, 0)


@pytest.mark.asyncio
async def test_server_error_and_second_failure_walk_down_the_list():
    router, (a, b, c), _ = make_router([("500",)], [("429", "try again in 5s")], [("ok", "from c")])
    text, _ = await ask(router)
    assert text == "from c" and (a.calls, b.calls, c.calls) == (1, 1, 1)


@pytest.mark.asyncio
async def test_a_limited_model_is_not_asked_again_until_its_cooldown_ends():
    router, (a, b), clock = make_router([("429", "Please try again in 10s."), ("ok", "a is back")],
                                       [("ok", "b1"), ("ok", "b2"), ("ok", "b3")])
    assert (await ask(router))[0] == "b1" and a.calls == 1
    clock.t += 5                                                # still inside the 10.5 s cooldown
    assert (await ask(router))[0] == "b2" and a.calls == 1      # a was NOT asked again
    clock.t += 6                                                # now past it
    assert (await ask(router))[0] == "a is back" and a.calls == 2


@pytest.mark.asyncio
async def test_cooldown_makes_no_background_requests():
    router, (a, b), clock = make_router([("429", "try again in 30s")], [("ok", "b")] * 5)
    await ask(router)
    clock.t += 3
    await asyncio.sleep(0.05)                                   # give any background "recovery" task a chance to run
    assert a.calls == 1                                         # LiveKit's stock adapter would have re-probed here


@pytest.mark.asyncio
async def test_slow_first_token_falls_back_within_the_budget():
    router, (a, b), _ = make_router([("slow", 5.0)], [("ok", "fast one")], first_token_timeout=0.2)
    text, _ = await ask(router)
    assert text == "fast one" and a.calls == 1 and b.calls == 1
    assert router.cooldown_until["c0"] > router.now()           # the slow provider rests


@pytest.mark.asyncio
async def test_unknown_tool_name_is_a_malformed_call_and_falls_back():
    router, (a, b), _ = make_router([("tool", "functions.not_a_real_tool")], [("tool", "get_balance_due")])
    text, calls = await ask(router, tools=[get_balance_due])
    assert calls == ["get_balance_due"] and (a.calls, b.calls) == (1, 1)


@pytest.mark.asyncio
async def test_a_valid_tool_call_passes_straight_through():
    router, (a, b), _ = make_router([("tool", "get_balance_due")], [("ok", "unused")])
    assert (await ask(router, tools=[get_balance_due]))[1] == ["get_balance_due"] and b.calls == 0


@pytest.mark.asyncio
async def test_payment_required_parks_a_provider_for_an_hour():
    router, (a, b), clock = make_router([("402",)], [("ok", "b")] * 3)
    await ask(router)
    clock.t += 1800
    await ask(router)
    assert a.calls == 1                                         # still parked after 30 minutes


@pytest.mark.asyncio
async def test_failure_after_speech_started_is_not_retried_elsewhere():
    router, (a, b), _ = make_router([("fail_midway",)], [("ok", "b")])
    with pytest.raises(APIStatusError):
        await ask(router)                                       # the caller already heard part of an answer
    assert b.calls == 0


@pytest.mark.asyncio
async def test_when_every_provider_fails_it_raises_a_connection_error():
    router, fakes, _ = make_router([("429", "try again in 5s")], [("500",)], [("429", "try again in 7s")])
    with pytest.raises(APIConnectionError, match="all LLM providers failed"):
        await ask(router)
    with pytest.raises(APIConnectionError, match="cooling down"):
        await ask(router)                                       # and immediately again: everyone is resting, fail fast
    assert [f.calls for f in fakes] == [1, 1, 1]                # nobody was hammered


# ---- logging ---------------------------------------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_one_usage_line_per_request_without_conversation_text(caplog):
    caplog.set_level(logging.INFO, logger="clinic.llm")
    router, _, _ = make_router([("ok", "Your balance is forty-five dollars, Sarah Johnson.")])
    await ask(router)
    e = router.usage_log[0]
    assert (e["prompt_tokens"], e["completion_tokens"], e["total_tokens"]) == (100, 5, 105) and e["model"] == "model-0"
    line = next(m for m in caplog.messages if m.startswith("LLM usage"))
    assert "prompt=100" in line and "completion=5" in line and "total=105" in line and "first_token_ms=" in line
    assert "Sarah" not in caplog.text and "forty-five" not in caplog.text


# ---- configuration -----------------------------------------------------------------------------------------------------
def test_providers_without_a_key_are_skipped_and_order_is_configurable():
    env = {"GROQ_API_KEY": "k", "sambanova_API_KEY": "k2"}                  # lowercase name, as in the real .env
    assert [c.name for c in r.build_candidates(env)] == ["groq-gpt-oss-120b", "groq-gpt-oss-20b", "groq-qwen", "sambanova"]
    assert [c.name for c in r.build_candidates({"GROQ_API_KEY": "k"})] == ["groq-gpt-oss-120b", "groq-gpt-oss-20b", "groq-qwen"]
    assert [c.name for c in r.build_candidates(env, order="sambanova, groq-qwen")] == ["sambanova", "groq-qwen"]
    assert [c.name for c in r.build_candidates({**env, "LLM_ORDER": "groq-qwen,bogus"})] == ["groq-qwen"]
    assert r.build_candidates({}) == []
    with pytest.raises(ValueError, match="no LLM candidate"):
        r.RouterLLM([])


def test_cloudflare_is_used_only_when_both_its_settings_exist():
    assert r.build_candidates({"CLOUDFLARE_API_TOKEN": "t"}, order="cloudflare") == []
    got = r.build_candidates({"CLOUDFLARE_API_TOKEN": "t", "CLOUDFLARE_ACCOUNT_ID": "acct"}, order="cloudflare")
    assert [c.name for c in got] == ["cloudflare"]


def test_the_model_ids_in_the_registry_are_the_verified_ones():
    ids = {c.name: c.model for c in r.build_candidates({"GROQ_API_KEY": "k", "SAMBANOVA_API_KEY": "k2"})}
    assert ids == {"groq-gpt-oss-120b": "openai/gpt-oss-120b", "groq-gpt-oss-20b": "openai/gpt-oss-20b",
                   "groq-qwen": "qwen/qwen3.8-27b", "sambanova": "gpt-oss-120b"}


# ---- Cloudflare Workers AI as the last fallback --------------------------------------------------------------------------
def test_cloudflare_is_the_last_default_candidate_and_accepts_either_key_name():
    assert r.DEFAULT_ORDER.split(",")[-1] == "cloudflare"
    base = {"GROQ_API_KEY": "gsk_test_not_real", "CLOUDFLARE_ACCOUNT_ID": "acct-not-real"}
    for name in ("CLOUDFLARE_API_TOKEN", "CLOUDFLARE_API_KEY", "cloudflare_API_KEY"):
        names = [c.name for c in r.build_candidates({**base, name: "cf-test-not-real"})]
        assert names[-1] == "cloudflare", name
    cf = r.build_candidates({**base, "CLOUDFLARE_API_KEY": "cf-test-not-real"})[-1]
    assert cf.model == "@cf/openai/gpt-oss-120b" and cf.provider == "cloudflare"


def test_cloudflare_is_skipped_without_the_account_id_or_a_key():
    only_key = {"GROQ_API_KEY": "gsk_test_not_real", "CLOUDFLARE_API_KEY": "cf-test-not-real"}
    assert "cloudflare" not in [c.name for c in r.build_candidates(only_key)]
    only_account = {"GROQ_API_KEY": "gsk_test_not_real", "CLOUDFLARE_ACCOUNT_ID": "acct-not-real"}
    assert "cloudflare" not in [c.name for c in r.build_candidates(only_account)]
