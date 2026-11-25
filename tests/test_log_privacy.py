"""Nothing personal in application logs, and no secrets in the repository. Offline."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pytest
from dotenv import dotenv_values

from clinic import booking_flow, callback_flow, gate, llm_router as lr, tools, verify_flow
from clinic.turn import assess_turn
from conftest import CLOSED_EVENING

ROOT = Path(__file__).resolve().parent.parent
PERSONAL = ["3344", "1985", "03-14", "March 14", "Sarah", "Johnson", "John Smith", "Jane Doe", "555-123", "555-111", "5551112222",
            "example.com", "gmail.com", "john@", "BS12345", "Blue Shield", "annual check-up", "billing question", "Metformin",
            "Lisinopril", "45.00", "1992-04-09", "April 9"]


def run_everything(st, call_state_closed):
    def hear(state, t):
        return assess_turn(state, gate.scrub_transcript(state, t))
    verify_flow.start(st)
    for t in ("Sarah Johnson", "yes", "March 14 1985", "three three four four"):
        hear(st, t)
    tools.prepare_refill(st, "metformin")
    hear(st, "yes")
    tools.speak_balance(st)
    hear(st, "what does my husband owe?")
    booking_flow.start(call_state_closed, "yes")
    for t in ("My name is John Smith", "yes", "April 9 1992", "555 123 4567", "john at example dot com", "Blue Shield",
              "B S 1 2 3 4 5", "annual check-up", "no preference", "the first one", "yes"):
        hear(call_state_closed, t)


def test_application_logs_contain_no_personal_details(caplog, make_state):
    caplog.set_level(logging.DEBUG)
    st, other = make_state(), make_state()
    run_everything(st, other)
    cb = make_state(CLOSED_EVENING)
    callback_flow.start(cb)
    for t in ("Jane Doe", "555 111 2222", "a billing question", "yes"):
        assess_turn(cb, t)
    text = caplog.text
    assert "audit" in text                                       # logging is on, so the absence below means something
    for needle in PERSONAL:
        assert needle.lower() not in text.lower(), f"{needle!r} leaked into the logs"


@pytest.mark.asyncio
async def test_llm_usage_and_failure_logs_have_no_conversation_text(caplog, state):
    caplog.set_level(logging.DEBUG)
    from livekit.agents import APIConnectionError, ChatContext
    import agent

    async def failing(self, chat_ctx, tools_, settings):
        raise APIConnectionError("all providers failed")
        yield  # pragma: no cover

    agent.Agent.default.llm_node, original = staticmethod(failing), agent.Agent.default.llm_node
    try:
        ctx = ChatContext()
        ctx.add_message(role="user", content="My name is Sarah Johnson and my birthday is March 14 1985")
        [c async for c in agent.Receptionist(state).llm_node(ctx, [], None)]
    finally:
        agent.Agent.default.llm_node = original
    assert "Sarah" not in caplog.text and "1985" not in caplog.text
    assert "LLM unavailable on every provider" in caplog.text


_SECRET_NAME = re.compile(r"key|secret|token|password|account_id", re.I)


def _secret_values(min_len: int) -> dict[str, str]:
    """The .env values that are credentials, judged by their NAME (key, secret, token, password, account id). Ordinary settings such
    as TTS_PROVIDER_ORDER=cloudflare,local are not secrets and legitimately appear in the documentation."""
    return {k: v for k, v in dotenv_values(ROOT / ".env").items() if v and len(v) >= min_len and _SECRET_NAME.search(k)}


def test_env_is_ignored_and_the_example_has_placeholders_only():
    ignore = (ROOT / ".gitignore").read_text(encoding="utf8").splitlines()
    assert ".env" in ignore and ".env.*" in ignore and "!.env.example" in ignore
    example = (ROOT / ".env.example").read_text(encoding="utf8")
    for name, value in _secret_values(13).items():               # a real key or secret
        assert value not in example, f"a real value for {name} is in .env.example"
    for line in example.splitlines():
        if re.match(r"^[A-Z_]+=", line):                         # active (uncommented) settings must be placeholders
            _, _, v = line.partition("=")
            assert v.startswith(("your-", "change-me", "wss://your-")) or v in ("", "router", "groq"), line


def test_no_real_secret_value_appears_in_any_project_file():
    skip = {".venv", "__pycache__", ".pytest_cache", "data"}
    secrets = _secret_values(16)
    for p in ROOT.rglob("*"):
        if p.is_file() and not set(p.parts) & skip and p.name != ".env" and p.suffix not in (".docx", ".db", ".pyc"):
            body = p.read_text(encoding="utf8", errors="ignore")
            for name, value in secrets.items():
                assert value not in body, f"the value of {name} appears in {p.relative_to(ROOT)}"


def test_router_log_lines_never_include_keys(caplog):
    caplog.set_level(logging.DEBUG)
    lr.build_candidates({"GROQ_API_KEY": "gsk_SECRETVALUE123456", "sambanova_API_KEY": "sn-SECRETVALUE654321"})
    assert "SECRETVALUE" not in caplog.text
