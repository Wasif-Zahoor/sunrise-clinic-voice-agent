"""Phase 1 automated checks. These do NOT test the microphone or audio quality.

Offline tests always run. The `live` test calls the Groq LLM and is skipped without GROQ_API_KEY.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clinic import config  # noqa: E402
from clinic.prompts import GREETING, INSTRUCTIONS, clean_for_speech  # noqa: E402

config.load_env()  # so the live test can see GROQ_API_KEY from .env


def test_missing_env_reports_names_only(monkeypatch):
    for name in config.REQUIRED_ENV:
        monkeypatch.delenv(name, raising=False)
    assert config.missing_env() == list(config.REQUIRED_ENV)
    monkeypatch.setenv("GROQ_API_KEY", "   ")  # blank counts as missing
    assert "GROQ_API_KEY" in config.missing_env()
    monkeypatch.setenv("GROQ_API_KEY", "x")
    assert "GROQ_API_KEY" not in config.missing_env()


def test_openai_key_not_required():
    assert not any("OPENAI" in n for n in config.REQUIRED_ENV)


def test_default_models_are_current_groq_ids(monkeypatch):
    for n in ("STT_MODEL", "LLM_MODEL", "TTS_MODEL", "TTS_VOICE"):
        monkeypatch.delenv(n, raising=False)
    s = config.get_settings()
    assert s.stt_model == "whisper-large-v3-turbo"
    assert s.llm_model == "openai/gpt-oss-120b"
    assert s.tts_model == "canopylabs/orpheus-v1-english"
    assert "playai" not in s.tts_model  # retired by Groq on 2025-12-31


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "openai/gpt-oss-20b")
    assert config.get_settings().llm_model == "openai/gpt-oss-20b"


def test_greeting_announces_recording_and_transcription_and_fits_tts_limit():
    # The assignment requires a "may be recorded ... for quality purposes" notice; transcripts are on by default.
    from clinic.clock import Clock
    from clinic.prompts import build_greeting
    assert "may be recorded and transcribed for quality purposes" in GREETING
    assert "Sunrise Family Clinic" in GREETING
    # With transcripts switched off nothing is kept, and the greeting says so truthfully.
    off = build_greeting(Clock(), transcripts=False)
    assert "not being recorded" in off and "transcribed" not in off
    # Groq Orpheus rejects input over 200 characters; LiveKit speaks one sentence per request.
    for sentence in GREETING.split(". "):
        assert len(sentence) < 200


def test_clean_for_speech():
    assert clean_for_speech("We close at 5 PM — see you") == "We close at 5 PM, see you"
    assert clean_for_speech("**Dr. Patel**") == "Dr. Patel"


def test_prompt_keeps_clinic_facts_and_ssn_rule():
    assert "9 AM to 5 PM" in INSTRUCTIONS
    assert "Never ask for the full number" in INSTRUCTIONS


async def test_agent_module_builds_session(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")       # this test checks the Groq default regardless of .env
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")
    monkeypatch.setenv("TTS_PROVIDER_ORDER", "groq")  # and the Groq TTS chain (the default chain depends on .env keys)
    import agent

    session = agent.build_session()
    assert session.stt is not None and session.llm is not None and session.tts is not None
    assert session.tts.model == "canopylabs/orpheus-v1-english"
    assert session.stt.model == "whisper-large-v3-turbo"
    assert session.llm.model == "openai/gpt-oss-120b"


@pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="needs GROQ_API_KEY")
async def test_live_llm_answers_hours_briefly(pace, state):
    from livekit.agents import AgentSession
    from livekit.plugins import groq

    import agent

    llm = agent.build_llm(config.get_settings())   # the provider .env selects
    async with AgentSession(llm=llm) as sess:
        await sess.start(agent.Receptionist(state))
        result = await sess.run(user_input="What time do you close on Fridays?")
        msg = result.expect.next_event().is_message(role="assistant")
        from conftest import require_real_llm_reply
        text = require_real_llm_reply(msg.event().item.text_content or "")
        print("\nLLM reply:", ascii(text))
        assert "5" in text
        assert len(text) < 400
    await llm.aclose()


async def test_preemptive_generation_disabled_and_interruptions_unchanged(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")
    import agent

    opts = agent.build_session().options
    assert opts.preemptive_generation["enabled"] is False
    # Interruption behaviour must stay as verified in the Phase 1 browser test.
    assert opts.turn_handling["turn_detection"] == "vad"
    assert opts.interruption["mode"] == "vad"
    assert opts.interruption.get("enabled", True) is True
    assert opts.interruption["min_duration"] == 0.5
    assert opts.interruption["resume_false_interruption"] is True


def test_pii_redaction_filter_hides_speech_but_keeps_technical_fields():
    import io
    import logging

    from clinic.logging_privacy import RedactPII, install_pii_redaction

    stream = io.StringIO()

    class ExtraFormatter(logging.Formatter):  # mimics LiveKit: prints record extras
        def format(self, record):
            extras = {k: v for k, v in record.__dict__.items() if k.startswith(("lk.", "tts"))}
            return f"{record.getMessage()} {extras}"

    handler = logging.StreamHandler(stream)
    handler.setFormatter(ExtraFormatter())
    root = logging.getLogger()
    root.addHandler(handler)
    old_level = root.level
    root.setLevel(logging.DEBUG)
    try:
        assert install_pii_redaction() >= 1
        install_pii_redaction()  # idempotent
        assert sum(isinstance(f, RedactPII) for f in handler.filters) == 1
        log = logging.getLogger("test.privacy")
        log.debug(
            "conversation_item_added",
            extra={"lk.pii.text": "my birthday is March 14 1985", "lk.pii.user_transcript": "3344", "tts": "groq"},
        )
    finally:
        root.removeHandler(handler)
        root.setLevel(old_level)
    out = stream.getvalue()
    assert "1985" not in out and "3344" not in out and "birthday" not in out
    assert "[redacted]" in out and "groq" in out


def test_prompt_covers_stop_goodbye_and_new_question():
    low = INSTRUCTIONS.lower()
    assert "stop" in low and "do not end the call" in low
    assert "goodbye" in low
    assert "new question" in low


async def _say(sess, text):
    result = await sess.run(user_input=text)
    msg = result.expect.next_event().is_message(role="assistant")
    from conftest import require_real_llm_reply
    return require_real_llm_reply((msg.event().item.text_content or "").strip())


@pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="needs GROQ_API_KEY")
async def test_live_conversation_behaviours(pace, state):
    """Text-level checks of stop / goodbye / new-question behaviour. Not an audio test."""
    from livekit.agents import AgentSession
    from livekit.plugins import groq

    import agent

    llm = agent.build_llm(config.get_settings())   # the provider .env selects
    async with AgentSession(llm=llm) as sess:
        await sess.start(agent.Receptionist(state))

        r = await _say(sess, "Which doctors do you have?")
        print("Q doctors   ->", ascii(r))
        assert "Patel" in r and "Lee" in r

        # interrupt with a new question
        r = await _say(sess, "Actually, where are you located?")
        print("Q address   ->", ascii(r))
        assert "Main" in r

        # "stop" must not end the call or say goodbye
        r = await _say(sess, "Stop.")
        print("Q stop      ->", ascii(r))
        assert len(r) < 60
        assert not any(w in r.lower() for w in ("goodbye", "bye", "take care", "have a"))

        # speaking immediately after the stop
        r = await _say(sess, "What time do you open?")
        print("Q open      ->", ascii(r))
        assert "9" in r

        # goodbye gives a short farewell with no follow-up question
        r = await _say(sess, "Okay, thanks. Goodbye.")
        print("Q goodbye   ->", ascii(r))
        assert "?" not in r and len(r) < 120
    await llm.aclose()
