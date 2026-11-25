"""LLM provider switch (Groq default, Gemini opt-in). Offline: no network, no real keys."""

from __future__ import annotations

import pytest

from clinic import config


@pytest.fixture
def clean_env(monkeypatch):
    for name in ("LLM_PROVIDER", "GEMINI_API_KEY", "GEMINI_LLM_MODEL", "GROQ_API_KEY", "LIVEKIT_URL",
                 "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_default_provider_is_the_fallback_router(clean_env):
    assert config.llm_provider() == "router"
    assert config.get_settings().llm_provider == "router"


def test_gemini_needs_its_own_key_and_groq_keys_stay_required(clean_env):
    clean_env.setenv("LLM_PROVIDER", "gemini")
    assert "GEMINI_API_KEY" in config.missing_env()
    clean_env.setenv("GEMINI_API_KEY", "placeholder-not-real")
    assert "GEMINI_API_KEY" not in config.missing_env()
    assert "GROQ_API_KEY" in config.missing_env()          # STT and TTS still use Groq


def test_groq_provider_does_not_require_gemini_key(clean_env):
    clean_env.setenv("LLM_PROVIDER", "groq")
    assert "GEMINI_API_KEY" not in config.missing_env()


def test_unknown_provider_fails_with_a_clear_message(clean_env):
    clean_env.setenv("LLM_PROVIDER", "openai")
    with pytest.raises(SystemExit, match="LLM_PROVIDER must be one of router, groq, gemini"):
        config.get_settings()


def test_gemini_model_default_and_override(clean_env):
    assert config.get_settings().gemini_llm_model == "gemini-3.5-flash"
    clean_env.setenv("GEMINI_LLM_MODEL", "gemini-2.5-flash-lite")
    assert config.get_settings().gemini_llm_model == "gemini-2.5-flash-lite"


def test_build_llm_picks_the_right_plugin(clean_env, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "placeholder-not-real")
    monkeypatch.setenv("GEMINI_API_KEY", "placeholder-not-real")
    import agent
    from livekit.plugins import google, groq

    groq_llm = agent.build_llm(config.Settings("w", "openai/gpt-oss-120b", "t", "v", 9, "groq", "gemini-2.5-flash"))
    assert isinstance(groq_llm, groq.LLM) and groq_llm.model == "openai/gpt-oss-120b"
    gem = agent.build_llm(config.Settings("w", "openai/gpt-oss-120b", "t", "v", 9, "gemini", "gemini-2.5-flash"))
    assert isinstance(gem, google.LLM) and gem.model == "gemini-2.5-flash"


async def test_session_keeps_the_same_voice_pipeline_with_either_llm(state, clean_env, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "placeholder-not-real")
    monkeypatch.setenv("GEMINI_API_KEY", "placeholder-not-real")
    monkeypatch.setenv("TTS_PROVIDER_ORDER", "groq")                       # the default chain depends on the keys in .env
    import agent
    for provider in ("groq", "gemini"):
        s = config.Settings("whisper-large-v3-turbo", "openai/gpt-oss-120b", "canopylabs/orpheus-v1-english", "hannah",
                            9, provider, "gemini-3.5-flash")
        sess = agent.build_session(settings=s, state=state)
        assert sess.tts.model == "canopylabs/orpheus-v1-english"          # the TTS chain is the same for either LLM
        assert sess.options.preemptive_generation["enabled"] is False
        assert sess.options.interruption["mode"] == "vad"


def test_live_tests_skip_instead_of_passing_when_the_llm_was_down():
    import agent
    from conftest import require_real_llm_reply
    assert require_real_llm_reply("We close at 5 PM.") == "We close at 5 PM."
    with pytest.raises(pytest.skip.Exception):
        require_real_llm_reply(agent.LLM_FALLBACK_SAY)


@pytest.mark.asyncio
async def test_llm_failure_fallback_uses_the_shared_constant(state, monkeypatch):
    import agent
    from livekit.agents import APIConnectionError, ChatContext

    async def failing(self, chat_ctx, tools_, settings):
        raise APIConnectionError("quota exhausted")
        yield  # pragma: no cover

    monkeypatch.setattr(agent.Agent.default, "llm_node", staticmethod(failing))
    out = [c async for c in agent.Receptionist(state).llm_node(ChatContext(), [], None)]
    assert out == [agent.LLM_FALLBACK_SAY]
