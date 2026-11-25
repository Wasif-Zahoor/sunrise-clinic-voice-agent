"""Environment-based settings. Secrets come from .env only and are never printed."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

REQUIRED_ENV = (
    "LIVEKIT_URL",
    "LIVEKIT_API_KEY",
    "LIVEKIT_API_SECRET",
    "GROQ_API_KEY",
)

# Defaults verified against Groq's live /models list and docs on 2026-11-24.
DEFAULT_STT_MODEL = "whisper-large-v3-turbo"
DEFAULT_LLM_MODEL = "openai/gpt-oss-120b"
DEFAULT_TTS_MODEL = "canopylabs/orpheus-v1-english"
DEFAULT_TTS_VOICE = "hannah"


DEFAULT_LLM_PROVIDER = "router"            # "router" (default: ordered fallback), "groq" (one model) or "gemini"
DEFAULT_GEMINI_LLM_MODEL = "gemini-3.5-flash"  # 2.5 models are closed to new users; free key = 20 requests/day
LLM_PROVIDERS = ("router", "groq", "gemini")


@dataclass(frozen=True)
class Settings:
    stt_model: str
    llm_model: str
    tts_model: str
    tts_voice: str
    tts_max_rpm: int = 9
    llm_provider: str = DEFAULT_LLM_PROVIDER
    gemini_llm_model: str = DEFAULT_GEMINI_LLM_MODEL


def load_env() -> None:
    """Load .env from the project root (existing environment variables win)."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)


def llm_provider() -> str:
    value = (os.getenv("LLM_PROVIDER") or DEFAULT_LLM_PROVIDER).strip().lower()
    if value not in LLM_PROVIDERS:
        raise SystemExit(f"LLM_PROVIDER must be one of {', '.join(LLM_PROVIDERS)} (got {value!r}).")
    return value


def transcripts_enabled() -> bool:
    """TRANSCRIPTS=off|0|false|no disables per-call transcript files (default: on)."""
    return (os.getenv("TRANSCRIPTS") or "on").strip().lower() not in ("off", "0", "false", "no", "disabled")


def transcript_dir() -> Path:
    raw = (os.getenv("TRANSCRIPT_DIR") or "transcripts").strip()
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path


def missing_env() -> list[str]:
    """Names (never values) of required variables that are unset or empty."""
    missing = [name for name in REQUIRED_ENV if not os.getenv(name, "").strip()]
    if llm_provider() == "gemini" and not os.getenv("GEMINI_API_KEY", "").strip():
        missing.append("GEMINI_API_KEY")
    return missing


def get_settings() -> Settings:
    return Settings(
        stt_model=os.getenv("STT_MODEL") or DEFAULT_STT_MODEL,
        llm_model=os.getenv("LLM_MODEL") or DEFAULT_LLM_MODEL,
        tts_model=os.getenv("TTS_MODEL") or DEFAULT_TTS_MODEL,
        tts_voice=os.getenv("TTS_VOICE") or DEFAULT_TTS_VOICE,
        tts_max_rpm=int(os.getenv("TTS_MAX_RPM") or 9),
        llm_provider=llm_provider(),
        gemini_llm_model=os.getenv("GEMINI_LLM_MODEL") or DEFAULT_GEMINI_LLM_MODEL,
    )
