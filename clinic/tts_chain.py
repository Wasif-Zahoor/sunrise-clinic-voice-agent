"""Text-to-speech with a fallback chain, so the caller does not hear silence when one provider fails.

TTS_PROVIDER_ORDER (names only; keys stay in .env) lists providers, first choice first. Default: deepgram,cloudflare,local.
  deepgram    Deepgram Aura-2 through the official LiveKit plugin (needs DEEPGRAM_API_KEY)
  local       Piper, in-process on the CPU: no key, no quota, no network (needs the voice file in models/piper/)
  cloudflare  Workers AI @cf/deepgram/aura-2-en (needs CLOUDFLARE_API_TOKEN or CLOUDFLARE_API_KEY and CLOUDFLARE_ACCOUNT_ID)
  groq        Groq Orpheus with the older pacing / grouping safeguards (clinic.tts_safety)
A provider without its key or model file is skipped (the log names the missing VARIABLE, never a value).

When a provider fails the chain speaks that chunk with the next one. A permanent failure (401/402/403/429: bad key, no credit, limit)
switches FOR THE REST OF THE CALL. A transient one (timeout, connection error, 5xx, no audio) only costs that chunk: the primary is
tried again on the next chunk, and two in a row make it permanent (so one network hiccup does not change the voice for good).
It deliberately does not use the framework's FallbackAdapter, whose recovery probes re-send real text to the failed provider
and would spend a one-time credit. If every provider fails the chunk is replaced by 0.4 s of silence, as before, so the call
survives, and `stats["degraded"]` counts it (the greeting check in agent.py reads it).
Nothing here logs spoken text: only provider names, error types and character counts.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import aiohttp
from livekit import rtc
from livekit.agents import (DEFAULT_API_CONNECT_OPTIONS, APIConnectionError, APIConnectOptions, APIStatusError,
                            APITimeoutError, tts, utils)

from . import config, voice_diag
from .llm_router import _env_key
from .tts_budget import CharBudget
from .tts_health import COOLDOWN_PERMANENT, COOLDOWN_TRANSIENT, ProviderHealth
from .tts_health import enabled as health_enabled

logger = logging.getLogger("clinic.tts")

DEFAULT_ORDER = "deepgram,cloudflare,local"            # a provider whose keys are missing is skipped, so this works with any subset
HOLD_BASE = 3.0                                        # whole-clip deadline: this many seconds ...
HOLD_PER_AUDIO_SECOND = 0.6                            # ... plus this share of the estimated speech length
CHARS_PER_AUDIO_SECOND = 16.0                          # measured: about 17-18 characters of text per second of Aura-2 speech
DEFAULT_DEEPGRAM_MODEL = "aura-2-thalia-en"            # verified on this account 2026-11-25 (also aura-2-andromeda-en)
DEFAULT_CLOUDFLARE_MODEL = "@cf/deepgram/aura-2-en"    # listed by Cloudflare's model search for this account
# Chosen by measurement against Deepgram's aura-2-thalia-en (female): closest pitch (225 vs 220 Hz) and brightness of the female Piper
# voices compared; pace and loudness are matched by the two settings below, so the switch is hard to notice. See README.
DEFAULT_PIPER_MODEL = "models/piper/en_US-hfc_female-medium.onnx"
DEFAULT_PIPER_LENGTH_SCALE = 1.04      # speaking pace (1.0 = the voice's own pace; Thalia is slightly slower)
DEFAULT_PIPER_VOLUME = 0.4             # Piper's default output is about 8 dB louder than Deepgram's
CHAIN_SAMPLE_RATE = 24000
FIRST_AUDIO_TIMEOUT = 8.0                              # seconds to the first audio before moving on
SILENCE_SECONDS = 0.4
PERMANENT_STATUS = {401, 402, 403, 429}
MAX_STRIKES = 2                                        # consecutive transient failures before a provider is given up on


# ---- local provider: Piper ------------------------------------------------------------------------------------------------
class PiperTTS(tts.TTS):
    """Piper neural TTS on the CPU. Not streaming: the chain feeds it one grouped chunk at a time."""

    def __init__(self, model_path: str | Path, length_scale: float = DEFAULT_PIPER_LENGTH_SCALE,
                 volume: float = DEFAULT_PIPER_VOLUME) -> None:
        self._path = Path(model_path)
        self._length_scale, self._volume = length_scale, volume
        cfg = json.loads(Path(str(self._path) + ".json").read_text(encoding="utf-8"))
        super().__init__(capabilities=tts.TTSCapabilities(streaming=False),
                         sample_rate=int(cfg["audio"]["sample_rate"]), num_channels=1)
        self._model_name = self._path.stem
        self._voice: Any = None
        self._lock = threading.Lock()

    @property
    def model(self) -> str:
        return self._model_name

    @property
    def provider(self) -> str:
        return "piper"

    def _load(self) -> Any:
        with self._lock:                                   # a warm-up thread and a first request may race
            if self._voice is None:
                from piper import PiperVoice
                self._voice = PiperVoice.load(str(self._path))
            return self._voice

    def sentences(self, text: str):
        """Audio bytes one sentence at a time, so the first sentence can start playing while the rest is still being made."""
        from piper import SynthesisConfig
        voice = self._load()
        config = SynthesisConfig(length_scale=self._length_scale, volume=self._volume)
        for chunk in voice.synthesize(text, config):
            yield chunk.audio_int16_bytes

    def render(self, text: str) -> bytes:
        return b"".join(self.sentences(text))

    def prewarm(self) -> None:
        """Load the model and run one short sentence in a thread: the very first synthesis is slow (about 4 s here)."""
        def warm() -> None:
            try:
                self.render("Hello.")
            except Exception as e:                         # noqa: BLE001  (a warm-up failure only costs latency later)
                logger.warning("local TTS warm-up failed (%s)", type(e).__name__)
        threading.Thread(target=warm, daemon=True, name="piper-warmup").start()

    def synthesize(self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS) -> tts.ChunkedStream:
        return _PiperStream(tts=self, input_text=text, conn_options=conn_options)


class _PiperStream(tts.ChunkedStream):
    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        pieces = self._tts.sentences(self._input_text)
        first = await asyncio.to_thread(next, pieces, b"")                         # CPU-bound: keep the event loop free
        if not first:
            raise APIConnectionError("local TTS produced no audio")
        output_emitter.initialize(request_id=utils.shortuuid(), sample_rate=self._tts.sample_rate, num_channels=1,
                                  mime_type="audio/pcm")
        output_emitter.push(first)                                                  # playing already; the rest follows
        while (more := await asyncio.to_thread(next, pieces, b"")):
            output_emitter.push(more)
        output_emitter.flush()


# ---- optional provider: Cloudflare Workers AI (Deepgram Aura-2 hosted by Cloudflare) -----------------------------------------
class CloudflareTTS(tts.TTS):
    def __init__(self, *, account_id: str, token: str, model: str = DEFAULT_CLOUDFLARE_MODEL, base_url: str | None = None,
                 http_session: aiohttp.ClientSession | None = None, speaker: str | None = None) -> None:
        super().__init__(capabilities=tts.TTSCapabilities(streaming=False), sample_rate=CHAIN_SAMPLE_RATE, num_channels=1)
        self._model_name, self._token, self._session, self._speaker = model, token, http_session, speaker
        self._url = f"{base_url or 'https://api.cloudflare.com/client/v4'}/accounts/{account_id}/ai/run/{model}"

    @property
    def model(self) -> str:
        return self._model_name

    @property
    def provider(self) -> str:
        return "cloudflare"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = utils.http_context.http_session()
        return self._session

    def synthesize(self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS) -> tts.ChunkedStream:
        return _CloudflareStream(tts=self, input_text=text, conn_options=conn_options)


class _CloudflareStream(tts.ChunkedStream):
    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        owner = self._tts
        body = {"text": self._input_text, "encoding": "linear16", "container": "none", "sample_rate": CHAIN_SAMPLE_RATE}
        if owner._speaker:                                  # the same voice as Deepgram's aura-2-<speaker>-en, so a switch is not heard
            body["speaker"] = owner._speaker
        try:
            async with owner._ensure_session().post(
                owner._url, headers={"Authorization": f"Bearer {owner._token}", "Content-Type": "application/json"},
                json=body,
                timeout=aiohttp.ClientTimeout(total=30, sock_connect=self._conn_options.timeout),
            ) as resp:
                if resp.status >= 400:
                    raise APIStatusError(message=f"Cloudflare TTS HTTP {resp.status}", status_code=resp.status,
                                         request_id=None, body=None)
                data = await resp.read()
            if not data:
                raise APIConnectionError("Cloudflare TTS returned no audio")
            output_emitter.initialize(request_id=utils.shortuuid(), sample_rate=CHAIN_SAMPLE_RATE, num_channels=1,
                                      mime_type="audio/pcm")
            output_emitter.push(data)
            output_emitter.flush()
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except (APIStatusError, APIConnectionError):
            raise
        except aiohttp.ClientError as e:
            raise APIConnectionError() from e


# ---- the chain ----------------------------------------------------------------------------------------------------------------
@dataclass
class Provider:
    name: str
    tts: tts.TTS
    hold: bool = False         # network providers: keep each clip until complete (see _speak_with); the local voice streams


def _describe(e: BaseException) -> str:
    """Error type and HTTP status only. Never the message: it could echo request text."""
    code = getattr(e, "status_code", None)
    cause = type(e.__cause__).__name__ if e.__cause__ is not None else None
    return f"{type(e).__name__}{f' HTTP {code}' if code else ''}{f' caused by {cause}' if cause else ''}"


class TtsChain(tts.TTS):
    def __init__(self, providers: list[Provider], budget: CharBudget | None = None, *,
                 sample_rate: int = CHAIN_SAMPLE_RATE, first_audio_timeout: float = FIRST_AUDIO_TIMEOUT,
                 health: ProviderHealth | None = None) -> None:
        super().__init__(capabilities=tts.TTSCapabilities(streaming=False), sample_rate=sample_rate, num_channels=1)
        self.providers = providers
        self.budget = budget or CharBudget()
        self.first_audio_timeout = first_audio_timeout
        self.health = health                               # None: no memory between calls (the tests, TTS_HEALTH=off)
        self.dead: set[str] = set()                        # providers given up on for the rest of this call
        self._marked_down: set[str] = set()                # providers this call wrote into the health file
        self._skip_recently_failed()
        self.strikes: dict[str, int] = {}                  # consecutive transient failures per provider
        self.active: str | None = None                     # who served the most recent chunk
        self.stats = {"degraded": 0, "fallbacks": 0}
        self.hold = True                                   # TTS_HOLD=0 turns the whole-clip hold off (see _speak_with)
        self.monitor = voice_diag.LoopMonitor()            # started by the agent once the event loop runs
        self.chunks: list[dict] = []                       # one entry per spoken chunk, for the end-of-call summary
        self.last_done: dict[str, float] = {}              # when each provider last finished a chunk (idle-connection hint)

    def _skip_recently_failed(self) -> None:
        """A provider the previous call gave up on is not asked again for a few minutes (see clinic/tts_health.py). The chain is
        never left without a provider: if every one is cooling down, all of them are tried."""
        if self.health is None:
            return
        cooling = self.health.cooling()
        skipped = [p.name for p in self.providers if p.name in cooling]
        if not skipped or len(skipped) == len(self.providers):
            return
        for name in skipped:
            self.dead.add(name)
            logger.warning("TTS provider %s skipped for this call: it failed shortly before (%s); it is tried again in about %d min",
                           name, cooling[name]["why"], cooling[name]["minutes_left"])

    def _remember_down(self, name: str, permanent: bool, why: str) -> None:
        if self.health is not None:
            self.health.mark_down(name, COOLDOWN_PERMANENT if permanent else COOLDOWN_TRANSIENT, why)
            self._marked_down.add(name)

    def _remember_ok(self, name: str) -> None:
        if self.health is not None and name in self._marked_down:
            self._marked_down.discard(name)
            self.health.mark_ok(name)

    def log_summary(self) -> None:
        line = voice_diag.summary(self.chunks)
        if line:
            logger.info(line)

    @property
    def model(self) -> str:
        return self.providers[0].tts.model if self.providers else "none"

    @property
    def provider(self) -> str:
        return "tts-chain"

    def prewarm(self) -> None:
        for p in self.providers:
            if p.name == "local":                          # loading the CPU model is the only warm-up that is not free of cost
                p.tts.prewarm()

    def synthesize(self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS) -> tts.ChunkedStream:
        return _ChainStream(tts=self, input_text=text, conn_options=conn_options)

    async def aclose(self) -> None:
        for p in self.providers:
            try:
                await p.tts.aclose()
            except Exception:                              # noqa: BLE001
                pass


class _ChainStream(tts.ChunkedStream):
    async def _frames(self, p: Provider):
        chain: TtsChain = self._tts
        resampler = None
        opts = APIConnectOptions(max_retry=0, timeout=chain.first_audio_timeout)      # no hidden retries: fall through instead
        async with p.tts.synthesize(self._input_text, conn_options=opts) as stream:
            it, first = stream.__aiter__(), True
            while True:
                try:
                    ev = await (asyncio.wait_for(it.__anext__(), chain.first_audio_timeout) if first else it.__anext__())
                except StopAsyncIteration:
                    break
                first = False
                frame = ev.frame
                if frame.sample_rate != chain.sample_rate:
                    resampler = resampler or rtc.AudioResampler(input_rate=frame.sample_rate, output_rate=chain.sample_rate,
                                                                num_channels=1)
                    for rf in resampler.push(frame):
                        yield rf
                else:
                    yield frame
            if resampler is not None:
                for rf in resampler.flush():
                    yield rf
        if first:
            raise APIConnectionError("provider returned no audio")

    def _deadline(self) -> float:
        """Seconds a whole clip may take: a fixed allowance plus a share of the (estimated) length of the speech."""
        return HOLD_BASE + HOLD_PER_AUDIO_SECOND * len(self._input_text) / CHARS_PER_AUDIO_SECOND

    async def _speak_with(self, p: Provider, output_emitter: tts.AudioEmitter) -> tuple[bool, BaseException | None, bool]:
        """Speak this chunk with `p` -> (success, error, audio_already_pushed)."""
        chain: TtsChain = self._tts
        pushed, retried = False, False
        while True:
            last = chain.last_done.get(p.name)
            trace = voice_diag.ChunkTrace(p.name, len(self._input_text), None if last is None else time.perf_counter() - last)
            held = p.hold and chain.hold
            buffered: list[rtc.AudioFrame] = []
            try:
                # A network provider's audio can arrive in bursts with long pauses (measured from this PC: one clip in three
                # stalled for 12-30 s). Playing it as it arrives turns every pause into a break in the voice. So the whole clip is
                # kept until it is complete and then played at once; a clip not complete by its deadline is dropped BEFORE
                # anything was played and the next provider speaks that chunk, so the caller never hears half a sentence.
                try:
                    async with asyncio.timeout(self._deadline() if held else None):
                        async for frame in self._frames(p):
                            trace.frame(frame.samples_per_channel / frame.sample_rate)
                            if held:
                                buffered.append(frame)
                            else:
                                output_emitter.push_frame(frame)
                                pushed = True
                except TimeoutError:
                    raise APITimeoutError() from None
                for frame in buffered:
                    output_emitter.push_frame(frame)
                    pushed = True
                output_emitter.flush()
                chain.budget.note(p.name, len(self._input_text))          # counted on success: that is what is billed
                chain.last_done[p.name] = time.perf_counter()
                result = trace.result(lag=chain.monitor.worst_since(trace.started), held=held)
                chain.chunks.append(result)
                voice_diag.report_chunk(result)
                chain.strikes[p.name] = 0
                chain.active = p.name
                chain._remember_ok(p.name)
                return True, None, pushed
            except asyncio.CancelledError:
                raise
            except Exception as e:                                         # noqa: BLE001
                # A dropped idle keep-alive connection (ServerDisconnectedError) fails instantly and works on a new connection:
                # retry the SAME provider once before anything else, so the caller's voice does not change because of it.
                # (A timeout is a subclass of APIConnectionError but is never retried: that would wait out the deadline twice.)
                if isinstance(e, APIConnectionError) and not isinstance(e, APITimeoutError) and not pushed and not retried:
                    retried = True
                    logger.info("TTS provider %s: connection error (%s); retrying once", p.name, _describe(e))
                    continue
                return False, e, pushed

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        chain: TtsChain = self._tts
        output_emitter.initialize(request_id=utils.shortuuid(), sample_rate=chain.sample_rate, num_channels=1,
                                  mime_type="audio/pcm")
        if chain.providers and all(p.name in chain.dead for p in chain.providers):
            logger.error("every TTS provider has failed in this call; trying them again")
            chain.dead.clear()
        for p in chain.providers:
            if p.name in chain.dead:
                continue
            ok, e, pushed = await self._speak_with(p, output_emitter)
            if ok:
                return
            chain.stats["fallbacks"] += 1
            chain.strikes[p.name] = chain.strikes.get(p.name, 0) + 1
            permanent = getattr(e, "status_code", None) in PERMANENT_STATUS or chain.strikes[p.name] >= MAX_STRIKES
            if permanent:
                chain.dead.add(p.name)
                chain._remember_down(p.name, getattr(e, "status_code", None) in PERMANENT_STATUS, _describe(e))
            at = chain.providers.index(p)
            next_now = next((q.name for q in chain.providers[at + 1:] if q.name not in chain.dead), None)   # speaks THIS chunk
            remaining = [q.name for q in chain.providers if q.name not in chain.dead]                       # speaks LATER chunks
            if next_now is None:
                outcome = "no provider left for this chunk"
            elif permanent:
                outcome = (f"{p.name} is off for the rest of this call; speaking this chunk with {next_now}"
                           f"{f', later chunks start with {remaining[0]}' if remaining and remaining[0] != next_now else ''}")
            else:
                outcome = (f"speaking this chunk with {next_now}; {p.name} will be tried again "
                           f"({chain.strikes[p.name]} of {MAX_STRIKES})")
            logger.warning("TTS provider %s failed (%s); %s", p.name, _describe(e), outcome)
            if pushed:                                                      # the caller already heard part of it
                output_emitter.flush()
                return
        chain.stats["degraded"] += 1
        logger.error("TTS unavailable on every provider; skipping this speech chunk. The conversation continues without "
                     "audio for it. Degraded chunks so far: %d.", chain.stats["degraded"])
        output_emitter.push(bytes(int(chain.sample_rate * SILENCE_SECONDS) * 2))
        output_emitter.flush()


# ---- building it from .env ----------------------------------------------------------------------------------------------------
def provider_order(env: dict[str, str] | None = None) -> list[str]:
    env = dict(os.environ if env is None else env)
    raw = env.get("TTS_PROVIDER_ORDER") or DEFAULT_ORDER
    return [n.strip().lower() for n in raw.split(",") if n.strip()]


def build_providers(settings: config.Settings, env: dict[str, str] | None = None) -> list[Provider]:
    env = dict(os.environ if env is None else env)
    out: list[Provider] = []
    for name in provider_order(env):
        try:
            if name == "deepgram":
                key = _env_key(env, "DEEPGRAM_API_KEY")
                if not key:
                    logger.info("TTS provider deepgram skipped: DEEPGRAM_API_KEY is not set")
                    continue
                from livekit.plugins import deepgram
                inst = deepgram.TTS(model=env.get("DEEPGRAM_TTS_MODEL") or DEFAULT_DEEPGRAM_MODEL, api_key=key,
                                    encoding="linear16", sample_rate=CHAIN_SAMPLE_RATE, mip_opt_out=True)
            elif name == "local":
                path = Path(env.get("PIPER_MODEL") or DEFAULT_PIPER_MODEL)
                path = path if path.is_absolute() else config.PROJECT_ROOT / path
                if not path.exists() or not Path(str(path) + ".json").exists():
                    logger.info("TTS provider local skipped: voice file not found (set PIPER_MODEL; see README, Local TTS)")
                    continue
                inst = PiperTTS(path, length_scale=float(env.get("PIPER_LENGTH_SCALE") or DEFAULT_PIPER_LENGTH_SCALE),
                                volume=float(env.get("PIPER_VOLUME") or DEFAULT_PIPER_VOLUME))
            elif name == "cloudflare":
                key, account = _env_key(env, "CLOUDFLARE_API_TOKEN", "CLOUDFLARE_API_KEY"), _env_key(env, "CLOUDFLARE_ACCOUNT_ID")
                if not key or not account:
                    logger.info("TTS provider cloudflare skipped: CLOUDFLARE_API_TOKEN (or _API_KEY) and "
                                "CLOUDFLARE_ACCOUNT_ID are both needed")
                    continue
                inst = CloudflareTTS(account_id=account, token=key, model=env.get("CLOUDFLARE_TTS_MODEL") or DEFAULT_CLOUDFLARE_MODEL,
                                     speaker=cloudflare_speaker(env))
            elif name == "groq":
                if not _env_key(env, "GROQ_API_KEY"):
                    logger.info("TTS provider groq skipped: GROQ_API_KEY is not set")
                    continue
                from .tts_safety import SafeGroqTTS, get_limiter
                inst = SafeGroqTTS(model=settings.tts_model, voice=settings.tts_voice, limiter=get_limiter(settings.tts_max_rpm))
            else:
                logger.warning("TTS_PROVIDER_ORDER names an unknown provider %r (known: deepgram, local, cloudflare, groq); "
                               "skipped", name)
                continue
        except Exception as e:                                                # noqa: BLE001  (one broken provider must not stop the rest)
            logger.warning("TTS provider %s could not be created (%s); skipped", name, type(e).__name__)
            continue
        out.append(Provider(name, inst, hold=name != "local"))
    return out


def cloudflare_speaker(env: dict[str, str]) -> str:
    """CLOUDFLARE_TTS_SPEAKER, else the speaker named in the Deepgram model (aura-2-thalia-en -> thalia), so both give one voice."""
    explicit = (env.get("CLOUDFLARE_TTS_SPEAKER") or "").strip().lower()
    if explicit:
        return explicit
    m = re.search(r"aura-2-([a-z]+)-en", env.get("DEEPGRAM_TTS_MODEL") or DEFAULT_DEEPGRAM_MODEL)
    return m.group(1) if m else "thalia"


def build_chain(settings: config.Settings, call_id: str = "", env: dict[str, str] | None = None,
                budget: CharBudget | None = None) -> TtsChain:
    providers = build_providers(settings, env)
    if not providers:
        logger.error("no TTS provider is usable (check TTS_PROVIDER_ORDER and the keys in .env): the agent will be silent")
    else:
        logger.info("TTS chain: %s", " -> ".join(p.name for p in providers))
    settings_env = dict(os.environ if env is None else env)
    # the memory of failed providers is only used for the real agent (env is None); tests and tools pass their own env
    health = ProviderHealth() if env is None and health_enabled() else None
    chain = TtsChain(providers, budget or CharBudget(call_id), health=health)
    chain.hold = (settings_env.get("TTS_HOLD") or "1").strip() != "0"
    return chain
