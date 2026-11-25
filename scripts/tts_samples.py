"""Synthesize a few agent sentences through each real TTS provider and report the time to first audio.

Spends a little real credit (about 300 characters per paid provider). Saves WAV files to tts_samples/ (git-ignored) so you can
listen. Prints statuses, timings and byte counts only: no keys, no caller data (the sentences are fixed clinic phrases).

    python scripts/tts_samples.py            # deepgram, cloudflare (if configured), local, then the chain with a bad Deepgram key
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
import urllib.request
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import aiohttp                                   # noqa: E402

from clinic import config, tts_chain as tc       # noqa: E402
from clinic.tts_budget import CharBudget         # noqa: E402

SENTENCES = [
    "Thank you for calling Sunrise Family Clinic. This call may be recorded and transcribed for quality purposes. "
    "I'm the clinic's voice receptionist. How can I help you today?",
    "May I have your full name, please?",
    "Your next appointment is Thursday, December 3rd at 10:30 AM with Dr. Patel.",
]
OUT = ROOT / "tts_samples"


def save_wav(path: Path, frames) -> float:
    rate = frames[0].sample_rate
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(rate)
        for f in frames:
            w.writeframes(bytes(f.data))
    return sum(f.samples_per_channel for f in frames) / rate


async def timed(provider, text: str):
    """(seconds to first audio, seconds total, frames) for one request."""
    t0, first, frames = time.perf_counter(), None, []
    async with provider.synthesize(text, conn_options=tc.APIConnectOptions(max_retry=0, timeout=15)) as stream:
        async for ev in stream:
            first = first if first is not None else time.perf_counter() - t0
            frames.append(ev.frame)
    return first, time.perf_counter() - t0, frames


async def run_provider(name: str, provider) -> None:
    print(f"\n== {name} (model {provider.model})")
    for i, text in enumerate(SENTENCES, 1):
        try:
            first, total, frames = await timed(provider, text)
            seconds = save_wav(OUT / f"{name}_{i}.wav", frames)
            nbytes = sum(len(bytes(f.data)) for f in frames)
            print(f"  sentence {i} ({len(text):3d} chars): first audio {first:5.2f}s  total {total:5.2f}s  "
                  f"audio {seconds:4.1f}s  {nbytes} bytes  @ {frames[0].sample_rate} Hz")
        except Exception as e:                                       # noqa: BLE001
            print(f"  sentence {i}: FAILED {tc._describe(e)}")


def cloudflare_account(token: str) -> str:
    account = tc._env_key(dict(os.environ), "CLOUDFLARE_ACCOUNT_ID")
    if account:
        return account
    req = urllib.request.Request("https://api.cloudflare.com/client/v4/accounts",
                                 headers={"Authorization": f"Bearer {token}", "User-Agent": "clinic-samples/1.0"})
    return json.loads(urllib.request.urlopen(req, timeout=30).read())["result"][0]["id"]


async def main() -> None:
    config.load_env()
    logging.basicConfig(level=logging.WARNING, format="  log: %(message)s")
    OUT.mkdir(exist_ok=True)
    env = dict(os.environ)
    async with aiohttp.ClientSession() as session:
        dg_key = tc._env_key(env, "DEEPGRAM_API_KEY")
        if dg_key:
            from livekit.plugins import deepgram
            await run_provider("deepgram", deepgram.TTS(model=env.get("DEEPGRAM_TTS_MODEL") or tc.DEFAULT_DEEPGRAM_MODEL,
                                                        api_key=dg_key, sample_rate=tc.CHAIN_SAMPLE_RATE, mip_opt_out=True,
                                                        http_session=session))
        cf_key = tc._env_key(env, "CLOUDFLARE_API_TOKEN", "CLOUDFLARE_API_KEY")
        if cf_key:
            await run_provider("cloudflare", tc.CloudflareTTS(account_id=cloudflare_account(cf_key), token=cf_key,
                                                              http_session=session))
        piper_path = ROOT / (env.get("PIPER_MODEL") or tc.DEFAULT_PIPER_MODEL)
        if piper_path.exists():
            piper = tc.PiperTTS(piper_path)
            t0 = time.perf_counter()
            piper.render("Hello.")                                   # the one-time model load + first synthesis
            print(f"\n(local model load + warm-up: {time.perf_counter() - t0:.2f}s, done once per process)")
            await run_provider("local", piper)

        # The chain with a deliberately wrong Deepgram key: a real HTTP 401 from Deepgram, then the local voice takes over.
        if dg_key and piper_path.exists():
            from livekit.plugins import deepgram
            print("\n== chain: Deepgram with a WRONG key -> local (real 401, no credit spent)")
            bad = deepgram.TTS(model=tc.DEFAULT_DEEPGRAM_MODEL, api_key="not-a-real-key", sample_rate=tc.CHAIN_SAMPLE_RATE,
                               http_session=session)
            chain = tc.TtsChain([tc.Provider("deepgram", bad), tc.Provider("local", piper)],
                                CharBudget("samples", path=OUT / "usage_demo.json"))
            for i, text in enumerate(SENTENCES, 1):
                first, total, frames = await timed(chain, text)
                save_wav(OUT / f"chain_fallback_{i}.wav", frames)
                print(f"  sentence {i}: served by {chain.active}, first audio {first:5.2f}s  total {total:5.2f}s  "
                      f"dead={sorted(chain.dead)} degraded={chain.stats['degraded']}")
    print(f"\nWAV files: {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
