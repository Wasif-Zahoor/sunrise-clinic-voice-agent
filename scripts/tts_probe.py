"""Measure Deepgram text-to-speech delivery from THIS machine, without the agent: when does each piece of audio arrive, and does a
connection that has sat idle stall? Uses short synthetic phrases (about 700 characters of Deepgram credit in total). Prints timings
only: no key, no audio text.

    python -m scripts.tts_probe            # the default run: back to back, then after idle gaps on one connection
    python -m scripts.tts_probe --fresh    # also: a new connection for every request, to compare
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

import aiohttp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from clinic import config  # noqa: E402
from clinic.llm_router import _env_key  # noqa: E402
from clinic.tts_chain import DEFAULT_DEEPGRAM_MODEL  # noqa: E402

config.load_env()

URL = "https://api.deepgram.com/v1/speak"
BYTES_PER_SECOND = 24000 * 2                      # linear16, 24 kHz, mono
PHRASES = [
    "Thank you. May I have your date of birth, please?",
    "I have these openings. Wednesday at ten, or Wednesday at one thirty. Which would you like?",
    "Your balance due is forty five dollars. Is there anything else I can help you with?",
    "Sorry, I couldn't find that one. Could you say the medication name again?",
    "Let me confirm. Wednesday at one thirty with Doctor Lee. Is all of that correct?",
]


class Marks:
    """Connection events from aiohttp, to tell a reused connection from a new one and to time DNS / TCP / TLS."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.reused, self.created, self.t = False, False, {}

    def config(self) -> aiohttp.TraceConfig:
        tc = aiohttp.TraceConfig()

        async def reuse(s, c, p):
            self.reused = True

        async def create_start(s, c, p):
            self.created, self.t["conn0"] = True, time.perf_counter()

        async def create_end(s, c, p):
            self.t["conn1"] = time.perf_counter()

        tc.on_connection_reuseconn.append(reuse)
        tc.on_connection_create_start.append(create_start)
        tc.on_connection_create_end.append(create_end)
        return tc


async def one(session: aiohttp.ClientSession, marks: Marks, key: str, model: str, text: str, label: str) -> dict:
    marks.reset()
    params = {"model": model, "encoding": "linear16", "sample_rate": "24000", "mip_opt_out": "true"}
    t0 = time.perf_counter()
    out: dict = {"label": label, "chars": len(text)}
    try:
        async with session.post(URL, params=params, json={"text": text}, headers={"Authorization": f"Token {key}"},
                                timeout=aiohttp.ClientTimeout(total=60, sock_connect=10)) as res:
            out["status"], out["headers_at"] = res.status, time.perf_counter() - t0
            if res.status != 200:
                return out
            arrivals, total = [], 0
            async for data, _ in res.content.iter_chunks():
                now = time.perf_counter() - t0
                total += len(data)
                arrivals.append((now, total / BYTES_PER_SECOND))
    except Exception as e:                                          # noqa: BLE001
        out["error"] = type(e).__name__
        return out
    out["connection"] = "reused" if marks.reused else "new"
    if "conn1" in marks.t:
        out["connect_s"] = marks.t["conn1"] - marks.t["conn0"]
    if arrivals:
        out["first_audio"], out["done"], out["audio"] = arrivals[0][0], arrivals[-1][0], arrivals[-1][1]
        gaps = [(b[0] - a[0], a[1]) for a, b in zip(arrivals, arrivals[1:])]
        out["max_gap"], out["max_gap_at_audio_s"] = max(gaps) if gaps else (0.0, 0.0)
        out["chunks"] = len(arrivals)
        # silence the listener would hear if playback started at the first byte
        under, played_from = 0.0, arrivals[0][0]
        got_before = 0.0
        for at, got in arrivals:
            ahead = played_from + got_before                        # when the audio received so far runs out
            if at > ahead:
                under += at - ahead
                played_from += at - ahead
            got_before = got
        out["underrun"] = under
    return out


def show(r: dict) -> None:
    if "error" in r or r.get("status") != 200:
        print(f"{r['label']:<28} FAILED {r.get('error') or 'HTTP ' + str(r.get('status'))}")
        return
    print(f"{r['label']:<28} {r.get('connection', '?'):<6} headers={r['headers_at']:.2f}s first_audio={r['first_audio']:.2f}s "
          f"done={r['done']:.2f}s audio={r['audio']:.2f}s max_gap={r['max_gap']:.2f}s (at {r['max_gap_at_audio_s']:.1f}s of audio) "
          f"underrun={r['underrun']:.2f}s chunks={r['chunks']}" + (f" connect={r['connect_s']:.2f}s" if "connect_s" in r else ""))


async def main(fresh: bool) -> None:
    key = _env_key(dict(os.environ), "DEEPGRAM_API_KEY")
    if not key:
        raise SystemExit("DEEPGRAM_API_KEY is not set")
    model = os.environ.get("DEEPGRAM_TTS_MODEL") or DEFAULT_DEEPGRAM_MODEL
    marks = Marks()
    print(f"model={model}  (times in seconds; underrun = silence a listener would hear if playback began at first audio)\n")
    async with aiohttp.ClientSession(trace_configs=[marks.config()]) as session:
        print("-- one connection, back to back")
        for i, text in enumerate(PHRASES):
            show(await one(session, marks, key, model, text, f"back-to-back {i + 1}"))
        for gap in (10, 25):
            print(f"-- same connection after {gap} s idle")
            await asyncio.sleep(gap)
            show(await one(session, marks, key, model, PHRASES[1], f"after {gap}s idle"))
    if fresh:
        print("-- a new connection every time")
        for i, text in enumerate(PHRASES[:3]):
            async with aiohttp.ClientSession(trace_configs=[marks.config()]) as s:
                show(await one(s, marks, key, model, text, f"fresh {i + 1}"))


async def keepwarm(rounds: int, idle: int, every: int) -> None:
    """Does a connection kept warm with free pings (GET /v1/projects: no speech, no credit) avoid the stalls of a cold one?"""
    key = _env_key(dict(os.environ), "DEEPGRAM_API_KEY")
    if not key:
        raise SystemExit("DEEPGRAM_API_KEY is not set")
    model = os.environ.get("DEEPGRAM_TTS_MODEL") or DEFAULT_DEEPGRAM_MODEL
    marks = Marks()
    print(f"warm connection: {idle}s of pings every {every}s before each request; {rounds} rounds\n")
    async with aiohttp.ClientSession(trace_configs=[marks.config()]) as session:
        for i in range(rounds):
            waited, ping_ms = 0, []
            while waited < idle:
                t = time.perf_counter()
                try:
                    async with session.get("https://api.deepgram.com/v1/projects", headers={"Authorization": f"Token {key}"},
                                           timeout=aiohttp.ClientTimeout(total=20)) as r:
                        await r.read()
                        ping_ms.append(int((time.perf_counter() - t) * 1000))
                except Exception as e:                                  # noqa: BLE001
                    ping_ms.append(type(e).__name__)
                await asyncio.sleep(every)
                waited += every
            print(f"  pings (ms): {ping_ms}")
            show(await one(session, marks, key, model, PHRASES[i % len(PHRASES)], f"warm round {i + 1}"))


async def cloudflare(speaker: str | None) -> None:
    """Same measurements for Cloudflare Workers AI (Deepgram Aura-2 served from Cloudflare's network). The body arrives whole."""
    token = _env_key(dict(os.environ), "CLOUDFLARE_API_TOKEN", "CLOUDFLARE_API_KEY")
    account = _env_key(dict(os.environ), "CLOUDFLARE_ACCOUNT_ID")
    if not (token and account):
        raise SystemExit("CLOUDFLARE_ACCOUNT_ID and the Cloudflare token must be set")
    url = f"https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/@cf/deepgram/aura-2-en"
    print(f"cloudflare aura-2-en speaker={speaker or '(default)'}\n")
    async with aiohttp.ClientSession() as session:
        for i, text in enumerate(PHRASES):
            body = {"text": text, "encoding": "linear16", "container": "none", "sample_rate": 24000}
            if speaker:
                body["speaker"] = speaker
            t0 = time.perf_counter()
            try:
                async with session.post(url, json=body, headers={"Authorization": f"Bearer {token}"},
                                        timeout=aiohttp.ClientTimeout(total=60)) as res:
                    headers_at = time.perf_counter() - t0
                    data = await res.read()
                    print(f"cloudflare {i + 1}: HTTP {res.status} headers={headers_at:.2f}s whole_body={time.perf_counter() - t0:.2f}s "
                          f"audio={len(data) / BYTES_PER_SECOND:.2f}s" + ("" if res.status == 200 else f" body={data[:120]!r}"))
            except Exception as e:                                  # noqa: BLE001
                print(f"cloudflare {i + 1}: FAILED {type(e).__name__}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--cloudflare", action="store_true", help="measure Cloudflare Workers AI instead of Deepgram directly")
    ap.add_argument("--speaker", default=None, help="Cloudflare aura-2 speaker, e.g. thalia")
    ap.add_argument("--keepwarm", action="store_true", help="keep the connection warm with free pings between requests")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--idle", type=int, default=20)
    ap.add_argument("--every", type=int, default=5)
    a = ap.parse_args()
    if a.keepwarm:
        asyncio.run(keepwarm(a.rounds, a.idle, a.every))
    else:
        asyncio.run(cloudflare(a.speaker) if a.cloudflare else main(a.fresh))
