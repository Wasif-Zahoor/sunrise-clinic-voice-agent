"""Run the REAL voice chain (as configured in .env) over typical clinic replies, with pauses like a conversation, and print what a
listener would experience: which provider spoke, how long until the voice started, and whether the audio ever ran dry.

Spends a little real credit (about 500 characters). Prints timings only (the phrases are fixed clinic sentences).

    python -m scripts.tts_chain_check            # about 1.5 minutes
    python -m scripts.tts_chain_check --quick    # no idle gaps
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import aiohttp  # noqa: E402
from livekit.plugins import deepgram  # noqa: E402,F401  (plugins must be imported on the main thread)

from clinic import config, tts_chain as tc  # noqa: E402
from clinic.tts_budget import CharBudget  # noqa: E402
from clinic.tts_safety import GroupedSentenceTokenizer  # noqa: E402

REPLIES = [
    "Thank you for calling Sunrise Family Clinic. This call may be recorded and transcribed for quality purposes. "
    "I'm the clinic's voice receptionist. How can I help you today?",
    "Is this your first visit with Sunrise Family Clinic?",
    "Let me verify your identity first. May I have your full name, please?",
    "Thank you. May I have your date of birth, please?",
    "I have these openings. Dr. Lee, Wednesday, December 2nd at 10:00 AM. Dr. Lee, Wednesday, December 2nd at 1:30 PM. Which would you like?",
    "Your balance due is forty five dollars. Is there anything else I can help you with?",
    "Sorry, I couldn't find that one. Could you say the medication name again, or spell it for me?",
    "Done. There are no refills left, so I've sent the request to your doctor for approval. Is there anything else I can help you with?",
]
GAPS = [0, 0, 12, 0, 25, 0, 8, 0]                  # seconds of "caller talking" before each reply


async def main(quick: bool) -> None:
    config.load_env()
    logging.basicConfig(level=logging.WARNING, format="  %(levelname)s %(message)s")
    logging.getLogger("clinic.tts").setLevel(logging.INFO)
    class OnlyTts(logging.Filter):
        def filter(self, r: logging.LogRecord) -> bool:
            return r.name == "clinic.tts" and not r.getMessage().startswith("TTS usage")
    for h in logging.getLogger().handlers:
        h.addFilter(OnlyTts())

    env = dict(os.environ)
    async with aiohttp.ClientSession() as session:
        providers = tc.build_providers(config.Settings("w", "m", "t", "v", 9), env)
        for p in providers:                                     # one shared HTTP session, because there is no LiveKit job here
            if hasattr(p.tts, "_session") and p.tts._session is None:
                p.tts._session = session
        chain = tc.TtsChain(providers, CharBudget("check", path=ROOT / "tts_samples" / "usage_check.json"))
        chain.hold = (env.get("TTS_HOLD") or "1").strip() != "0"
        chain.monitor.start()
        print(f"chain: {' -> '.join(p.name for p in providers)}   hold={'on' if chain.hold else 'off'}\n")
        tokenizer = GroupedSentenceTokenizer()
        for i, (text, gap) in enumerate(zip(REPLIES, GAPS), 1):
            if gap and not quick:
                await asyncio.sleep(gap)
            t0 = time.perf_counter()
            heard_after = None
            for chunk in tokenizer.tokenize(text):
                async with chain.synthesize(chunk) as stream:
                    async for _ev in stream:
                        heard_after = heard_after if heard_after is not None else time.perf_counter() - t0
            print(f"reply {i}: voice starts after {heard_after:.2f}s, whole reply ready after {time.perf_counter() - t0:.2f}s "
                  f"(served by {chain.active})\n")
        chain.monitor.stop()
        chain.log_summary()
        print(f"\ndead providers: {sorted(chain.dead) or 'none'}  fallbacks={chain.stats['fallbacks']}  degraded={chain.stats['degraded']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    asyncio.run(main(ap.parse_args().quick))
