"""A simulated 15-turn call through the REAL router and the real per-state prompt / tool / history path.

Measures prompt tokens per turn (from the provider's own usage report) and the busiest 60-second window at a normal
pace (one caller turn every PACE seconds). Live: needs real keys, takes about 3-4 minutes. Writes $TEMP/sim_15_turns.json.
Run with:  python -m pytest tests/test_live_sim_15_turns.py -q -s
"""

from __future__ import annotations

import asyncio
import json
import os
import time

import pytest

from clinic import gate, llm_router as lr
from clinic.turn import assess_turn

pytestmark = pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="needs real provider keys")
PACE = 12.0                       # seconds between caller turns (listening + the agent speaking + thinking)
LIMIT = 6000                      # target: prompt tokens in any 60 s window

TURNS = [
    "Hi, I'd like to check my balance.",              # LLM -> start_verification
    "Sarah Johnson", "yes", "March fourteenth nineteen eighty five", "three three four four",   # code (verification)
    "What do I owe?",                                  # LLM -> get_balance_due -> templated answer
    "What's the status of my lab results?",            # LLM -> templated
    "When is my next appointment?",                    # LLM -> templated
    "I need a refill on my Lisinopril.",               # LLM -> prepare_refill -> templated read-back
    "Yes, that's right.",                              # code (saves, scripted answer)
    "What are your office hours?",                     # LLM answers in text
    "Where are you located?",                          # LLM answers in text
    "Can I book an appointment with Dr. Lee?",         # LLM -> start_booking, then code asks the questions
    "sarah at gmail dot com",                          # code
    "Never mind, that's all for today.",               # note to the LLM -> short farewell
]


async def test_fifteen_turn_call_stays_under_the_token_budget(state):
    from livekit.agents import AgentSession
    import agent

    router = lr.build_router()
    sess = AgentSession(llm=router, userdata=state)
    ag = agent.Receptionist(state)
    await sess.start(ag)
    rows = []
    t_start = time.monotonic()
    for i, text in enumerate(TURNS, 1):
        turn_started = time.monotonic()
        before = len(router.usage_log)
        d = assess_turn(state, gate.scrub_transcript(state, text))
        source = "code"
        if not d.say:
            source = "LLM"
            if d.note:
                ctx = ag.chat_ctx.copy()
                ctx.add_message(role="system", content=d.note)
                await ag.update_chat_ctx(ctx)
            await sess.run(user_input=text)
        used = router.usage_log[before:]
        rows.append({"turn": i, "handled_by": source, "llm_requests": len(used),
                     "prompt_tokens": sum(u["prompt_tokens"] or 0 for u in used),
                     "total_tokens": sum(u["total_tokens"] or 0 for u in used),
                     "models": sorted({u["model"] for u in used}), "first_token_ms": [u["first_token_ms"] for u in used],
                     "at": round(turn_started - t_start, 1)})
        await asyncio.sleep(max(0.0, PACE - (time.monotonic() - turn_started)))
    await sess.aclose()

    # busiest 60 s window of prompt tokens, from the real request timestamps
    events = [(u["at"] - t_start, u["prompt_tokens"] or 0) for u in router.usage_log]
    busiest = max((sum(t for at, t in events if w <= at < w + 60) for w, _ in [(e[0] - 0.001, 0) for e in events]), default=0)
    out = {"pace_seconds": PACE, "turns": rows, "busiest_60s_prompt_tokens": busiest, "limit": LIMIT,
           "total_llm_requests": len(router.usage_log), "total_prompt_tokens": sum(r["prompt_tokens"] for r in rows)}
    json.dump(out, open(os.path.join(os.environ.get("TEMP", "."), "sim_15_turns.json"), "w"), indent=1)

    print(f"\n{'turn':>4} {'by':>5} {'reqs':>4} {'prompt tok':>10}  model(s)")
    for r in rows:
        print(f"{r['turn']:>4} {r['handled_by']:>5} {r['llm_requests']:>4} {r['prompt_tokens']:>10}  {', '.join(r['models'])}")
    print(f"busiest 60 s window: {busiest} prompt tokens (target < {LIMIT}); total {out['total_prompt_tokens']} in "
          f"{out['total_llm_requests']} LLM requests over {len(TURNS)} turns")
    assert busiest < LIMIT, f"{busiest} prompt tokens in one minute exceeds {LIMIT}"
