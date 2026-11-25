"""Sunrise Family Clinic voice receptionist.

Pipeline: browser mic -> LiveKit Cloud -> Groq Whisper STT -> [SSN/emergency gate in code]
          -> Groq LLM + backend tools -> Groq Orpheus TTS.

Run:  lk agent dev agent.py --log-level INFO   (connect from the LiveKit Cloud Agent Console)
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterable

from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    ChatContext,
    ChatMessage,
    JobContext,
    ModelSettings,
    SessionConnectOptions,
    StopResponse,
    TurnHandlingOptions,
    cli,
    function_tool,
    inference,
    room_io,
    stt,
    tts,
)
# Plugins register themselves on import and may only do so on the MAIN thread. On Windows each call's job runs in a worker thread,
# so every plugin used during a call must be imported here, when the worker starts (deepgram is otherwise imported lazily).
from livekit.plugins import deepgram, groq  # noqa: F401

from clinic import (booking_flow, callback_flow, config, db, gate, llm_router, prompts, silence, tools, transcript,
                    verify_flow)
from clinic.clock import Clock
from clinic.services import ServiceError
from clinic.logging_privacy import install_pii_redaction
from clinic.prompts import GREETING, INSTRUCTIONS, build_greeting, build_instructions, clean_for_speech
from clinic.state import CallState
from clinic.tts_chain import build_chain
from clinic.tts_safety import GroupedSentenceTokenizer
from clinic.turn import assess_turn

config.load_env()
logger = logging.getLogger("sunrise-agent")

# Voice is the only input: typed room text skips stt_node / on_user_turn_completed, i.e. it would bypass the
# SSN scrubber and the verification state machine, so it is switched off.
ROOM_OPTIONS = room_io.RoomOptions(text_input=False)

LLM_FALLBACK_SAY = ("Sorry, I'm having a technical issue. I can connect you with a staff member, "
                    "or take a message for a callback. Which would you prefer?")

__all__ = ["GREETING", "INSTRUCTIONS", "Receptionist", "build_session", "server"]


class Receptionist(Agent):
    def __init__(self, state: CallState | None = None, writer: transcript.TranscriptWriter | None = None) -> None:
        self._state = state or CallState()
        self._writer = writer
        super().__init__(instructions=build_instructions(self._state.clock))

    def _log_caller(self, text: str) -> None:
        """Store the caller's words in the transcript (sensitive answers are withheld). Never raises."""
        if self._writer is not None:
            try:
                self._writer.caller(text, transcript.step_tag(self._state, text))
            except Exception:                                   # noqa: BLE001  (transcripts must never break a call)
                pass

    # ---- gates that run in code before the LLM sees anything ----------------------------------
    async def stt_node(
        self, audio: AsyncIterable[rtc.AudioFrame], model_settings: ModelSettings
    ) -> AsyncIterable[stt.SpeechEvent]:
        """Strip spoken SSN digits from transcripts; they go to backend state only, never the LLM."""
        final = stt.SpeechEventType.FINAL_TRANSCRIPT
        async for ev in Agent.default.stt_node(self, audio, model_settings):
            if ev.alternatives and ev.type in (final, stt.SpeechEventType.INTERIM_TRANSCRIPT,
                                               stt.SpeechEventType.PREFLIGHT_TRANSCRIPT):
                for alt in ev.alternatives:
                    alt.text = gate.scrub_transcript(self._state, alt.text, capture=ev.type == final)
            yield ev

    async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage) -> None:
        text = new_message.text_content or ""
        self._log_caller(text)                  # BEFORE assess_turn: the transcript tag depends on which question was open
        decision = assess_turn(self._state, text)
        if decision.silent:     # the caller is probably mid-answer (a date said in pieces): wait, do not speak
            raise StopResponse()
        if decision.say:  # emergency / verification result: scripted reply, LLM skipped
            self.session.say(decision.say)
            raise StopResponse()
        if decision.note:
            turn_ctx.add_message(role="system", content=decision.note)

    async def llm_node(self, chat_ctx, tools_, model_settings):
        """Keep every LLM request small (free-tier token limits): this turn's system prompt is the CORE plus the one module
        that fits the call state, only the tools that module needs are shown, and only the last 6 caller turns are sent
        (a one-line CALL STATE stands in for older ones). If every provider fails, the caller hears a short apology and a
        staff/callback offer, and the answer is handled in code. The call never goes silent or crashes."""
        chat_ctx = prompts.compact_history(chat_ctx, prompts.instructions_for(self._state), keep_user_turns=6)
        allowed = prompts.tools_for(self._state)
        tools_ = [t for t in tools_ if getattr(t, "id", None) in allowed]
        try:
            async for chunk in Agent.default.llm_node(self, chat_ctx, tools_, model_settings):
                yield chunk
        except (APIConnectionError, APIStatusError) as e:
            logger.error("LLM unavailable on every provider (%s); offering staff or a callback", type(e).__name__)
            self._state.choice = "staff_or_callback"      # the caller's answer is handled by code, not by the LLM
            turn_audit = db.connect(self._state.db_path)
            try:
                from clinic.security import audit
                audit(turn_audit, self._state.call_id, "llm_unavailable", "all_providers_failed",
                      self._state.verified_patient_id)
            finally:
                turn_audit.close()
            yield LLM_FALLBACK_SAY

    def tts_node(
        self, text: AsyncIterable[str], model_settings: ModelSettings
    ) -> AsyncIterable[rtc.AudioFrame]:
        async def cleaned() -> AsyncIterable[str]:
            async for chunk in text:
                yield clean_for_speech(chunk)

        return Agent.default.tts_node(self, cleaned(), model_settings)

    def _answer(self, say: str | None, text: str) -> str:
        """Speak a templated answer written by code and skip the second LLM request; on a refusal hand the text to the LLM."""
        if say:
            self.session.say(say)
            raise StopResponse()
        return text

    # ---- tools (thin wrappers; every rule is enforced in clinic.services) ----------------------
    @function_tool
    async def check_clinic_hours(self) -> str:
        """Clinic open/closed status, hours, doctors, address."""
        return tools.check_clinic_hours(self._state)

    @function_tool
    async def start_verification(self, full_name: str = "") -> str:
        """Start identity verification. Pass full_name only if the caller already said it. The system then asks every
        verification question itself, one at a time."""
        try:
            r = verify_flow.start(self._state, full_name)
        except ServiceError as e:
            return f"NOT ALLOWED ({e.code}): {e.message}"
        if r.say:                          # the first question is spoken by code, not by the LLM
            self.session.say(r.say)
            raise StopResponse()           # and the LLM does not add a second, overlapping prompt
        if r.silent:                       # duplicate call: do nothing, say nothing
            raise StopResponse()
        return r.text

    @function_tool
    async def get_next_appointment(self) -> str:
        """Get the verified caller's next appointment."""
        return self._answer(*tools.speak_next_appointment(self._state))

    @function_tool
    async def prepare_refill(self, medication_name: str) -> str:
        """Find the medication the caller named, for read-back."""
        return self._answer(*tools.speak_refill_readback(self._state, medication_name))

    @function_tool
    async def confirm_refill(self) -> str:
        """Save the refill after the caller said yes."""
        return tools.confirm_refill(self._state)

    @function_tool
    async def get_lab_status(self) -> str:
        """Get the verified caller's lab status (status only)."""
        return self._answer(*tools.speak_lab_status(self._state))

    @function_tool
    async def get_balance_due(self) -> str:
        """Get the verified caller's balance due."""
        return self._answer(*tools.speak_balance(self._state))

    @function_tool
    async def start_booking(self, first_visit: str = "") -> str:
        """Start booking an appointment (new or existing patient). The system asks every booking question itself, one at
        a time. first_visit: "yes" or "no" only if the caller already said whether it is their first visit."""
        try:
            r = booking_flow.start(self._state, first_visit)
        except ServiceError as e:
            return f"NOT ALLOWED ({e.code}): {e.message}"
        if r.say or r.silent:
            if r.say:
                self.session.say(r.say)            # spoken by code; the LLM adds no second prompt
            raise StopResponse()
        return r.text

    @function_tool
    async def start_callback(self, caller_name: str = "") -> str:
        """Take a callback message (name, number, short reason). The system asks every question itself, one at a time.
        Pass caller_name only if the caller already said it."""
        try:
            r = callback_flow.start(self._state, caller_name)
        except ServiceError as e:
            return f"NOT ALLOWED ({e.code}): {e.message}"
        if r.say or r.silent:
            if r.say:
                self.session.say(r.say)            # spoken by code; the LLM adds no second prompt
            raise StopResponse()
        return r.text

    @function_tool
    async def request_staff_transfer(self, reason: str) -> str:
        """Ask for staff. reason: requested_human, upset, confused, verification_failed or cannot_complete."""
        return tools.request_staff_transfer(self._state, reason)

    @function_tool
    async def cancel_pending_request(self) -> str:
        """Discard the staged booking, refill or callback."""
        return tools.cancel_pending_request(self._state)

    @function_tool
    async def report_emergency(self) -> str:
        """Call for any emergency, chest pain, trouble breathing or suicidal thoughts."""
        return tools.report_emergency(self._state)


def build_llm(s: config.Settings):
    """The LLM for LLM_PROVIDER: 'router' (default, ordered fallback), 'groq' (one model) or 'gemini' (opt-in)."""
    if s.llm_provider == "router":
        return llm_router.build_router()
    if s.llm_provider == "gemini":
        from livekit.plugins import google
        import os
        return google.LLM(model=s.gemini_llm_model, temperature=0.4, api_key=os.getenv("GEMINI_API_KEY", ""))
    return groq.LLM(model=s.llm_model, temperature=0.4)


# Connection behaviour. TTS: at most ONE framework retry (each request spends provider credit; the TTS chain never raises: it
# switches provider or degrades to silence). STT/LLM: the framework closes the session
# after N consecutive unrecoverable errors; 3 is too eager for a free tier with brief outages, so allow more before ending.
CONN_OPTIONS = SessionConnectOptions(
    tts_conn_options=APIConnectOptions(max_retry=1, retry_interval=2.0, timeout=10.0),
    max_unrecoverable_errors=8,
)


def build_clients(s: config.Settings):
    """The STT and LLM clients. Creating them builds TLS contexts (100-400 ms of CPU), so the entrypoint runs this in a worker
    thread instead of blocking the event loop at call start."""
    return groq.STT(model=s.stt_model, language="en"), build_llm(s)


async def build_session_async(settings: config.Settings | None = None, state: CallState | None = None) -> AgentSession:
    s = settings or config.get_settings()
    return build_session(s, state, clients=await asyncio.to_thread(build_clients, s))


def build_session(settings: config.Settings | None = None, state: CallState | None = None, clients=None) -> AgentSession:
    """Create the STT -> LLM -> TTS session. Groq plugins read GROQ_API_KEY from the env."""
    s = settings or config.get_settings()
    kwargs = {"userdata": state} if state is not None else {}
    chain = build_chain(s, call_id=state.call_id if state is not None else "")
    stt_client, llm_client = clients or build_clients(s)
    session = AgentSession(
        stt=stt_client,
        llm=llm_client,
        # TTS_PROVIDER_ORDER chain (Deepgram, then local Piper by default); grouped sentences keep requests few. The chain
        # switches provider for the rest of the call on failure and only plays silence if every provider fails.
        tts=tts.StreamAdapter(tts=chain, sentence_tokenizer=GroupedSentenceTokenizer()),
        conn_options=CONN_OPTIONS,
        # Groq STT is request/response (not streaming), so it is paired with VAD.
        vad=inference.VAD(model="silero"),
        # Local VAD only: no cloud turn-detector or adaptive-interruption models.
        turn_handling=TurnHandlingOptions(
            turn_detection="vad",
            # 0.7 s instead of 0.5 s of silence before a turn ends: dictated dates, numbers and names have natural pauses
            endpointing={"min_delay": 0.7, "max_delay": 4.0},
            interruption={"mode": "vad"},
            # The LLM waits for the confirmed end of the caller's turn instead of guessing
            # early from partial transcripts. Costs a little latency; avoids discarded LLM
            # calls and keeps unconfirmed speech away from the LLM (matters for the SSN gate).
            preemptive_generation={"enabled": False},
        ),
        **kwargs,
    )
    session.tts_chain = chain            # so the entrypoint can tell whether the greeting really reached the caller
    return session


def attach_silence_watch(session: AgentSession, state: CallState) -> None:
    """When the caller goes quiet (user_away_timeout, 15 s) repeat the open question a few times, then stop. The task is
    cancelled the moment the caller speaks. Uses no LLM; each reprompt is one short TTS request."""
    task: dict[str, asyncio.Task | None] = {"t": None}

    def on_state(ev) -> None:
        if ev.new_state == "away":
            if task["t"] is None or task["t"].done():
                task["t"] = asyncio.create_task(silence.watch(session, state))
        elif task["t"] is not None and not task["t"].done():
            task["t"].cancel()

    session.on("user_state_changed", on_state)


async def deliver_greeting(session: AgentSession, state: CallState, writer: transcript.TranscriptWriter) -> bool:
    """Speak the greeting (with the recording/transcription notice) WITHOUT letting the caller cut it off, and start storing
    the caller's words only if it was really delivered. Returns whether it was."""
    safe = getattr(session, "tts_chain", None)
    degraded_before = safe.stats["degraded"] if safe is not None else 0
    handle = session.say(build_greeting(state.clock, transcripts=writer.enabled), allow_interruptions=False)
    await handle
    delivered = not getattr(handle, "interrupted", False) and (safe is None or safe.stats["degraded"] == degraded_before)
    writer.notice_given(delivered)
    if not delivered:
        logger.warning("greeting was not fully delivered; this call will not be transcribed")
    return delivered


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    install_pii_redaction()  # no caller speech / identifiers in logs, even at DEBUG
    db.init_db()             # creates and seeds the synthetic database on first run
    state = CallState(db_path=str(db.get_db_path()), clock=Clock.from_env())
    if state.clock.simulated:
        logger.info("simulated clinic time in use (CLINIC_NOW_OVERRIDE)", extra={"now": state.clock.now().isoformat()})
    session = await build_session_async(state=state)
    writer = transcript.TranscriptWriter(state, enabled=config.transcripts_enabled())
    writer.start()
    transcript.attach(session, writer)
    attach_silence_watch(session, state)

    chain = getattr(session, "tts_chain")
    chain.monitor.start()                # measures event-loop stalls, to tell "provider paused" from "this PC was busy"

    async def finish() -> None:          # runs on every way a job can end, including a crash or the caller just hanging up
        writer.close("session shutdown")
        chain.monitor.stop()
        chain.log_summary()              # one VOICE SUMMARY line: breaks, worst gap, loop lag
    ctx.add_shutdown_callback(finish)

    # record=False: LiveKit does not store call audio for this prototype (the text transcript is made by transcript.py).
    await session.start(room=ctx.room, agent=Receptionist(state, writer), record=False, room_options=ROOM_OPTIONS)
    await deliver_greeting(session, state, writer)
    getattr(session, "tts_chain").prewarm()     # load the local fallback voice AFTER the greeting (it competes for CPU)


if __name__ == "__main__":
    missing = config.missing_env()
    if missing:
        raise SystemExit(
            "Missing required environment variables: "
            + ", ".join(missing)
            + "\nCopy .env.example to .env and fill them in (see README.md)."
        )
    cli.run_app(server)
