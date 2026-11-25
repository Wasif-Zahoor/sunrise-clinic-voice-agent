# Progress

## Round 12: fifth live call, after hours (2026-11-25, 15:11)

An after-hours call (clock 20:30) saved a callback message correctly. Then "Can you tell me my last status?" (speech-to-text for "lab status")
reached the LLM, which said "I can't access that information. Would you like me to have a staff member call you back?": vague, and a second
offer right after the message had been saved. The call log also showed the greeting taking about 33 s: Deepgram timed out (a 5 s wait),
Cloudflare answered HTTP 429 (its free daily allowance was used up by earlier tests), the next chunk waited 9 s for Deepgram's deadline
before the local voice spoke, and the log line said "using deepgram for the rest of this call" although the chunk went to the local voice.
(The first phone-number answer was also not understood; the parser handles all ordinary dictations I tried, so that was speech-to-text
returning no digits. Not changed.)

**Changed (offline-tested):**
* `intent.closed_say` / `intent.route`: after hours, a clear patient request (labs, balance, refill, my appointment, scheduling) gets one fixed
  answer in code: closed, the hours, and an offer to take a message (a "yes" starts it), or "Your message is recorded..." if one was already
  saved. Nothing is looked up, even for a verified caller. Emergency and third-party turns still come first. The lab pattern also matches
  "last status" / "lap results".
* `clinic/tts_health.py` (new) + `tts_chain`: a provider the chain gave up on is skipped by later calls for 10 min (two timeouts / errors) or
  60 min (HTTP 401 / 402 / 403 / 429), then tried again; never leaves the chain empty; names, reason and time only; `TTS_HEALTH=off` disables it.
* The failover log now says who speaks THIS chunk and who starts the next ones.

**Why Cloudflare was dead:** its free allowance for this voice is small (about 3,600 characters a day if the free 10,000 Neurons / day
holds: unconfirmed on Cloudflare's own page) and resets at 00:00 UTC; earlier test calls used it up. Deepgram's network path from the
developer's PC is the other half of the problem (see Round 10). For a recording, `TTS_PROVIDER_ORDER=local` is the only option that cannot stall.

**Results (actual).** Offline suite **706 passed**, 46 live deselected (was 677; +29 in `tests/test_round12.py`, 3 existing tests updated on
purpose). **Not tested:** a live call with these fixes.

## Round 11: fourth live call, new-patient flow (2026-11-25, 14:27)

A new patient (Wasif) booked successfully, then asked "what is my lab status": the agent restarted verification (name, date of birth, last
four digits). The caller said "I already gave you that"; the agent asked again. Root cause: verification only matches existing patients
(`is_new = 0`) and a new patient has no stored last four digits, so it could never succeed. Other things the transcript showed:
"Yes, book 1 for 10am" was booked as **1:30 PM** (`10am` has no word boundary before `am`, so the time was read as hour 1, which matched
1:30 PM); "Sophie" (speech-to-text for "self-pay") was accepted as the insurance provider and a member ID was asked for. The voice did not
break this time (every chunk logged `held=yes`, `underrun=0.00s`); Deepgram timed out twice again, Cloudflare took over, and one Cloudflare
chunk timed out and was spoken by the local voice.

**Changed (offline-tested):**
* `services.get_new_patient_info` + `tools.speak_new_patient` + `intent.route`: a caller who registered as a new patient in this call is
  answered about labs / balance / appointment / refill from the record created in this call (nothing on file yet; the appointment they just
  booked), audited, with the after-hours / emergency / third-party rules, and **not** marked verified (verified-only tools still refuse
  them). `CallState.new_patient_id` is set by `confirm_booking`. A caller who then verifies as an existing patient gets the normal answers.
* `speech.choose_slot`: "10am" is split into "10 am"; the clock time before am/pm is used rather than the first number in the sentence;
  "book 1" / "take two" is the Nth option only when no clock time is said.
* `booking_flow` (new step `provider_confirm`) + `speech.is_known_insurer`: an unrecognised insurer is read back and kept only after a yes
  ("no" asks again, "self-pay" skips insurance, a different name is read back in turn); `transcript.py` masks the heard name and tags the answer.

**Known and not changed:** a second booking by the same new patient in one call goes to staff (the duplicate check blocks it); on Cloudflare
a long reply can pause between chunks (the second chunk downloads while the first plays and is slower than the first's audio); the
spelled-name bug ("S-E-R-E-H ... G-O-H ... SON" read as "Son") is still open; the name-tolerance decision (accept a name within one letter
when date of birth and last four match) is still open and would relax a CLAUDE.md rule.

**Results (actual).** Offline suite **677 passed**, 46 live deselected (was 630; +47 in `tests/test_round11.py`). **Not tested:** a live
call with these fixes.

## Round 10: third live call, the voice kept breaking (2026-11-25, 13:39)

Booking and the balance worked; the refill only got saved because the LLM took over after the code flow had given up (see below); the voice broke up. The new per-chunk timing lines in the
call log showed it: `first_audio=2.67s synthesis=16.99s audio=3.41s` and `first_audio=0.96s synthesis=24.66s audio=10.73s`: Deepgram sent the
first piece and then paused for 15-24 s; several other chunks arrived at only about real-time speed, so any jitter was a gap.

**Cause (measured, not guessed).** `scripts/tts_probe.py` talks to Deepgram directly from the developer's PC: first request timed out after 60 s;
a 6.9 s clip took 30.7 s (22 s of silence); a connection took 8.5 s to open; connect time to `api.deepgram.com` 0.84 s median against
0.08-0.13 s for Groq, Cloudflare and LiveKit; tiny `GET` pings took 1-10 s and some ended in `ServerDisconnectedError`. The idle-connection
theory was tested (`--keepwarm`) and **rejected**. So: the network path from that PC to Deepgram, at that time. Not this program.
Cloudflare's copy of the same voice delivered 10 of 10 clips in 2.3-4 s.

**Changed (offline-tested, plus one real run):**
* `clinic/tts_chain.py`: **whole-clip hold** for network providers: the clip is kept until complete, then played at once; a clip that misses its
  deadline (3 s + 0.6 s per estimated second of speech) is dropped before anything plays and the next provider speaks it (a timeout is not
  retried; two in a row give the provider up for the call). Local Piper is not held. `TTS_HOLD=0` switches it off. Default order is now
  `deepgram,cloudflare,local`; Cloudflare requests `speaker=thalia` (follows `DEEPGRAM_TTS_MODEL`) so the switch is not heard.
* `clinic/tts_safety.py`: the first chunk of a reply is at most 90 characters (merge cap), so the wait before the voice starts stays short.
* `clinic/voice_diag.py` (new): per-chunk arrival trace (frames, longest gap and where, silence the caller would hear, gap hidden by the hold,
  idle time before the chunk, event-loop lag during it), `VOICE BREAK` warning with the likely culprit, `VOICE loop lag` warning, and a
  `VOICE SUMMARY` line at the end of every call. No spoken text is ever logged.
* `clinic/refill_flow.py`: "Thank you.", "Yeah.", "Okay" and the request said again ("I would like to get a refill") no longer count as
  failed tries. In the call they used up all three tries before "Metformin", so it was the LLM that saved the request, by luck.
* `scripts/tts_probe.py`, `scripts/tts_chain_check.py` (new): measurements from this PC, no agent.
* `.env`: `CLOUDFLARE_ACCOUNT_ID` added (the only change; gitignored). Cloudflare LLM and TTS are no longer skipped.

**Results (actual).** Offline suite **630 passed**, 46 live deselected (was 604; +26 in `tests/test_round10.py`; 3 existing tests updated on purpose:
the default order, and "greeting is one request" is now "two requests, the first short"). `python -m scripts.tts_chain_check` (real providers,
8 replies, 12 chunks, idle gaps of 0-25 s): **0 audible breaks**, 1.4 s of gaps hidden by the hold; the voice started 1.5-3.8 s after each reply was ready, and once after 9.8 s
(the reply that waited out Deepgram's deadline before Cloudflare took over). Deepgram stalled twice in a row after a 25 s pause and was given up for
that call; Cloudflare took over with the same voice.
**Not tested:** a live call with these changes; by-ear similarity of Cloudflare's voice to Deepgram's; Cloudflare's daily limit
(about 3,600 characters if the free 10,000 Neurons/day holds: unconfirmed on Cloudflare's own page).

## Round 9: second live call (2026-11-25, 12:28)

A live call (Deepgram, one chunk on local) completed an appointment booking but failed the refill: the caller's drug name arrived as "Lisey
Nuperell" and then spelled "L-I-S-I-L" / "N-O-P-R-I-L", the LLM never called `prepare_refill` (no audit row) and said "I can't help with that";
"Thank you." was accepted as the name "Thank You"; "schedule an appointment" took three LLM round trips before anything happened ("Hello?");
one 34-character chunk fell back to the local voice because of `APIConnectionError caused by ServerDisconnectedError`. The caller also
reported **the voice breaking up overall; its cause is not established** (see below).

**Fixed (offline-tested):**
* `clinic/refill_flow.py` (new): code-owned refill conversation; fuzzy matching (similarity at least 0.55 and 0.15 ahead of the runner-up, so
  `Lisey Nuperell` 0.61 and the spelled version 0.95 match Lisinopril, `I'd like to leave food` 0.31 does not) among the verified patient's
  OWN medications only; read-back; the saved request still needs a clear yes; three misses offer staff / callback; wait, cancel, staff
  and emergency leave it cleanly.
* `clinic/intent.py`: now also starts booking in code and, for a verified caller, answers lab / balance / appointment lookups and starts
  the refill flow immediately; still not used when closed, mid-flow, or when the turn was flagged (injection / third party).
* Names: acknowledgements (thank you, hello, okay, please, goodbye...) are not names.
* TTS: a connection error is retried once on the same provider (the stale-connection case) before any fallback; every chunk logs
  provider, characters, time to first audio, synthesis time and audio length (no text); local Piper streams sentence by sentence and is
  warmed after the greeting, not during it.

**Voice breaking: what is and is not known.** The logs show no TTS failure except the one stale connection (now retried) and a few 105-152 ms
event-loop stalls inside the framework (openai request transform, asyncio internals). They cannot show choppy playback. Candidate causes, none
confirmed: that stale-connection switch to a different voice for one sentence; CPU contention on this PC (the Piper warm-up used to run during the
greeting; speech detection and the LLM client share one process on Windows); the agent hearing itself through speakers (use headphones);
network jitter to LiveKit Cloud. The new `TTS chunk` log lines will show whether Deepgram falls behind playback. **Needs a live call to settle.**

**Results (actual):** offline suite **604 passed**, 46 live deselected (was 554; +46 `test_round9.py` incl. a replay of the failed refill, +4 updated
for the new routing). Real audio re-run (`scripts/tts_samples.py`): Deepgram first audio 0.36-1.64 s, local 0.27-0.65 s (greeting 0.36 s, was 1.07 s).
**Not tested:** a live call with these changes.

## Round 8: call-flow fixes from a live test call (2026-11-25)

A live call (Deepgram, then local) showed: (1) "I would like to get a prescription refill" got "I can't handle prescription requests" from the
LLM; (2) a date of birth said as "My date of birth is" / "March." / "14, 1985." arrived as three speech turns, each counted as an unclear
answer, so the flow gave up and offered staff; (3) the callback asked for a name already given; (4) "I don't know" at the read-back was taken
as "no"; (5) a single Deepgram `APIConnectionError` moved the whole call to the local voice.

**Fixed (all in code, offline-tested; the cause of (5)'s error is not known):**
* `clinic/intent.py`: a clear refill / lab / balance / "my appointment" request from an unverified caller starts verification in code (not when the
  turn is flagged as an injection attempt or a third-party request, when closed, locked, or mid-flow). The request is remembered and continued
  right after verification: refill asks "Which medication...", the others are answered from the templates.
* Fragments: a date said in pieces is joined; up to two pieces are held silently (no reply, no unclear count); the same for booking.
  Last-four digits may come in up to three pieces, a lone spoken digit is captured while the digits are awaited, half-heard digits and pieces
  are dropped on a silence reprompt. Turn endpointing 0.5 s -> 0.7 s (maximum 4 s) so dictated answers are split less often.
* "I don't know / not sure / maybe" is neither yes nor no; "Nothing / no changes / it's fine" at the callback change step reads the message back again;
  a callback after a failed verification reuses the name the caller already confirmed.
* TTS: transient failures no longer switch the voice for the rest of the call (see README); the log names the underlying error type.
* Fallback voice: female `hfc_female`, chosen by measurement, pace and loudness tuned to Deepgram (README, "Making the fallback sound like Deepgram").

**Results (actual):** offline suite **554 passed**, 46 live deselected (was 506; +38 `test_call_flow_fixes.py` incl. a replay of the failed call, +10
TTS chain / voice tests; 6 existing tests updated for the new intended behavior). Real audio from every provider re-run with the new voice
(`scripts/tts_samples.py`): Deepgram first audio 0.36-1.07 s, local 0.23-1.07 s. **Not tested:** a live voice call with these changes; whether
0.7 s endpointing is enough for your speaking pace; how close the fallback voice sounds by ear.

## Round 7: TTS provider change (2026-11-25)

**Problem:** Groq's free Orpheus allowance (about 3,600 characters a day) ran out, and with a new Groq account the first request failed with
HTTP 400 `model_terms_required` (terms not yet accepted). The caller heard nothing either way.

**Done (approved by the developer):** primary TTS is now Deepgram Aura-2 (`aura-2-thalia-en`) via the official LiveKit plugin, with an
automatic chain `TTS_PROVIDER_ORDER=deepgram,local` (optional `cloudflare`, `groq`). Local fallback: Piper `en_US-lessac-medium` (replaced in round 8 by the female `en_US-hfc_female-medium`)
(63 MB, `models/`, git-ignored). On any provider failure the chain moves to the next one for the rest of the call; if all fail the call survives with
silence. Character counter per call / day / total with a configurable warning (`data/tts_usage.json`, `python -m clinic.tts_budget`).
Cloudflare added as the last LLM fallback (verified: 200 OK, tool call made, first token 2.2-2.7 s). Heavy client construction moved off the
event loop (cold-start stall about 300 ms -> about 110 ms; the rest is inside the framework). Details: README "Text-to-speech".

**Results (actual):**

| Check | Result |
|---|---|
| Offline suite `pytest -k "not live"` | **505 passed**, 46 live deselected (was 466; +36 `test_tts_chain.py`, +2 router, +1 resilience; 3 existing tests made independent of `.env`) |
| Provider order from `.env`; keyless / missing-file provider skipped | passed (`test_tts_chain.py`) |
| Fallback after mocked 402 / 429 / 401 / 500 / 503 / empty / timeout | passed; failed provider not asked again in the call |
| Mocked 24-turn call, Deepgram failing on turn 10 (direct, and through `StreamAdapter` + grouping) | passed: every turn has audio, turns 10-24 served by local, 0 skipped |
| Character counter and warning | passed (per call / day / total, persists, warns once, free provider never warned, unwritable file tolerated) |
| Real audio, `scripts/tts_samples.py` | Deepgram, Cloudflare, local Piper all returned real PCM (saved to `tts_samples/`); first audio 0.4-1.1 s / 1.5-6.5 s / 0.2-1.0 s |
| Real Deepgram HTTP 401 (wrong key) -> local | worked: first chunk 1.9 s, later 0.3-0.5 s |
| No keys / spoken text in logs; `.env` ignored | covered by tests (`test_tts_chain.py`, `test_log_privacy.py`); `.env`, `models/`, `tts_samples/`, `data/tts_usage.json` are in `.gitignore` |
| Worker startup | registers with LiveKit Cloud |

**Not tested (needs your live voice call):** how the Deepgram and local voices sound in a real call, end-to-end latency, the silence
reprompts, and everything listed in Round 6. Deepgram credit remaining is unknown to the code (see its dashboard).

## Requirements audit (2026-11-24)

Source: the assignment as summarised in `CLAUDE.md` sections 4-9 and 14 (`Task_requirements.docx` is the original).
Legend: **V** = implemented and verified by automated tests (offline, fake STT / LLM / TTS, so *no audio was involved*);
**V-live** = also seen working in a real voice call by the developer; **N** = implemented, **not verified** in a live voice call
since the latest changes; **M** = missing; **B** = known broken. Nothing is marked "voice verified" without a human voice test.

| # | Requirement | Status | Evidence / note |
|---|---|---|---|
| 1 | Natural real-time voice | V-live (Phase 1), N for later changes | Phase 1 browser test passed. Scripted flows, router and TTS grouping were never live-tested |
| 2 | Announce the call may be recorded for quality purposes | V (text) / N (audio) | Greeting now says "may be recorded and transcribed for quality purposes", uninterruptible (`test_resilience`, `test_phase1`). It previously said "not being recorded", which did not satisfy the assignment |
| 3 | Check clinic hours (Eastern, DST) | V | `test_phase2` (DST, 5:00 PM edge, weekends); configurable clock `CLINIC_NOW_OVERRIDE` |
| 4 | Verify existing patients: name + DOB + last 4 on the same record, in backend code | V | `test_verification_flow`, `test_phase2`: wrong field combos, mixed records, generic failure, no field disclosed |
| 5 | Max 3 failed attempts, then staff offer | V | lock persists for the call; unclear answers are counted separately |
| 6 | No PHI before verification; enforced at the tool level | V | `services.operation()`; every tool refused when unverified (`test_phase2`) |
| 7 | Never send SSN to the LLM; never store / repeat the full SSN | V | digits scrubbed in `stt_node`; only a keyed hash of the last 4; **the STT vendor still hears the audio** (documented) |
| 8 | Sequential, one-question-at-a-time verification, natural dates | V | `verify_flow`, `dates`; ambiguous / invalid / incomplete dates covered |
| 9 | Prescription refill (confirm medication + pharmacy, doctor approval when none left, never "approved") | V | `test_templated_answers`, `test_phase2` |
| 10 | Existing appointment lookup; none -> offer booking or callback | V | **new this round:** the offer is made and handled in code (`book_or_callback`), no longer left to the LLM |
| 11 | Lab status only | V | status template; interpretation refused (prompt + backend returns status only) |
| 12 | Billing balance only | V | template; locked when unverified / closed / third party |
| 13 | Booking, new and existing patients; real slots; 2-3 options; confirm; read back; spell the email; save; remove slot; simulated email; arrival instructions | V | `test_booking_flow`; no double booking; atomic transaction |
| 14 | Save requests, bookings, callbacks | V | tables `requests`, `appointments`, `callback_messages`, `email_outbox`; duplicate-save guards |
| 15 | After-hours flow: closed message, hours, no PHI even if verified, collect name / number / reason, read back, save, next business day, 911 | V | `test_callback_flow`, `test_phase2`; enforced in `services.operation()` |
| 16 | Emergency detection -> 911 immediately, workflow stops | V | regex gate before the LLM, 11+ phrasings, repeated message afterwards |
| 17 | No medical advice | V (gate/prompt) / N (LLM wording) | depends partly on the model following the prompt; live LLM scenario tests exist (`-k live`), results in the round-5 table |
| 18 | Third-party access restricted | V | `third_party_turn` guard refuses patient tools in code; live: gpt-oss-20b once tried a tool, fixed |
| 19 | Prompt-injection resistance | V (structure) / N (wording) | no tool accepts ids or flags; refusals partly prompt-driven |
| 20 | Human handoff (requested, upset, confused, cannot verify) | V (simulated) | writes a `handoff` request; a real transfer does not exist (stated) |
| 21 | Short, natural spoken replies | V (templates, <=190-char TTS chunks) / N (LLM replies) | |
| 22 | Mask sensitive data in logs and transcripts | V | log redaction (`test_log_privacy`) and, **new**, transcript redaction (`test_transcript`) |
| 23 | Audit log without PHI | V | `audit_logs` holds actions / outcomes only |
| 24 | Do not expose the whole DB / prompt / credentials | V | tools return one fact each; no secret appears in any project file (`test_log_privacy`) |
| 25 | Error handling / recovery | V | LLM outage -> apology + staff-or-callback handled in code; TTS failure -> silence for that chunk; STT/LLM unrecoverable-error limit raised 3 -> 8; TTS framework retry 3 -> 1 |
| 26 | Silence / unclear answers / interruption | V (code) / N (live) | **new:** `silence.py` reprompts (max 3 then quiet). Whether LiveKit's `user_away` event fires after 15 s in a real call is not observed |
| 27 | **Conversation transcripts** (added request) | V (offline) / N (live) | **new:** `clinic/transcript.py`; see below |
| 28 | Synthetic data only | V | `data/seed.json`; no real data anywhere |
| 29 | HIPAA documentation, no compliance claim, BAA facts | V (docs) | README "HIPAA principles and limits"; Groq BAA excludes free / preview services; nothing is claimed |
| 30 | README, `.env.example`, `.gitignore`, tests | V | updated this round |
| 31 | Twilio (optional) | **M** | not started, deliberately (secondary milestone) |
| 32 | GitHub repository | **M** (user action) | the folder is not a git repository; a dry run in a temp copy showed `.env`, `*.db` and `transcripts/` are ignored. No `git init`, commit or push was done without approval |
| 33 | 5-7 minute demo video | **M** (user action) | scenarios and commands are in README "Demo scenarios" |
| - | Known **B**roken | none known | open weaknesses: `qwen` (3rd fallback) often refuses to call `start_verification`; SambaNova needs a payment method; token target missed at 12 s per turn; no live voice test of recent changes |

### Round 6: final completion audit (2026-11-24)

**Fixed / added (all in code, all tested offline):**

1. **Greeting** now announces "may be recorded and transcribed for quality purposes" (assignment requirement; the earlier "not being
   recorded" wording did not meet it) and is **uninterruptible**; transcripts are enabled for the caller's words only after it was
   delivered (not interrupted and no TTS chunk degraded). `TRANSCRIPTS=off` falls back to the truthful "not being recorded".
2. **Transcripts** (`clinic/transcript.py`, `python -m clinic.transcript list|show`): one `.txt` per call in `transcripts/`, call id,
   start and end times, timestamped `CALLER` / `AGENT` lines in order, interruption markers, error-type events, flushed per line,
   closed by the `close` event **and** a job shutdown callback (idempotent), never raises into the call, configurable
   (`TRANSCRIPTS`, `TRANSCRIPT_DIR`), `transcripts/` git-ignored. Redaction is two-layer (step-based "not stored" + pattern/known-value);
   limits documented in README "Transcripts".
3. **No appointment on file** -> "book one, or leave a callback message?" is now spoken and answered by code (`book_or_callback`), not by the LLM.
4. **Silence**: reprompt the open question up to three times, then offer staff / callback and stay quiet (`clinic/silence.py`).
5. **TTS / connection options** (investigation of the Orpheus failures, no live calls spent):
   * Every Groq TTS *request* is already one grouped chunk (<=190 chars), paced at 9/min, with `Retry-After` honoured once and a
     daily-limit cooldown. The binding limit is **3,600 characters per day** (not requests), so reducing requests does not extend it;
     only speaking fewer characters does. Per-call cost is in README "Groq TTS limits".
   * The only remaining multiplier was the framework's own retry (default `max_retry=3`), which could re-send a failed request; it is now
     **1**. `SafeGroqTTS` already turns a failed chunk into silence instead of raising, so no unrecoverable TTS error can close the session.
   * STT / LLM unrecoverable-error tolerance raised from 3 to 8 so a brief outage no longer ends the call.
   * The silence reprompts are capped at three short sentences, so they cannot drain the daily characters.
   * No provider, plan or billing change was made.
6. `.env.example`: added `TRANSCRIPTS`, `TRANSCRIPT_DIR`; corrected the stale LLM provider text.

**Test results (actual, this machine, 2026-11-24):**

| Suite | Result |
|---|---|
| Offline: `python -m pytest -q -k "not live"` | **466 passed**, 46 live-LLM tests deselected, none run (was 395; +71 new: 42 in `test_transcript.py`, 29 in `test_resilience.py`; 2 existing tests updated for the new greeting / no-appointment behaviour). Note: `-k "not live"` matches test *names*, so a test named `...delivered...` was silently skipped until renamed |
| Worker startup: `lk agent dev agent.py --log-level INFO` (project venv) | registers with LiveKit Cloud, no errors, no secrets printed |
| SQLite: `PRAGMA integrity_check` / `foreign_key_check` on a fresh DB and on `data/clinic.db` | `ok` / no violations |
| Git dry run in a temp copy | `.env`, `data/clinic.db`, `transcripts/` ignored; `.env.example` included |
| Offline sample transcripts through the real `Receptionist.on_user_turn_completed` hook (verified refill/balance, new-patient booking, after-hours callback, emergency + third party) | credentials, names, DOB, phone, email and reasons absent from the files; 911 kept |
| Live LLM / voice | **not run** in this round (to protect the Groq quota); last live results are in the round-5 tables |

**Scenario status for the demo (what is verified vs. what still needs a human voice test):**

| # | Scenario | Offline code-path test | Live voice |
|---|---|---|---|
| 1 | Existing patient verified | yes | not re-tested since round 3 |
| 2 | Appointment inquiry / booking | yes | no |
| 3 | Refill | yes | no |
| 4 | Failed verification | yes | yes in Phase 2 voice test (round 1), not since |
| 5 | New patient booking | yes | round 3-4 voice tests found email-dictation problems; fixes untested live |
| 6 | After-hours callback | yes | no |
| 7 | Medical advice | prompt + live text test | no |
| 8 | Emergency | yes (gate) | no |
| 9 | Third party | yes (guard) | no |
| 10 | Review transcripts | yes (generation, redaction, save on abnormal end) | no |

## Phase 1 - Voice foundation: DONE (voice tested by the developer)

- Browser voice test passed: Whisper STT, Groq LLM, interruptions and latency good.
- Fix round completed: TTS 429 root cause found (Groq free-tier limits, see below), greeting says the call is not
  recorded, preemptive generation disabled, interruption behaviour kept, log redaction added.
- Developer reported being satisfied with the voice setup before Phase 2.

## Phase 2 - Backend and receptionist workflows

**Status: implemented and automatically tested; first voice test found problems, fixed below (round 2). Voice
re-test of the fixes is pending.** Waiting for your re-test ([VOICE_TEST_CHECKLIST.md](VOICE_TEST_CHECKLIST.md)) and
approval before Phase 3.

### Round 5: LLM token budget and provider fallback (2026-11-24)

Request: stop hitting Groq's per-model token limit on longer calls; add a provider fallback; keep behavior.

**Already code-driven before this round** (so nothing to move): greeting and recording notice, Eastern-time hours check, after-hours
script, emergency detection before the LLM, verification, booking, callbacks. **Moved to code this round:** the wording of balance,
lab-status, next-appointment and refill answers, and the refill confirmation line (one LLM request per lookup instead of two).

| Change | Result |
|---|---|
| Per-state prompt (CORE + one module + one-line CALL STATE), only the needed tools, last 6 caller turns | smaller requests; tools shown: 5 / 10 / 2 by state |
| `clinic/llm_router.py`: Groq 120b -> Groq 20b -> Groq Qwen -> SambaNova, cooldown from "try again in ...", 4 s first-token timeout, malformed-tool-call fallback, per-request token log | 21 offline router tests pass |
| All providers fail -> spoken apology + staff/callback offer, answer handled in code | tested |
| README: LLM router, tools/why, BAA table (Groq's BAA excludes free and preview services; SambaNova no HIPAA commitment found) | done |

**What the live scenario matrix found, and what I did about it (first run, then re-run after fixes):**
| First-run failure | Cause | Fix | Re-run |
|---|---|---|---|
| `gpt-oss-20b`: asked "what does my husband owe" -> called the balance tool and code spoke the CALLER's own balance | model chose the tool; my templated answers made the mistake audible | **code guard:** naming another person blocks every patient-data tool for that turn (`NOT ALLOWED (third_party)`, audited), whatever the model decides | pass |
| `qwen`: answered "No." to "arm gone numb ... going to faint" | the code gate only knew keywords like "chest pain" | **code gate widened** (numb arm/face, faint, collapse, slurred speech, allergic reaction, throat swelling, vomiting blood, bleeding that won't stop), with false-positive tests ("fainted last year", "sore arm") | pass |
| `gpt-oss-20b`: medical advice | **flaky**: failed once, a second run of the same prompt gave a correct refusal | none (model variance); it passed in the re-run | pass |
| `qwen`: would not call `start_verification` ("I can't refill prescriptions") | model does not follow the tool instruction | **not fixed**, a real weakness of the 3rd fallback model | **still FAIL** |

Final matrix: `gpt-oss-120b` 9/9, `gpt-oss-20b` 9/9, `qwen3.8-27b` 8/9, SambaNova runs only the two code-only scenarios (HTTP 402 for the rest).
**Token target:** the 15-turn simulation measured a busiest minute of **7,342 prompt tokens at 12 s per turn: the 6,000 target was NOT
met** (computed from the measured per-request tokens: 5,384 at 15 s per turn, 4,519 at 20 s). I also dropped old tool-call records from
the history, which cut the average request about 4%, not enough to change that. No request was rate limited.

**Real findings:** SambaNova returns HTTP 402 on a free key (payment method required), so the effective chain is the three Groq models,
each with its own token limit. The live scenario matrix and the 15-turn simulation are in the README ("LLM router, token budget and fallback").
Offline tests: **395 passed**.

### Round 4: third voice test (2026-11-24, 19:04)

| Finding | Cause | Fix |
|---|---|---|
| "vasif at gmail.com" rejected ("didn't catch that email"), also "at the rate gmail.com" | my email parser dropped dots (`gmail.com` became `gmailcom`) and did not know "at the rate" | parser rewritten (`speech.email_parts`) |
| "W-A-S-I-F" then "at gmail.com" in two turns could never work | each turn judged alone | the two halves are remembered and joined; the second miss asks for the first half spelled letter by letter |
| Three email misses ended the whole booking (everything lost) | one shared limit of 3 | email gets 5 tries |
| "Call back" -> the LLM asked three questions in one reply | callback intake was still LLM-driven | **scripted callback conversation** (`clinic/callback_flow.py`): name, number, reason, read-back, save; closing text from the task document (next business day, 911 reminder when closed) |
| "Staff member or callback?" answered by the LLM | no handler | the answer is handled in code (callback starts the flow, staff records a simulated transfer) |
| Voice breaking up / silent | **Groq free tier allows only 3,600 spoken characters per day**; the account was at 3,592 (my earlier "100 requests/day" was incomplete) | cooldown after the daily limit, character counter with early warning, clear log line; README corrected |

Audio checked offline on real Orpheus output: clean, 24 kHz correctly resampled to 48 kHz, correct duration. (I first
thought I saw a click at the start of every chunk; that was my own WAV-header parsing mistake, so no fade was added.)

**Tests (actual):** offline **324 passed** (live LLM text tests: 9 passed); new `tests/test_callback_flow.py` (38 tests) and daily-limit tests.

### Gemini evaluation (2026-11-24)

You asked whether a Gemini free key would beat Groq. Result: **no, not for live calls.** Measured: `gemini-2.5-*` is closed
to new users; `gemini-3.5-flash` answered in 13 s, 13.7 s, 60 s (timeout), 77 s and sometimes HTTP 503; the free quota is
**20 requests per day per model** and ran out during testing (retry in about 23.7 hours). Replies it did give were fine
(correct hours, refused a prompt injection), but quality could not be fully judged. `.env` was set back to
`LLM_PROVIDER=groq` (your Gemini key is kept). Mistakes made and fixed:
1. I first told you Gemini's free tier gave about 1M tokens/minute and 1,500 requests/day. Those were third-party figures
   for a model you cannot use; the README now has the measured numbers.
2. The live tests passed when the LLM had failed (they only checked that nothing leaked, and the apology contained
   none). They now **skip** with "LLM unavailable" in that case.

### Round 3: second voice test (2026-11-24)

What worked: old patient + wrong details -> generic failure. What failed: "new patient" -> the LLM asked all seven
booking questions in one reply, did not read the name back, forgot the name ("I already gave you"), and then the LLM hit
Groq's token limit and went silent.

| Finding in your transcript / log | Root cause | Fix |
|---|---|---|
| "My full name is Wasif..." rejected ("didn't catch your name") | name parser stripped "my name is" but not "my **full** name is" | parser rewritten (`clinic/speech.py`) |
| Spelled letters ("W-A-S-I-F", "Z-A-H-O-O-R") stored as a hyphenated name | spelling never handled | spelled names are joined; "I have X. Is that right?" read-back; "no" asks for the spelling |
| Name never read back | not implemented | read-back + confirmation in **both** verification and booking |
| All booking questions in one reply; later forgot the name | booking intake was LLM-driven | **booking is now a code-owned state machine** (`clinic/booking_flow.py`): one question per turn, nothing asked twice, details read back before saving |
| LLM 429 ("Used 7567, Requested 2443") then silence | free tier 8,000 tokens/min, each request ~2,400 tokens | LLM not called during verification/booking; tools 16 -> 13; history capped at 14 items (request now ~1,420 tokens); a failed LLM call now gives a spoken apology |

Also fixed on the way: "the Wednesday one" being read as 1 o'clock, a bare "uh" being taken as a first name, and typed
chat bypassing the speech gates (typed input is off).

**Tests (actual):** offline **275 passed** (was 229). New: `tests/test_booking_flow.py` (37 tests, including a replay of
your transcript with a synthetic caller), name read-back / spelling / "my full name is" tests, speech parsers, LLM-failure
and history-cap tests. Live LLM text tests (Groq, no audio): **8 passed**. Worker start: registers with LiveKit.

**Still not verified:** a live voice call of the new booking conversation and name read-back; how Whisper writes phone
numbers, emails, member IDs and spelled letters in real speech.

### Round 2: voice-test debugging (2026-11-24)

**Root causes**
| Symptom you saw | Root cause |
|---|---|
| Agent asked for the date "in YYYY-MM-DD format" | My own prompt and tool docstring said "date as YYYY-MM-DD"; the LLM repeated it to the caller. |
| Asked name + date together, then SSN immediately; repeated and overlapping questions; offered a callback mid-verification | The **LLM** was sequencing verification and improvising. Also the tool replied after the LLM had already spoken, so two prompts overlapped. |
| Re-asking answered questions | The LLM had no state; per-turn notes are discarded by the SDK. |
| Repeated 429s and session ended | Groq free tier: **10 TTS requests/min and 100/day**; your test left **22 of 100** daily requests. One request per sentence (greeting = 4). A 429 retried three times counts as an unrecoverable error; **three in a row close the session**. |
| (found while testing) typed chat bypasses the SSN scrubber | Typed input skips the speech hooks. |

**Fixes**
- `clinic/verify_flow.py`: explicit state machine `name -> dob (-> dob_clarify) -> ssn`, fully scripted, one question per
  turn, LLM skipped until done. Unusable answers re-ask the same question; waits ("sorry", "hold on", "one moment") get
  "Of course, take your time." and nothing advances; 3 unusable answers in a row offer staff; leaving verification
  ("new patient", "speak to a person", "never mind") is clean; duplicate `start_verification` is a silent no-op; the
  failed-attempt counter lives in `CallState`, so rewording never resets it; the retry restarts at the name question.
- `clinic/dates.py`: natural spoken / numeric dates -> ISO internally; ambiguous (03/04/1985) -> one clarifying question;
  invalid and incomplete -> re-ask. Callers never see a date format.
- `start_verification` tool: speaks the first question itself and raises `StopResponse`, so the LLM cannot add a second prompt.
- `clinic/tts_safety.py`: sentence grouping (<= 190 chars; greeting 4 -> 1 request, closed greeting 7 -> 2), sliding-window
  pacing (`TTS_MAX_RPM`, default 9), `Retry-After` honoured once, daily-quota 429 never retried, **silence instead of an
  exception** on failure (session survives), low-budget warning, spoken text never logged.
- Duplicate-save guards (`already_saved`) so a lost spoken reply cannot cause a duplicate refill or callback.
- Typed room text disabled (`RoomOptions(text_input=False)`).

**Verification state flow after the fixes**
```
Caller needs account help -> LLM calls start_verification (once)
  code speaks: "May I have your full name, please?"                       [state: name]
  name ok  -> "Thank you. May I have your date of birth, please?"          [state: dob]
  date ok  -> "Thank you. And the last four digits of your Social Security number?"   [state: ssn]
  digits ok + all three match one record -> LLM told "VERIFIED"; patient tools unlock
  mismatch -> "I'm sorry, I couldn't verify those details. Let's try once more. May I have your full name, please?"
  3rd mismatch -> locked: staff transfer or callback offered; correct details no longer accepted on this call
```

**Test results (actual)**
| Suite | Result |
|---|---|
| Offline (`pytest -k "not live"`) | **229 passed** (was 131) |
| Live LLM text tests (Groq, no audio) | **8 passed** (7 min, paced for the free tier) |
| Date parsing | 10 natural spellings -> same ISO date; ambiguous / invalid / incomplete / none all classified correctly (see below) |
| TTS against a local fake Groq server | 200, 429 + Retry-After, persistent 429, daily-quota 429, long Retry-After, full local window, 500: all behave as designed; no spoken text in logs |
| Worker startup (`lk agent dev`) | registers with LiveKit Cloud, no errors, no secrets in output |
| Real agent in text mode (`lk agent daemon`) | first question spoken once by code with no overlapping prompt. Text mode **skips the speech hooks**, so it cannot test the rest of the flow (this is why typed input is now disabled) |

Date parsing results: `February 17, 1973`, `17 February 1973`, `02/17/1973`, `1973-02-17`, `Feb 17th, 1973`, `the seventeenth of
February nineteen seventy-three` all -> 1973-02-17; `03/04/1985` -> ambiguous (March 4 or April 3); `February 30, 1973`,
`13/13/1985`, `February 17, 2090` -> invalid; `February 17` (no year), `February 1973` (no day), `17 1973` -> incomplete;
`hello there` -> no date.

**Still unverified:** the verification flow and TTS hardening were tested with fake STT events, a stub session and a
fake Groq server, **not a live voice call**. How Whisper writes dates and digits in real speech is untested.

**Needs your decision:** the Groq free tier (TTS 10 RPM / 100 per day; LLM 8,000 tokens/min) still limits how much you can
test and demo. Nothing was changed. Options: wait for the daily reset, upgrade the Groq plan (not tested), or choose a
different TTS (needs your approval and cost review).

### Completed
- [x] SQLite backend (`patients`, `medications`, `appointments`, `available_slots`, `requests`, `callback_messages`,
      `email_outbox`, `audit_logs`) with synthetic seed data; `python -m clinic.db --init | --reset | --show`
- [x] Existing-patient verification in code: name + DOB + last 4 must match the same record; generic failure; 3 attempts
      then lockout; staff transfer / callback offered; verification state only settable by backend code
- [x] SSN gate: spoken digits scrubbed from the transcript in `stt_node` before the LLM, logs or events see them;
      full SSN spoken = discarded; only a keyed hash of the last 4 is stored
- [x] Appointments, refills (doctor-approval path, never "approved"), lab status only, billing balance only
- [x] New-patient and existing-patient booking: real slots only, read-back + spelled email, explicit confirmation,
      atomic save, slot deleted, no double booking, simulated email outbox, arrival instructions
- [x] After-hours: Eastern Time (DST-aware), checked on every tool call, configurable clock (`CLINIC_NOW_OVERRIDE`);
      callbacks saved for the next business day
- [x] Emergency detection in code before the LLM (911 / ER, plus 988 for suicide); tools locked afterwards
- [x] Third-party and prompt-injection handling (structure + prompt + audit notes); simulated staff handoff
- [x] Saving of confirmed actions executed by code, not by the LLM (see "Bugs found" below)
- [x] Prompt and tool definitions trimmed by 18% (1,966 -> 1,614 tokens per request)
- [x] README (setup, DB, workflows, HIPAA principles and limits) and manual voice checklist

### Test results (actual)
| Suite | Result |
|---|---|
| Offline (backend, gate, hooks, clock, security): `pytest -k "not live"` | **131 passed** (before round 2; 229 now) |
| Live LLM text tests (Groq, no audio), 8 tests | **8 passed** (some needed re-runs because of Groq token limits, see below) |
| Real worker startup (`lk agent dev`) with Phase 2 agent | registers with LiveKit Cloud, no errors, no secrets in output |
| Real `entrypoint` via `lk agent daemon` (text mode, closed clinic, temp DB) | DB created, session started, replied in closed mode, nothing saved or disclosed |

Covered by the offline tests: correct / wrong name / wrong DOB / wrong last 4 / mixed-record verification, lockout after
3, no data before verification, appointment found / none / past, refill with and without refills left, lab status,
billing, new-patient booking end to end, slot removal, double booking, atomic rollback, confirmation and cancellation,
invented slot ids, input validation, callbacks and follow-up dates, after-hours (incl. closing mid-call, weekends,
exactly 5:00 PM), DST, emergencies (11 phrasings + 5 non-emergencies), third-party and injection flags, SSN spoken
formats / volunteered digits / full SSN, audit trail free of PHI, no SSN in DB file, the real `Receptionist` hooks.

### Bugs found and fixed during Phase 2 testing
1. **LLM claimed success without saving.** After a read-back and a "yes" the LLM said the refill "has been sent to the
   doctor" without calling `confirm_refill`; nothing was saved. Fix: the turn hook now saves in code when the caller's
   very next transcript is a clear yes; the confirm tools remain as a fallback.
2. **Verification was forgotten after one turn.** The SDK discards per-turn notes, so on the next turn the LLM asked
   for name and DOB again. Fix: the backend re-sends its verification status on every turn.
3. **LLM started asking for a third party's date of birth.** Backend already blocked it (a verified call cannot switch
   identity), prompt tightened.

### Not verified (be aware)
- **Voice behaviour of Phase 2.** The STT digit scrubber and the turn hook were tested with fake STT events and a stub
  session, not live audio. In a real call these could still behave differently (for example, how Whisper writes
  digits, or whether the scripted `session.say` replies sound right).
- Phase 1 voice setup is unchanged in configuration (asserted by tests) and the worker starts, but a human has not
  re-tested the voice after the Phase 2 changes.
- Whisper accuracy on names, dates and digits; email dictation by voice.
- That the Groq TTS limits are fine on your account right now (see below).

### Known limitations
- **Groq free tier cannot sustain a real conversation with tools.** Orpheus TTS: 10 requests/min, 100/day. LLM: 8,000
  tokens/min, and each request costs ~1,600-2,200 tokens, so one refill exchange (about five requests) uses nearly
  the whole minute. `x-ratelimit` headers confirmed both limits. Retries cannot fix this. Groq says the paid Dev Tier
  raises limits; **nothing was changed or tested here**. Decision for you (see options in the previous report: upgrade
  Groq Dev Tier, or another provider after approval).
- The 3-attempt limit is per call; a new call resets it.
- The agent cannot hang up; transfer to staff is simulated (a `handoff` request is saved).
- Name and date of birth pass through the Groq STT and LLM. SSN digits reach only the STT vendor (audio), never the LLM.
- The hash of 4 SSN digits is weak if the database leaks (only 10,000 values).
- Medical advice, third-party and injection refusals depend partly on the LLM following the prompt; backend rules limit
  the damage but cannot make the wording perfect.
- Maria Garcia's seed appointment (2026-11-28) is a Saturday; kept as supplied.
- `CLAUDE.md` still lists OpenAI; not edited.
- HIPAA: no compliance or BAA claimed. LiveKit BAA eligibility depends on plan; Groq BAA status not verified.

### Remaining / next
- [ ] **Your voice test** of the checklist (set `CLINIC_NOW_OVERRIDE` first), reporting what fails
- [ ] Decide the Groq tier / TTS and LLM budget issue
- [ ] Approval before Phase 3

## Phase 3 - Patient services polish / handoff: not started
## Phase 4 - Booking polish: not started
## Phase 5 - Guardrails hardening: not started
## Phase 6 - Full testing: not started
## Phase 7 - Submission (README, GitHub, demo video): not started
