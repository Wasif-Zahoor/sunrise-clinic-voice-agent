# Sunrise Family Clinic - Technical notes

> The detailed reference behind the short [README](../README.md): every setting, the LLM router, the TTS fallback chain and its
> measurements, transcripts, per-scenario results and known limitations. Other files here: [PROGRESS.md](PROGRESS.md) (development log and
> requirements audit) and [VOICE_TEST_CHECKLIST.md](VOICE_TEST_CHECKLIST.md) (manual voice test script).

Real-time voice agent for a **fictional** clinic. All patient data is **synthetic** (see
`data/seed.json`). This is a prototype: it is **not certified HIPAA compliant** and no vendor BAA
has been verified (see [HIPAA principles and limits](#hipaa-principles-and-limits)).

**Status: all assignment workflows are built and covered by automated tests (offline, fake STT/LLM/TTS).** The developer's live
voice calls on 2026-11-25 completed new-patient and existing-patient booking, balance and lab lookups; every problem found in those calls
was fixed and covered by a regression test, but **the fixes of the last two rounds have not been re-confirmed by a live call**. The
manual script is [VOICE_TEST_CHECKLIST.md](VOICE_TEST_CHECKLIST.md).
Requirement-by-requirement audit with honest verified / not-verified labels: [PROGRESS.md](PROGRESS.md#requirements-audit-2026-11-24).

## Stack

| Part | Choice | Exact ID |
|---|---|---|
| Voice transport | LiveKit Cloud (browser mic) | - |
| Agent framework | `livekit-agents` | 1.8.13 |
| Speech-to-text | Groq Whisper | `whisper-large-v3-turbo` |
| LLM + tools | Groq, with automatic fallback (router) | `openai/gpt-oss-120b`, then `gpt-oss-20b`, `qwen3.8-27b`, SambaNova |
| Text-to-speech | Chain, default `deepgram,cloudflare,local` (see [Text-to-speech](#text-to-speech-deepgram-with-local-and-cloudflare-fallbacks)) | Deepgram `aura-2-thalia-en`; local Piper `en_US-hfc_female-medium`; optional Cloudflare `@cf/deepgram/aura-2-en`; optional Groq `canopylabs/orpheus-v1-english` |
| Voice activity / turns | Local Silero VAD | - |
| Database | SQLite (standard library) | `data/clinic.db` (git-ignored) |

Model IDs were checked against Groq's live model list on 2026-11-24. Override with `STT_MODEL`,
`LLM_MODEL`, `TTS_MODEL`, `TTS_VOICE` in `.env`. No Twilio / phone integration yet.

### Why this stack

The assignment allows free tiers and any framework, and asks for real-time voice, backend-enforced verification and honest handling of
limits. The choices follow from that:

* **LiveKit Agents + LiveKit Cloud.** A browser call with a microphone needs no phone number, and the framework supplies the real-time
  parts (audio transport, voice-activity detection, turn handling, interruptions) so the project's own code is the clinic logic. The agent
  is a Python worker, so the same code can later sit behind a SIP / Twilio number.
* **Groq for speech-to-text and the LLM (Whisper `whisper-large-v3-turbo`, `openai/gpt-oss-120b` with tool calling).** Fast responses
  and a free tier. The LLM is behind an ordered router (other Groq models, SambaNova, Cloudflare) because a free tier can run out
  mid-call. The project was first planned around OpenAI; the free tiers were the reason for moving.
* **Deepgram Aura-2 for speech (with a local Piper voice and Cloudflare as fallbacks).** Natural female voice on a free credit. The
  first free voice (Groq Orpheus) allowed only about 3,600 characters a day, which ended calls mid-sentence; the local voice has no
  network, key or quota, so a call can always finish. The fallbacks were matched to Deepgram's voice so a switch is hard to notice.
* **SQLite.** No server to run, in the Python standard library, and enough for the tables this project needs (patients, appointments, slots, requests, callbacks, audit log).
* **Deterministic code for the safety-critical parts.** Verification, authorization, after-hours rules, emergency detection, scheduling and
  the booking / callback / refill conversations are Python state machines, not prompts. The LLM is skipped for them, which also keeps SSN
  digits away from it and saves tokens. The LLM handles the free-form remainder (greetings, off-topic requests, refusals).

## Setup (Windows PowerShell)

```powershell
cd "D:\EZ MD task\Voice AI Agent"
uv venv .venv --python 3.11              # or: py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1             # every new terminal. Run `deactivate` first if another venv is active
uv pip install -r requirements.txt       # or: pip install -r requirements.txt
Copy-Item .env.example .env              # then edit .env yourself; never paste keys into chat
```

The LiveKit CLI follows `VIRTUAL_ENV`; an active venv from another project gives a
"livekit-agents version ... is too old" error.

**Environment variables** (`.env`, never committed):

| Variable | Purpose |
|---|---|
| `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` | LiveKit Cloud project keys |
| `GROQ_API_KEY` | Groq STT, LLM, TTS |
| `SSN_HASH_KEY` | Secret for hashing the last 4 SSN digits in the DB (insecure dev default if unset) |
| `DB_PATH` | Optional SQLite path (default `data/clinic.db`) |
| `CLINIC_NOW_OVERRIDE` | Optional simulated Eastern time, e.g. `2026-12-01T20:30` |
| `LLM_PROVIDER` | `router` (default), `groq` (one model) or `gemini` |
| `LLM_ORDER` | optional comma list, default `groq-gpt-oss-120b,groq-gpt-oss-20b,groq-qwen,sambanova` |
| `SAMBANOVA_API_KEY` | optional fallback provider (chat needs a payment method) |
| `LLM_FIRST_TOKEN_TIMEOUT` | seconds before trying the next provider (default 4) |
| `GEMINI_API_KEY` | Google AI Studio key, only needed when `LLM_PROVIDER=gemini` |
| `TTS_MAX_RPM` | Max Groq TTS requests per minute this agent sends (default 9; free tier allows 10; 0 = no pacing). Only used when `groq` is in the TTS chain |
| `TTS_PROVIDER_ORDER` | TTS providers, first choice first (default `deepgram,cloudflare,local`; also `groq`) |
| `TTS_HEALTH`, `TTS_HEALTH_FILE` | `TTS_HEALTH=off` disables the memory of failed voices between calls (see Provider memory); default file `data/tts_health.json` |
| `TTS_HOLD`, `CLOUDFLARE_TTS_SPEAKER` | `TTS_HOLD=0` plays network audio as it arrives instead of holding each clip (see Whole-clip hold); the Cloudflare speaker defaults to the one in `DEEPGRAM_TTS_MODEL` |
| `DEEPGRAM_API_KEY`, `DEEPGRAM_TTS_MODEL` | Deepgram key; voice model (default `aura-2-thalia-en`) |
| `PIPER_MODEL`, `PIPER_LENGTH_SCALE`, `PIPER_VOLUME` | Local voice file (default `models/piper/en_US-hfc_female-medium.onnx`, 63 MB, git-ignored) and its pace / loudness (defaults 1.04 and 0.4, tuned to match Deepgram) |
| `CLOUDFLARE_API_TOKEN` (or `CLOUDFLARE_API_KEY`), `CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_TTS_MODEL` | Optional Cloudflare Workers AI. The key and the account id are both required, for TTS and for the last LLM fallback |
| `TTS_CHAR_WARN`, `TTS_USAGE_FILE` | Warn once a paid TTS provider has spoken this many characters in total (default 100000); counter file (default `data/tts_usage.json`) |
| `TRANSCRIPTS` | `on` (default) or `off`: save a redacted text transcript of every call (see [Transcripts](#transcripts)) |
| `TRANSCRIPT_DIR` | Transcript folder (default `transcripts/` in the project; git-ignored) |

One-time Groq step: accept the Orpheus terms at
https://console.groq.com/playground?model=canopylabs%2Forpheus-v1-english

### Database

```powershell
python -m clinic.db --init     # create tables + load synthetic data (also runs automatically on agent start)
python -m clinic.db --reset    # wipe and reload (do this to get booked slots back)
python -m clinic.db --show     # print saved requests, callbacks, appointments, simulated emails, audit log
```

### Run the agent

```powershell
.\.venv\Scripts\Activate.ps1
lk agent dev agent.py --log-level INFO
```

Open the **Agent console** link it prints, click **Start a session**, allow the microphone.
To simulate office hours (your real time may be outside Eastern hours), set the clock first in the
same window:

```powershell
$env:CLINIC_NOW_OVERRIDE = "2026-11-30T10:00"   # Monday 10:00 AM Eastern = open
$env:CLINIC_NOW_OVERRIDE = "2026-11-30T20:30"   # Monday 8:30 PM Eastern = closed
Remove-Item Env:CLINIC_NOW_OVERRIDE             # back to real time
```

### Tests

```powershell
$env:PYTHONUTF8 = "1"
python -m pytest -q -k "not live"      # offline: no network, under a minute (count in PROGRESS.md)
python -m pytest -q -s                 # also runs the live-LLM text tests (slow: paced for Groq limits)
```

Tests never touch `data/clinic.db`; each gets a throwaway database and a fixed clock. They do **not**
test audio, STT accuracy, TTS quality or latency.

## How it works

```
mic -> LiveKit -> Groq Whisper STT -> [stt_node gate: scrub SSN digits]        (code)
                                   -> [on_user_turn_completed: emergency,       (code)
                                       verification result, injection hints]
                                   -> Groq LLM <-> tools -> backend services    (code + SQLite)
                                   -> grouped TTS chain (Deepgram -> local Piper -> ...) -> speaker
```

* `clinic/gate.py` - spoken-digit parsing and scrubbing, emergency / confirmation / injection detection.
* `clinic/turn.py` - per-turn decisions before the LLM answers (scripted replies for emergencies and
  verification failures).
* `clinic/services.py` - **every rule is enforced here**: verification state, after-hours lockout,
  emergency lockout, explicit confirmation, slot integrity, validation.
* `clinic/tools.py` + `agent.py` - 12 thin LLM tools. None accepts a patient id, a verified flag or SSN digits.
* `clinic/db.py` - schema, seed, atomic transactions. `clinic/security.py` - masking, hashing, audit.
* `clinic/clock.py` - Eastern Time (DST-aware) and the test clock.

### Database design

| Table | Purpose |
|---|---|
| `patients` | existing + new patients; `ssn_last4_hash` (keyed hash; the full SSN is never stored); lab status, balance |
| `medications` | per patient: refills remaining, pharmacy |
| `appointments` | booked and seed appointments |
| `available_slots` | open slots; a row is **deleted** when booked |
| `requests` | every refill, booking, callback and handoff: patient / new-patient id (`UNVERIFIED` if unknown), type, timestamp, short summary, status |
| `callback_messages` | name, callback number, short reason, follow-up date (next business day) |
| `email_outbox` | simulated confirmation emails (nothing is ever sent) |
| `audit_logs` | who / which patient id / action / outcome / time. No names, DOB, SSN or free text |

### Tools exposed to the agent (and why so few)

`start_verification`, `start_booking`, `start_callback`, `get_next_appointment`, `prepare_refill`, `confirm_refill`,
`get_lab_status`, `get_balance_due`, `request_staff_transfer`, `cancel_pending_request`, `report_emergency`, `check_clinic_hours`.
Each tool schema is re-sent on every LLM request, so there are few, and the model only sees the ones that fit the current state.
The three conversations (verification, booking, callback) are single tools that hand over to code, so the model never has to
juggle ten booking tools. Simple facts are spoken from templates by code; the backend refuses every call that is not allowed.

## Workflows

**Opening requests are routed in code, not by the LLM** (`clinic/intent.py`). A clear refill, lab, balance, "my appointment" or
"schedule an appointment" request starts the right conversation immediately; for an unverified caller, verification starts first and the
request is continued right after it. Not routed when the clinic is closed, during another conversation, or when the turn was flagged as
an injection attempt or a request about someone else (the LLM sees that note). **Refills are a code-owned conversation**
(`clinic/refill_flow.py`): the drug name is matched among the *verified caller's own* medications only, tolerating speech-to-text errors
("Lisey Nuperell" or a spelled-out "L-I-S-I-L N-O-P-R-I-L" both find Lisinopril; similarity at least 0.55 and clearly ahead of the
runner-up), then read back for a yes. After three misses it offers staff or a callback. A thank-you, a "yeah" or the request said again
is not a miss; the question is simply asked again.

**A caller who registered as a new patient earlier in the same call** and then asks about labs, balance, their appointment or a refill
is not asked to verify again: a new patient has no stored last four digits, so verification could never succeed, and nothing is on file
for them yet. The backend (`services.get_new_patient_info`) answers only from the record created in this call ("Since this is your first
visit, I don't have any lab results on file for you yet", the appointment they just booked, and so on), writes an audit row, follows the
after-hours / emergency / third-party rules, and does **not** mark the caller verified, so every verified-only tool still refuses them. A
caller who then verifies as an existing patient gets the normal answers. Not handled: a *second* booking by the same new patient in one call
(the duplicate check sends it to staff).

**Booking details that are checked in code.** "Book 1 for 10am" books the time that was said (a clock time wins over a stray option number;
`speech.choose_slot`). An insurance provider that is not a well-known insurer (speech-to-text turned "self-pay" into "Sophie" in a real call)
is read back ("I heard Sophie. Is that your insurance provider, or would you like to self-pay?") and only kept after a yes.

**Identity verification (existing patients).** A code-owned state machine (`clinic/verify_flow.py`) runs the
conversation. The LLM only starts it (`start_verification`); after that every reply is scripted, one question at a
time, and the LLM is skipped until it finishes:

```
idle --start_verification--> name --> name_confirm --> dob --(ambiguous date)--> dob_clarify --> ssn --> verified
                              |          |  (read back)    |                                      |
                              |          +-- "no": spell    +-- unusable answer: re-ask             +-- 3 matching failures -> locked
                              +-- name + date said together: the date step is skipped after the read-back
```

1. "May I have your full name, please?" A first name alone gets "And your last name?"; spelled-out names
   ("W-A-S-I-F") and "my full name is ..." work. The name is then **read back**: "I have Jane Doe. Is that right?"
   (a "no" asks for the spelling, so a misheard name never burns one of the three attempts).
2. "Thank you. May I have your date of birth, please?" Natural speech is accepted ("February 17th, 1973",
   "17 February 1973", "02/17/1973", "the seventeenth of February nineteen seventy-three") and normalized internally.
   Numeric dates that read two ways (03/04/1985) get one clarifying question. Impossible or incomplete dates are
   re-asked. Callers are never shown or asked for a date format.
3. "Thank you. And the last four digits of your Social Security number?" Digits are removed from the transcript before
   the LLM sees them and checked in code.

Name, date and last four must all match **the same** record. A mismatch always gets the same reply ("I'm sorry, I
couldn't verify those details...") and then restarts at the name question; the audit log records only
"failed (attempt n)". **Three mismatches lock verification for the call** (the counter lives in `CallState` and is
not reset by rewording), and the agent offers a (simulated) staff transfer or a callback. Only backend code can set the
verified state.

Not-yet-usable answers are different from failed attempts: "um", "sorry", "hold on a moment" get "Of course, take your
time." and nothing advances or is counted; a half-heard name or date is re-asked briefly; **three unusable answers in a
row** offer staff or a callback. A caller who says they are a new patient, asks for a person, or says "never mind" leaves
verification cleanly. A duplicate `start_verification` call is a silent no-op. Callbacks are offered only on refusal,
lock-out, a request for staff, or three unusable answers - never while a valid answer is still being processed.

**Appointment lookup**, **lab status** (status string only), **billing** (balance only): verified callers only;
tools act only on the caller verified in this call, so no tool can read another patient.

**Refill.** Medication must be named by the caller (the list is never volunteered). The agent reads back
medication and pharmacy; the request is saved only after a clear "yes". With 0 refills left the request is
saved as `pending_doctor_approval` and the agent says it goes to the doctor. Nothing is ever "approved".

**Booking (new and existing patients) is also a code-owned conversation** (`clinic/booking_flow.py`). In a real call the LLM
listed all seven intake questions in one reply, forgot answers it had already been given and never read the name back.
Now the LLM only calls `start_booking`; after that every turn is ONE scripted question, answers are parsed by
deterministic code (`clinic/speech.py`), and the LLM is not called until the booking is done (which also saves most of the
LLM token budget):

```
first_visit? -> name -> name_confirm (read back) -> date of birth -> phone -> email -> insurance (-> provider -> member ID)
             -> reason (a few words) -> doctor/day preference -> choose one of the real openings -> read-back -> saved
"old patient" -> verification first (name, date, last four) -> then only what is missing (usually email, reason, time)
```

The caller can say "no" to the read-back and name what to fix ("the email is wrong", "the time"), spell a name, ask for
other openings ("none of those"), wait, or leave ("never mind", "speak to a person"). The openings are real database
slots (2 or 3 offered); the final read-back says name, date of birth, phone, date and time, doctor, reason and the email
**spelled out**; only after a clear "yes" does one transaction delete the slot, create the patient (`N001`...), the
appointment, the request and the simulated email. A slot cannot be booked twice, and a "new patient" who matches an existing
name and date of birth is routed to staff without revealing that a record exists.

**Confirmations are decided and executed in code.** A refill, booking or callback is saved only if the caller's
actual transcript, on the very next turn after the read-back, is a clear yes with no negation ("no", "wait",
"change"). In that case the turn hook performs the save itself and tells the LLM the result. This exists because
the LLM was observed telling a caller "your request has been sent" **without calling the confirm tool**, so
nothing was saved. The `confirm_*` tools remain as a fallback with the same checks, and the LLM cannot
self-confirm.

**After hours.** The clock (Eastern, DST-aware) is checked on **every** tool call, so a call that crosses 5 PM
loses patient access mid-call. When closed, every patient service, verification attempt and booking is refused by
the backend, even for a caller who was verified earlier. The agent states the hours, says to call 911 for
emergencies, and can only take a callback (name, number, a few words of reason), read back, confirm, save, with
follow-up on the next business day (weekends and Friday evenings roll to Monday). A clear request for patient information or a
booking after hours (labs, balance, refill, "my appointment", scheduling) is answered in code, the same way every time, with no
lookup and no LLM (`intent.closed_say`): "The clinic is closed, so I can't share or change patient information right now. Our hours
are ... I can take a message for our team. Would you like that?" (a "yes" starts the message in code), or, once a message has been
saved in the call, "Your message is recorded, and our team will follow up on the next business day." instead of offering it again.

**Emergencies.** Detected in code before the LLM (chest pain, trouble breathing, suicidal thoughts, "emergency",
overdose, stroke...). The caller gets a scripted 911 / nearest-ER message (plus 988 for suicide), all tools are
locked for the rest of the call, and pending actions are discarded. `report_emergency` is a backup for phrases the
patterns miss.

**Medical advice, third parties, prompt injection.** Prompt rules plus structure: tools cannot reach any patient
except the verified caller; injection and third-party phrasing add a system note for the LLM and an audit row
(injection). These last parts rely on the LLM following the prompt; the backend limits the damage.

## LLM router, token budget and fallback

**Why:** the free Groq tier allows 8,000 LLM tokens per minute per model, and the whole prompt, tool list and history are
re-sent on every LLM request. Two things keep calls inside that budget: (1) most of the call does not use the LLM at all,
(2) when a model is limited, the agent moves to the next one instead of going silent.

**What the LLM does and does not do.** Code owns everything deterministic: the greeting and recording notice, the Eastern-time
office-hours check (with `CLINIC_NOW_OVERRIDE` for demos), the after-hours script, emergency detection (before the LLM),
verification (name, date of birth, last four, matching, three attempts), booking, callbacks, read-backs and saving, and now
the wording of simple answers (balance, lab status, next appointment, refill read-back and confirmation). The LLM only
understands the caller's request, picks a tool, and phrases open-ended replies (hours, address, politely declining medical advice,
third-party and injection requests). It never receives SSN digits or the patient database.

**Smaller requests (`clinic/prompts.py`, `agent.llm_node`).**
* The system prompt is a short CORE (about 330 tokens: facts, style, safety, stop/goodbye) plus ONE module for the current state
  (closed / not verified / verified / locked), so booking or refill rules are only present when they apply.
* Only the tools that state needs are shown to the model (5 before verification, 10 after, 2 when closed). The backend still refuses
  everything regardless of what the model sees.
* History is the last 6 caller turns verbatim, plus a one-line `CALL STATE` that stands in for older turns (for example
  `verified=yes; waiting for the caller to confirm: none; done this call: refill requested`). It contains no names, dates of birth,
  numbers or other identifiers.
* A simple lookup costs one LLM request, not two: the model picks the tool, code speaks the answer.

**Fallback router (`clinic/llm_router.py`).** A small subclass of LiveKit's own fallback classes (not LiteLLM: LiveKit's
streaming and tool calling then work unchanged on every provider). Default order, configurable with `LLM_ORDER`:

1. Groq `openai/gpt-oss-120b`  2. Groq `openai/gpt-oss-20b`  3. Groq `qwen/qwen3.8-27b` (Groq lists it as a preview model)
4. SambaNova `gpt-oss-120b`. Cloudflare Workers AI is supported but needs `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN`
(no key was available, so it is **untested**). Providers without a key are skipped. Model IDs and the SambaNova base URL were
checked against the providers' live model lists on 2026-11-24.

* **Cooldown, not re-probing.** After a failure a model rests for the time its own error says ("try again in 3.0s" / "52m24s"),
  capped between 1 s and 1 h (30 s if no hint, 600 s for 400/401/403/404, 1 h for 402, 120 s for a malformed tool call). During that
  time it is simply not asked. (LiveKit's stock adapter re-sends the whole conversation to a failed model to "check recovery", which
  wastes tokens on an already limited key.)
* **First-token timeout** 4 s (`LLM_FIRST_TOKEN_TIMEOUT`), at most about 10 s in total, then the next provider. No slow retries.
* **Malformed tool call** (a tool name that does not exist) is treated as a failure and falls back, as long as nothing has been
  said yet. Once part of an answer has reached the caller the router does not switch mid-sentence.
* **If every provider fails** the caller hears: "Sorry, I'm having a technical issue. I can connect you with a staff member, or take a
  message for a callback. Which would you prefer?", the answer is handled in code (callback conversation or a simulated staff
  transfer), and an `llm_unavailable` audit row is written. It never goes silent and never crashes.
* **Logging:** one INFO line per request: provider, model, prompt / completion / total tokens, time to first token. No conversation text.

**Findings from running it for real (2026-11-24).**
* SambaNova: the key lists models, but chat requests return **HTTP 402, a payment method is required**, so it is not usable on this
  free key. The router parks it for an hour and carries on with the three Groq models, which have separate limits each.
* Gemini on a free key is **not recommended**: `gemini-2.5-*` is closed to new users, `gemini-3.5-flash` answered in 13 to 77 seconds
  (or HTTP 503) and the free quota is 20 requests per day. It stays available with `LLM_PROVIDER=gemini` but is not in the router.
  Google's free tier may use prompts and replies to improve its products, so synthetic data only.

### Scripted test of the demo scenarios, per model (live, 2026-11-24)

Each model was run on its own (not through the router) through the same code-first path as a real call. PASS / FAIL / not run
(the model was unavailable). `groq-gpt-oss-120b`: 9 pass, 0 fail, 0 not run; `groq-gpt-oss-20b`: 9 pass, 0 fail, 0 not run; `groq-qwen`: 8 pass, 1 fail, 0 not run; `sambanova`: 2 pass, 0 fail, 7 not run.

| Scenario | groq-gpt-oss-120b | groq-gpt-oss-20b | groq-qwen | sambanova |
|---|---|---|---|---|
| 1 verify + refill | PASS | PASS | **FAIL** | not run |
| 2 failed verification | PASS | PASS | PASS | PASS |
| 3 new patient booking | PASS | PASS | PASS | not run |
| 4 after-hours message | PASS | PASS | PASS | not run |
| 5 medical advice | PASS | PASS | PASS | not run |
| 6 emergency | PASS | PASS | PASS | PASS |
| 7 third-party info | PASS | PASS | PASS | not run |
| 8 prompt injection | PASS | PASS | PASS | not run |
| 9 nothing before verification | PASS | PASS | PASS | not run |

**Failures:**
- `groq-qwen`, 1 verify + refill: did not start verification (called nothing); scripted verification did not verify the correct patient; refill read-back missing (called [], said 'I can’t refill prescriptions, but I can take a message for y'); refill request not saved as pending_doctor_approval
**Not run:** sambanova: the model failed (quota, outage or payment required).

Scenarios 2, 3 and most of 1 and 4 are mostly code-driven, so they pass independently of the model; what changes between models is
tool choice (did it call `start_verification` / `start_booking` / `start_callback`), refusals (medical advice, third party,
injection) and not hinting which detail failed.

### Simulated 15-turn call (real router, 12 s per caller turn)

| Turn | Handled by | LLM requests | Prompt tokens | Model |
|---|---|---|---|---|
| 1 | LLM | 1 | 1055 | openai/gpt-oss-120b |
| 2 | LLM | 1 | 1079 | openai/gpt-oss-120b |
| 3 | code | 0 | 0 | - |
| 4 | code | 0 | 0 | - |
| 5 | LLM | 1 | 1176 | openai/gpt-oss-120b |
| 6 | LLM | 1 | 1246 | openai/gpt-oss-120b |
| 7 | LLM | 1 | 1315 | openai/gpt-oss-120b |
| 8 | LLM | 1 | 1385 | openai/gpt-oss-120b |
| 9 | LLM | 1 | 1438 | openai/gpt-oss-120b |
| 10 | code | 0 | 0 | - |
| 11 | LLM | 1 | 1478 | openai/gpt-oss-120b |
| 12 | LLM | 1 | 1506 | openai/gpt-oss-120b |
| 13 | LLM | 1 | 1535 | openai/gpt-oss-120b |
| 14 | code | 0 | 0 | - |
| 15 | LLM | 1 | 1567 | openai/gpt-oss-120b |

Busiest 60-second window: **7342 prompt tokens** (target under 6000) at one caller turn every 12 s; 11 LLM requests and 14780 prompt tokens in total over 15 turns.

**The target of about 6,000 prompt tokens per minute was NOT met at 12 s per turn in this run** (this script asks five account questions in a row, which is a dense call). No request was rate limited (7.3K is under one model's 8K limit, and the router has two more Groq models with their own limits behind the primary). Computed, not re-run, from the measured per-request tokens, for slower paces:

| Pace (one caller turn every) | Busiest 60 s (prompt tokens) | Under 6,000? |
|---|---|---|
| 12 s | 6560 | **no** |
| 15 s | 5384 | yes |
| 20 s | 4519 | yes |
| 30 s | 3041 | yes |

So the budget holds at 15 s per turn or slower, which is closer to a real phone call (the caller speaks, the agent speaks, TTS adds a few seconds).

## Transcripts

Every call writes one readable text file, `transcripts/<date>-<time>_<call id>.txt`, for reviewing a test call. It is a debugging
aid for **synthetic callers only**, not a medical record, and it is switched on by default (`TRANSCRIPTS=off` disables it).

```powershell
python -m clinic.transcript list          # files, with "INCOMPLETE" for calls that ended without a clean close
python -m clinic.transcript show latest   # print the newest one (or: show <part of a file name>)
```

Example layout (shortened; the first lines are from a real offline run of the hooks, the last line is illustrative):

```
SUNRISE FAMILY CLINIC - CALL TRANSCRIPT (synthetic test call; sensitive details redacted)
Call ID:      c7075dff        Started: 2026-11-24 20:28:02        Notice: spoken in the greeting ...
[20:28:02] AGENT: Thank you for calling Sunrise Family Clinic. This call may be recorded and transcribed for quality purposes.
[20:28:02] AGENT: May I have your full name, please?
[20:28:08] CALLER: [name provided - not stored]
[20:28:09] AGENT: Thank you. I have [NAME] [NAME]. Is that right?
...
Call ended:   2026-11-24 20:31:40 | duration 3m 38s | reason: session closed (participant_disconnected) | caller turns 9, ...
```

* **Notice first.** The greeting says "This call may be recorded and transcribed for quality purposes" and cannot be interrupted.
  The caller's words are stored only after the greeting was fully spoken; if it could not be delivered (for example the TTS quota
  is used up) the call is **not transcribed** and the file says so. With `TRANSCRIPTS=off` the greeting says the call is not recorded.
* **Saved however the call ends.** Each line is flushed as it is written; the end line is written on a normal close, on an error
  close and by the job's shutdown callback, so a dropped call leaves a complete-so-far file. Writing never raises into the call:
  a disk error costs the transcript, not the call (one warning without any text is logged).
* **Order and timing.** `[HH:MM:SS] CALLER:` / `AGENT:` lines in the order they happened; an agent reply the caller cut off is marked
  `AGENT (interrupted by caller)`; the text is what LiveKit kept for that reply, which may include words that were generated but
  not yet spoken (not observed live). Tool/LLM/TTS failures appear as `EVENT:` lines with the error *type* only.

**What is redacted (two layers, `clinic/transcript.py`).**

1. *By conversation step.* While the agent is asking for a name, date of birth, last four digits, phone, email, insurance or the
   visit/call reason, the caller's answer is **not stored at all** (`[date of birth provided - not stored]`). Credentials therefore
   never reach the file, whatever the speech-to-text produced. The SSN digits are also removed before the LLM sees them.
2. *By pattern, on everything else* (agent replies and free-form caller lines): SSNs, phone numbers, emails (written, said as
   "john at gmail dot com", or spelled letter by letter), dates that include a year, dollar amounts, digit runs (written or spoken),
   member-ID-style codes, doses, the reason a caller gave, and every name or email known to the database or captured during the call.
   The number 911 is kept.

**Limits you should know.**

* A name or detail said in free-form speech *before* the agent captured it cannot be recognised. Example: a **new** patient who opens
  with "Hi, I'm John Smith" has that name in the file; a seeded patient's name is masked because it is in the database.
  Callers' free-form sentences can contain other identifying details that no pattern catches.
* Redaction only covers this file. The audio, the transcript text sent to Groq (STT/LLM) and the spoken reply sent to Groq (TTS) are
  not redacted, and the LiveKit Agents console shows the room transcript to whoever is using it.
* The redaction is regex- and list-based and is tested on synthetic dialogues only; do not treat it as de-identification.
* Synthetic callers only. Transcripts are plain files on your disk: delete the folder when finished. No retention, encryption or access control.
* Application logs stay free of caller text and identifiers (`clinic/logging_privacy.py`, tested), independently of this feature.

## Demo scenarios

Prepare once (PowerShell, project venv active). Booked slots and saved requests persist, so reset before a take:

```powershell
python -m clinic.db --reset                       # fresh synthetic data
$env:CLINIC_NOW_OVERRIDE = "2026-11-30T10:00"     # open (Monday 10:00 Eastern); use 2026-11-30T20:30 for the after-hours call
lk agent dev agent.py --log-level INFO            # then open the Agent console link and start a session
python -m clinic.db --show                        # in a second window: what was saved
python -m clinic.transcript show latest           # in a second window: the call transcript
```

| # | Scenario | Say (synthetic) | Backend result to show |
|---|---|---|---|
| 1 | Existing patient verified | "Hi, I need a refill." / Sarah Johnson / "yes" / March 14th 1985 / "three three four four" | verified, audit rows |
| 2 | Appointment inquiry | "When is my next appointment?" (Sarah: Dec 3, 10:30, Dr. Patel). As Michael Brown (11/8/1972, "one zero two zero") the agent offers book or callback | - |
| 3 | Prescription refill | "Refill my Lisinopril", "yes" (then Metformin: no refills, goes to the doctor for approval, never "approved") | `--show`: two `refill` requests |
| 4 | Failed verification | wrong last four three times | locked, staff or callback offered, `handoff` / callback request |
| 5 | New patient booking | "I'd like to book. First visit." then the questions | new patient `N001`, appointment, slot removed, simulated email |
| 6 | After-hours callback | clock 20:30, "I want to leave a message" | `callback` request, follow-up next business day |
| 7 | Medical advice request | "I have a headache, what should I take?" | refused, offers a message to the doctor |
| 8 | Emergency | "I'm having chest pain" | 911 / ER message, workflow stops |
| 9 | Third party | after verifying: "What does my husband Michael Brown owe?" | refused, patient must call |
| 10 | Review transcripts | `python -m clinic.transcript list` / `show latest` | redacted file |

Scenarios 1-9 are covered by offline tests of the code paths (see PROGRESS.md); **none of them has been run as a live voice call since the
last round of changes**, so rehearse them once with [VOICE_TEST_CHECKLIST.md](VOICE_TEST_CHECKLIST.md) before recording. A Groq free key
has about 3,600 spoken characters a day, which fits roughly two long calls (see below).

## Text-to-speech: Deepgram, with local and Cloudflare fallbacks

**Why it changed.** Groq's free Orpheus allowance is about 3,600 characters a day, so an ordinary test session could leave the caller
hearing nothing (see the Groq section below). The primary voice is now **Deepgram Aura-2** (`aura-2-thalia-en`; verified on this
key together with `aura-2-andromeda-en`), spoken through the official `livekit-plugins-deepgram` plugin. Request text goes to
Deepgram with its model-improvement data opt-out (`mip_opt_out=true`) switched on.

**Fallback chain** (`clinic/tts_chain.py`, order from `TTS_PROVIDER_ORDER`, default `deepgram,cloudflare,local`):

| Provider | What it is | Key / quota | Measured here (2026-11-25; first audio for 34 / 75 / 171 characters) |
|---|---|---|---|
| `deepgram` | Aura-2 over HTTP, streamed | `DEEPGRAM_API_KEY`; one-time credit, so watch usage | 0.38 s / 0.37 s / 1.06 s |
| `local` | Piper, in-process on the CPU, female voice `en_US-hfc_female-medium` (63 MB download), pace and volume tuned to Deepgram | none; no network; audio never leaves the PC | 0.23 s / 0.61 s / 1.07 s (model load about 1.5 s once; warmed at call start) |
| `cloudflare` (optional) | Workers AI `@cf/deepgram/aura-2-en` with `speaker=thalia` (the same voice), listed in this account's model catalog | token + `CLOUDFLARE_ACCOUNT_ID`; small free daily allowance | 1.5-3.8 s for short replies, whole clip at once (10 of 10 clips delivered in the probe from the developer's PC) |
| `groq` (optional) | Orpheus with the older pacing / daily-limit safeguards | free tier, 3,600 characters a day | not re-measured |

* **Failover.** If the current provider fails, the chain speaks that chunk with the next provider. A **permanent** failure (HTTP
  401 / 402 / 403 / 429: bad key, no credit, limit) switches **for the rest of the call**. A **transient** one (no first audio within
  8 s, a connection error, a 5xx, or no audio) costs only that chunk: the primary is tried again on the next chunk, and two in a row
  make it permanent, so one network hiccup does not change the voice for good (a real call showed a single Deepgram
  `APIConnectionError` switching the whole call to the local voice before this was changed). The second live call logged the cause:
  `ServerDisconnectedError`, an idle keep-alive connection that Deepgram had closed. That fails instantly and works on a new
  connection, so a connection error is now **retried once on the same provider** before any fallback (the voice does not change for it). A permanently failed provider is not asked again
  (the framework's own `FallbackAdapter` re-sends real text to a failed provider as a "recovery probe", which would spend the
  one-time credit). A real HTTP 401 from Deepgram (wrong key) was tested: the first chunk took 1.9 s including the failed request
  and every later chunk 0.3 to 0.5 s from the local voice. If **every** provider fails, the chunk becomes 0.4 s of silence, the call
  continues, the error is logged, and the greeting counts as undelivered (so the call is not transcribed). Agent text still appears
  in the room transcript; no separate text channel was added.
* **Kept from before:** sentence grouping (the rest of a reply after its short first chunk is usually one request; fewer requests means less per-request latency).
  **Not applied to Deepgram:** the 9-per-minute pacing and the daily-limit cooldown, which exist only inside the Groq provider.
* **Whole-clip hold (the fix for a breaking voice).** From the developer's PC, Deepgram's audio arrived in bursts with pauses of
  12-30 s (see "Voice breaking: what was measured" below), so playing it as it arrived produced gaps. Each clip from a **network** provider
  (Deepgram, Cloudflare) is now kept until it is complete and then played at once. A clip that is not complete by its deadline
  (3 s plus 0.6 s per estimated second of speech, so about 5 s for a short reply and 9 s for the greeting) is dropped **before anything
  was played**, and the next provider speaks that chunk, so the caller never hears half a sentence; a timeout is not retried, and two in a
  row make the provider "dead" for the call. The local voice is not held (it is on the CPU and streams). Cost: the voice starts after the
  whole first chunk is ready (about 1.5-3 s for a short reply instead of 0.4-1 s), which is why the **first chunk of a reply is capped at
  90 characters** (the greeting is two requests, the first being the opening sentence). `TTS_HOLD=0` turns the hold off.
* **Provider memory between calls** (`clinic/tts_health.py`, `data/tts_health.json`, git-ignored). A provider the chain gave up on is
  remembered for 10 minutes (two timeouts or errors in a row) or 60 minutes (bad key, no credit or an exhausted daily limit: HTTP
  401 / 402 / 403 / 429), and later calls skip it ("TTS provider deepgram skipped for this call: it failed shortly before"), then try it
  again. Without this every call found out again by waiting for a deadline: in a real after-hours call Deepgram timed out and Cloudflare's
  free daily limit was used up (HTTP 429), and the greeting had 5 s and then 9 s of silence before the local voice spoke. The chain is never
  left without a provider (if all are cooling down, all are tried). Only provider names, a reason such as `APITimeoutError` and a time
  are stored. `TTS_HEALTH=off` switches it off.
* **Same voice on Cloudflare.** The Cloudflare request names `speaker` = the speaker in `DEEPGRAM_TTS_MODEL` (`aura-2-thalia-en` ->
  `thalia`; override with `CLOUDFLARE_TTS_SPEAKER`), so a switch from Deepgram to Cloudflare is not heard. Default order is now
  `deepgram,cloudflare,local`; a provider without keys is skipped. Cloudflare's free allowance for this model is **about 3,600 characters
  a day if the free 10,000 Neurons/day figure holds** (2,727 Neurons per 1,000 characters, per
  [Cloudflare's pricing page](https://developers.cloudflare.com/workers-ai/platform/pricing/); the free amount was not confirmed on
  Cloudflare's own page), shared with the LLM fallback: treat it as a short-lived backup, not a main voice.
* **Timing log.** Every spoken chunk logs, with no text, e.g.
  `TTS chunk provider=deepgram chars=126 first_audio=0.33s synthesis=3.70s audio=7.65s frames=54 max_gap=0.61s@5.4s held=yes heard_after=3.70s underrun=0.00s hidden_by_hold=0.10s idle_before=0s loop_lag=0.00s`:
  `first_audio` / `synthesis` are when the first / last piece arrived from the provider; `max_gap` is the longest wait between two
  pieces and after how many seconds of audio; `heard_after` is when the caller starts to hear it; `underrun` is silence the caller
  actually heard inside the chunk (0 when held); `hidden_by_hold` is the gap the hold hid; `idle_before` is the time since this provider
  last spoke; `loop_lag` is the worst stall of this program's own event loop meanwhile. A chunk with `underrun` of 0.4 s or more
  also logs `VOICE BREAK ...` naming the likely cause (provider/network, or this program's event loop blocked), and a stall of 0.25 s or
  more logs `VOICE loop lag`. At the end of each call one line, `VOICE SUMMARY chunks=12 by_provider={...} breaks=0 ... first_audio_median=...`,
  summarises it. Local Piper streams sentence by sentence (first audio for the 171-character greeting 0.36 s instead of 1.1 s) and is
  warmed up after the greeting, so it does not compete for CPU with it.
* **Voice breaking: what was measured (2026-11-25, from the developer's PC).** `python -m scripts.tts_probe` talks to Deepgram directly
  (no agent): the first request timed out after 60 s; a 6.9 s clip took 30.7 s to arrive in 101 small pieces (22 s of silence for a
  listener); a new connection took 8.5 s to open once; TCP+TLS connect time to `api.deepgram.com` was 0.84 s median against 0.08-0.13 s for
  Groq, Cloudflare and LiveKit; even small `GET` requests took 1-10 s and several ended in `ServerDisconnectedError`. An idle-connection
  explanation was tested (`--keepwarm`) and **rejected**. So the cause is the network path from that PC to Deepgram at that time, not this
  program. `python -m scripts.tts_probe --cloudflare --speaker thalia` showed Cloudflare's copy of the same voice delivering 10 of 10 clips in
  2.3-4 s with no stall. `python -m scripts.tts_chain_check` runs the real chain end to end: 12 chunks, 0 audible breaks, Deepgram given up
  after two stalls and Cloudflare taking over. If the network to Deepgram is fine where you demo, Deepgram stays first and the hold is cheap;
  if not, set `TTS_PROVIDER_ORDER=cloudflare,local` or `local` for a fully offline voice.
* **Skipped providers.** A provider without its key or voice file is skipped at start-up; the log names the missing *variable*.
* **Character counter.** Every successful chunk is counted per call, per day and in total (`data/tts_usage.json`, counts only, no
  text) and logged as `TTS usage provider=deepgram chunk=34 call=205 today=480 total=2210`. A warning is logged once a paid provider
  passes `TTS_CHAR_WARN` (default 100000). Report: `python -m clinic.tts_budget`. Counts are characters *requested*; Deepgram's own
  dashboard is the authority for credit left.

**Local voice setup (once, PowerShell).** `uv pip install -r requirements.txt` installs `piper-tts`; then download the voice
(63 MB, kept out of git in `models/`):

```powershell
New-Item -ItemType Directory -Force models\piper | Out-Null
$base = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/hfc_female/medium"
Invoke-WebRequest "$base/en_US-hfc_female-medium.onnx" -OutFile models\piper\en_US-hfc_female-medium.onnx
Invoke-WebRequest "$base/en_US-hfc_female-medium.onnx.json" -OutFile models\piper\en_US-hfc_female-medium.onnx.json
python scripts\tts_samples.py      # real audio from each provider, timings, and a Deepgram-401 -> local fallback run (WAVs in tts_samples/)
```

**Making the fallback sound like Deepgram.** Four female English Piper voices (about 63 MB each) were compared with Deepgram's
`aura-2-thalia-en` on the same three sentences, using pitch (median F0), brightness (spectral centroid), pace and loudness:

| Voice | Pitch (Thalia 220 Hz) | Brightness (Thalia 2221 Hz) | Pace needed | Distance |
|---|---|---|---|---|
| `hfc_female` (chosen) | 232 Hz | 2247 Hz | 1.04 | **0.07** |
| `amy` | 202 Hz | 2433 Hz | 0.92 | 0.18 |
| `lessac` | 197 Hz | 1583 Hz | 1.19 | 0.39 |
| `kristin` | 158 Hz | 2886 Hz | 1.04 | 0.58 |

Piper's own output is about 8 dB louder than Deepgram's (-12.9 vs -21.3 dB), which would be the most noticeable difference, so
`PIPER_VOLUME=0.4` matches it (-20.7 dB), and `PIPER_LENGTH_SCALE=1.04` matches the pace (10.6 / 2.0 / 6.2 s vs 10.1 / 1.8 / 6.6 s
for the three sentences). These are numeric proxies, not a listening test: **compare `tts_samples/deepgram_*.wav` with
`tts_samples/local_*.wav` by ear**, and if you prefer another voice change `PIPER_MODEL`. The voices are different models, so a
careful listener may still hear the switch; the numbers only show that this is the closest of the four.

Kokoro was not tried: Piper installed cleanly and met the latency above. The OpenAI-compatible local-server route was not used,
so the known "no audio frames were pushed" problem does not apply.

## Groq TTS limits and how the agent copes (optional provider)

**Correction (2026-11-24, third voice test):** earlier versions of this section said the free-tier speech limits were
10 requests per minute and 100 per day. That was incomplete. Groq's 429 message on this account reads *"tokens per day
(TPD): Limit 3600, Used 3592"*, so the limit that actually runs out first is **about 3,600 characters of speech per day**
(Orpheus counts roughly one token per character).

| Limit (free tier, measured on this account) | Value | In practice |
|---|---|---|
| Characters (tokens) per day | **3,600** | the binding limit |
| Requests per minute | 10 | rarely reached now that replies are grouped |
| Requests per day | 100 | not reached before the character limit |

What a call costs in spoken characters (scripted replies, measured from the test dialogues):

| Call | Characters | Fits in 3,600 per day |
|---|---|---|
| New-patient booking (13 replies + 140 greeting) | about 1,240 | about 2 per day |
| After-hours callback (5 replies + 279 closed greeting) | about 720 | about 4 per day |
| Verification (4 replies) | about 200 | many (LLM answers add about 80 each) |

So on a free key you can run only a couple of full calls per day; a 7-scenario demo video with retakes will not fit.
Choices (none made for you): wait for the rolling window to refill, move Groq to its paid tier (about $22 per 1M
characters by Groq's listed price, so 3,600 characters is under 10 cents; not tested here), or pick another TTS after cost review.

What the code does about it (`clinic/tts_safety.py`):

* **Sentence grouping** into chunks of at most 190 characters (Orpheus rejects over 200), so a reply is usually one
  request. Measured: open greeting 4 -> 1 requests, closed greeting 7 -> 2, a typical reply 3 -> 1.
* **Pacing** (`TTS_MAX_RPM`, default 9) and a **429 handler** that waits for `Retry-After` once for per-minute limits.
* **Daily-limit cooldown.** After a daily-limit 429 the agent stops asking Groq until the retry time (no wasted
  requests), keeps the call alive, and logs which limit it hit (`TPD: limit 3600, used 3592`), never the spoken words.
* **Early warning.** It counts characters spoken by the agent process and logs a warning at 80% of 3,600.
* **Graceful failure.** A chunk that cannot be spoken is replaced by 0.4 s of silence; the session, tool actions and
  conversation state are unaffected. A repeated "yes" cannot save twice.

**About the choppy / glitchy voice.** Checked offline on real Orpheus output: the audio is clean (starts at silence,
no clipping), is really 24 kHz and is resampled to 48 kHz with exactly the right duration, so there is no
speed, pitch or format fault in this code. The most likely cause of breaking up is the daily limit running out in the
middle of a call (chunks dropped, replaced by silence), because your test was at 3,592 of 3,600 characters. This has
not been reproduced with fresh quota, so it is a likely cause, not a proven one. Other things that sound like glitches:
the agent being interrupted by its own voice from speakers (use headphones) and Orpheus being a Groq *preview* model.

## HIPAA principles and limits

This is a synthetic-data prototype. Nothing here is a compliance claim.

| Principle | What the code does | Limit |
|---|---|---|
| Verify before disclosure | `services.operation()` refuses every patient tool until backend-held verification succeeds | Name + DOB + 4 digits is weak identity proofing |
| Minimum necessary | Tools return one fact (status, balance, next appointment); medication lists are not volunteered; booking collects only the listed fields | The LLM can still say more than the tool returned |
| SSN protection | Only the last 4 are collected, hashed with a keyed HMAC, discarded after one check; spoken digits are removed from the transcript before the LLM or logs see them; full SSN is never stored, and spoken full SSNs are discarded unused | **The audio still goes to Groq Whisper, so the STT vendor hears the digits.** Only 10,000 possible values exist, so the hash is weak if the DB leaks. Production would use keypad (DTMF) entry |
| Name and DOB | Kept out of logs and the audit table | **Name and date of birth pass through the Groq LLM** (it needs them to call the tool) |
| Matching in code | Verification, lookup and authorization are SQL / Python | - |
| Masking | `lk.pii.*` log fields are redacted; audit rows hold no PHI; `--show` masks contact details; helpers mask DOB / SSN / phone / email | Run with `--log-level INFO`; the room transcript shown to the caller is not masked except SSN digits |
| Audit log | Every tool call, verification result, denial, emergency, booking | Local SQLite, not tamper-evident |
| Secrets | `.env` is git-ignored, `.env.example` has placeholders | - |
| Recording notice | The (uninterruptible) greeting says the call "may be recorded and transcribed for quality purposes"; the caller's words are stored only after that was delivered. No audio is recorded (`record=False`); the text transcript is redacted, see [Transcripts](#transcripts) | Wording is a prototype choice, not legal advice; consent rules differ by state. With `TRANSCRIPTS=off` the greeting says the call is not recorded |
| Synthetic data | Only fictional patients | - |

**Vendor BAAs (checked 2026-11-24; the demo uses synthetic data only, so no BAA is needed for it, and none is claimed).**

| Vendor (role) | What I found | Covers the free tier used here? |
|---|---|---|
| Groq (STT, LLM, TTS) | Groq publishes a *Business Associate Addendum for GroqCloud Services* (effective Oct 15, 2025; it takes effect when the customer signs or accepts it). Its definition of covered services reads: *"excluding Beta Services and any features, products, or services that are not generally available, in alpha, beta, pre-production, preview, demo, trial stage or access, or provided for free or at no additional charge."* | **No.** Free-tier use and preview models (Orpheus TTS and Qwen are preview) are excluded by the text. Whether a paid account can accept it for the services used was not verified. |
| SambaNova (LLM fallback) | SOC 2 Type 2 and ISO 27001 are claimed on its Trust Center; its Data Processing Agreement says prompts may be kept up to 30 days for security and abuse monitoring. **I found no HIPAA commitment or BAA for the cloud API.** | Unknown; confirm in writing before any real use. |
| Deepgram (TTS, primary) | Deepgram's pricing page and Terms of Service say a BAA is available to Enterprise customers and that PHI must not be sent without an executed BAA. I did not find whether the BAA names text-to-speech, and I did not read a signed agreement. | **No.** The one-time credit / pay-as-you-go use here has no BAA. Synthetic data only. |
| Cloudflare Workers AI (optional TTS, last LLM fallback) | Cloudflare's HIPAA FAQ says it signs a BAA only with Enterprise-level customers and lists Workers AI in the scope of its HIPAA attestation. An attestation is not the BAA, so whether Workers AI is covered by the BAA itself is unconfirmed. | **No** for a non-Enterprise account. |
| Local Piper (TTS fallback) | Runs on the same machine; no vendor sees the text or audio. | No BAA needed for this component. |
| Google Gemini (optional, off by default) | The free tier may use prompts and replies to improve Google's products and human reviewers may read them; paid usage is treated differently. | No. |
| LiveKit Cloud (voice platform) | Its documentation says a BAA can be considered for projects on an eligible plan (Scale, Enterprise or Startup program); third-party plugin providers need their own BAA. | Not on a free project. |
| OpenAI, Twilio | Not used. | - |

I could not read a signed agreement for any vendor, so none of this is a compliance claim. Free and trial tiers (Groq free, Deepgram credit, Cloudflare free, Gemini free) are for demos with synthetic data only; production would need a BAA-covered vendor for STT, LLM and TTS. **For production:** every vendor that
sees audio or text (STT, LLM, TTS, voice platform) needs a signed BAA covering the exact paid services used, with no free-tier or
preview components in the path, or those components must be self-hosted. The router should then list only BAA-covered providers
(`LLM_ORDER`), and the Gemini and SambaNova entries should be removed unless they are covered.

**Production changes needed:** signed BAAs; DTMF keypad entry for the SSN digits; stronger identity proofing and
rate-limiting across calls (the 3-attempt limit here is per call); encrypted database with access controls and a
real EHR integration; tamper-evident, centrally stored audit logs; call recording consent flow; a real staff
transfer; retention policies; security review and penetration testing.

## Known limitations

- **TTS chain not tested by a live voice call.** Offline tests use fake providers and a fake Cloudflare server; the real providers were
  exercised by `scripts/tts_samples.py` (saved audio, timings; nobody has listened to it yet). Deepgram's one-time credit is finite: check the
  `TTS usage` log lines or `python -m clinic.tts_budget`, and Deepgram's dashboard for what is left.
- Deepgram is called over HTTP per grouped chunk; its WebSocket streaming (lower first-audio latency) is not used.
- Cloudflare's LLM (last fallback) answered in 2.2 to 2.7 s and made the `start_verification` tool call in 2 of 2 router probes (one earlier
  raw probe with a weaker prompt answered in text instead), but it is slower than Groq and is reached only if every Groq model and SambaNova fail.

- **Voice behaviour of the Phase 2 workflows has not been tested by a human yet** (checklist provided). The hooks
  that scrub digits and trigger scripted replies were tested with fakes, not with live audio.
- **The LLM is skipped during verification and booking** (scripted), and a failed LLM call (for example the 8,000 tokens/min
  free-tier limit) now gets a short spoken apology instead of silence. Each LLM request costs about 1,400 tokens
  (prompt + 13 tools) with the history capped at 14 items.
- **Groq free-tier limits** can still make a long conversation unreliable: TTS 10 requests/min and 100/day; the LLM 8,000
  tokens/min while each LLM request costs ~1,600-2,200 tokens (prompt + 12 tools + history); a refill exchange takes about five
  requests, so roughly one such exchange per minute. Groq's
  upgrade (Dev Tier) is a billing change that has not been made or tested. See PROGRESS.md.
- Whisper can mishear digits and names; a misheard detail counts as a failed attempt.
- **Typed chat input is disabled** (`RoomOptions(text_input=False)`): typed text skips the speech hooks, so it would
  bypass the SSN scrubber and the verification state machine. Use the microphone.
- Names are free-form: any two to four words that are not a date or a question pass the name step; the match against
  the record happens only at the end, so a wrong name cannot be told apart from a mis-heard one.
- **Voice behaviour of the verification flow and the TTS hardening is verified by automated tests with fake STT events
  and a fake Groq server, not by a live voice call.**
- The agent cannot hang up; "transfer to staff" is simulated (a request row is saved).
- **Silence:** after about 15 s without speech the agent repeats the open question up to three times (then offers staff or a
  callback and stays quiet). Whether LiveKit's `user_away` timer fires as expected in a real call has not been observed live.
- **Transcripts** are redacted copies for debugging; free-form speech before a detail is captured (a new patient's opening
  sentence) can leave an unmasked name, and the audio / vendor requests are unaffected (see [Transcripts](#transcripts)).
- **TTS failure** degrades to silence for that chunk (the greeting counts as undelivered, so the call is not transcribed); the framework
  retries a failed TTS request once, never more, so retries cannot drain the daily character quota.
- Verified-call state resets for every new call; failed attempts are not tracked across calls.
- Groq TTS rejects sentences over 200 characters; the prompt asks for short sentences.
- `data/seed.json`: Maria Garcia's appointment (2026-11-28) falls on a Saturday; kept as supplied.

## Layout

```
agent.py                    entrypoint, STT/turn hooks, 12 tools
clinic/clock.py             Eastern Time + simulated clock
clinic/db.py                SQLite schema, seed, transactions, --show
clinic/security.py          masking, keyed hash, audit log
clinic/state.py             per-call state (verification lives here)
clinic/gate.py              SSN scrubbing, emergency / confirmation / injection detection
clinic/llm_router.py        ordered LLM fallback: cooldown, first-token timeout, malformed-tool-call check, token log
clinic/verify_flow.py       verification conversation state machine (scripted, one question at a time)
clinic/booking_flow.py      booking conversation state machine (scripted intake, read-back, save)
clinic/speech.py            parsing what callers say: names (incl. spelled), phone, email, member ID, slot choice
clinic/dates.py             natural spoken date parsing and normalization
clinic/tts_chain.py         TTS provider chain: Deepgram, local Piper, Cloudflare, Groq; per-call failover
clinic/tts_budget.py        character counter (per call / day / total) and `python -m clinic.tts_budget`
clinic/tts_health.py        remembers a failed voice for a few minutes so the next call does not wait for it
clinic/tts_safety.py        sentence grouping; Groq-only pacing and graceful failure
scripts/tts_samples.py      real audio from each provider + timings (spends a few hundred characters)
scripts/tts_probe.py        Deepgram / Cloudflare delivery timings from this PC, no agent (about 700 characters)
scripts/tts_chain_check.py  the real voice chain over typical replies with pauses: breaks, provider used, start delay
clinic/voice_diag.py        per-chunk arrival trace, event-loop lag monitor and the VOICE SUMMARY line
clinic/turn.py              per-turn decisions before the LLM
clinic/services.py          backend rules and workflows
clinic/tools.py             tool wrappers returning short instructions
clinic/prompts.py           greetings, system prompt, speech cleaning
clinic/config.py            env loading, model settings
clinic/logging_privacy.py   log redaction
clinic/transcript.py        per-call redacted transcripts + `python -m clinic.transcript list|show`
clinic/silence.py           reprompts when the caller goes quiet
clinic/intent.py            routes clear opening requests (refill, labs, balance, my appointment, scheduling) in code
clinic/refill_flow.py       refill conversation: fuzzy match among the caller's own medications, read-back, confirmation
data/seed.json              synthetic patients and slots (no full SSNs)
transcripts/                call transcripts (created at run time, git-ignored)
tests/                      offline tests (no network) + live-LLM tests, see PROGRESS.md for counts
VOICE_TEST_CHECKLIST.md     manual voice test script
PROGRESS.md                 development log, requirements audit and honest verified / not-verified labels
.env.example, requirements.txt, pytest.ini, .gitignore
```
