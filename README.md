# Sunrise Family Clinic - AI Voice Receptionist

A real-time voice agent that answers patient calls for a **fictional** clinic: it verifies identity, handles refill requests, appointment
lookups, lab status, billing and new / existing patient booking, takes messages after hours, and refuses emergencies, medical advice,
third-party requests and prompt injection. All patient data is **synthetic**. This is a prototype: it is **not certified HIPAA compliant**
and no vendor BAA has been verified (see [HIPAA](#hipaa-measures-and-vendor-baas)).

Voice call in a browser through LiveKit (no phone number). Twilio is not implemented. 706 offline tests pass; the developer's live voice
calls completed booking, balance, lab and after-hours flows. More detail: [docs/TECHNICAL_NOTES.md](docs/TECHNICAL_NOTES.md),
[docs/PROGRESS.md](docs/PROGRESS.md) (development log and requirements audit), [docs/VOICE_TEST_CHECKLIST.md](docs/VOICE_TEST_CHECKLIST.md).

## Stack and why

| Part | Choice | Why |
|---|---|---|
| Voice transport / framework | LiveKit Agents 1.8 + LiveKit Cloud | Browser microphone call with no phone number; supplies audio transport, voice-activity detection and turn handling, so the project code is the clinic logic. A Python worker, so it can later sit behind a SIP / Twilio number |
| Speech-to-text | Groq Whisper `whisper-large-v3-turbo` | Fast, free tier |
| LLM + tools | Groq `openai/gpt-oss-120b`, with automatic fallback to other models (router) | Tool calling, fast, free tier; a router because a free tier can run out mid-call |
| Text-to-speech | Deepgram Aura-2 `aura-2-thalia-en`, then Cloudflare (same voice), then a local Piper voice | Natural voice on a free credit; the local voice has no network or quota, so a call can always finish |
| Database | SQLite | No server, standard library, enough for patients, appointments, slots, requests, callbacks and the audit log |
| Language | Python 3.11 | |

The safety-critical parts are **code, not prompts**: verification, authorization, after-hours rules, emergency detection, scheduling and
the booking / callback / refill conversations are Python state machines that run before the LLM and skip it. The LLM only handles the
free-form remainder (greetings, off-topic requests, refusals) and calls a small set of tools that never accept a patient id, a verified
flag or SSN digits.

```
mic -> LiveKit -> Groq STT -> [code: scrub SSN digits, emergency check, verification / booking / refill / callback state machines]
                           -> Groq LLM <-> tools -> backend services (clinic/services.py) -> SQLite
                           -> TTS chain (Deepgram -> Cloudflare -> local Piper) -> speaker
```

## Setup (Windows PowerShell)

```powershell
cd "D:\EZ MD task\Voice AI Agent"
uv venv .venv --python 3.11              # or: py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt       # or: pip install -r requirements.txt
Copy-Item .env.example .env              # then edit .env: keys stay in .env, which is git-ignored
```

Needed in `.env`: `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` (a free LiveKit Cloud project), `GROQ_API_KEY` (free), and
`SSN_HASH_KEY` (any long random string). Optional: `DEEPGRAM_API_KEY` for the main voice, `CLOUDFLARE_API_TOKEN` +
`CLOUDFLARE_ACCOUNT_ID`, `SAMBANOVA_API_KEY`. Every setting is documented in `.env.example`. Install the LiveKit CLI (`lk`) and run
`lk cloud auth` once.

### Local voice (optional, recommended as the last fallback)

The local Piper voice (63 MB, kept out of git) needs no key, quota or network:

```powershell
New-Item -ItemType Directory -Force models\piper | Out-Null
$base = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/hfc_female/medium"
Invoke-WebRequest "$base/en_US-hfc_female-medium.onnx" -OutFile models\piper\en_US-hfc_female-medium.onnx
Invoke-WebRequest "$base/en_US-hfc_female-medium.onnx.json" -OutFile models\piper\en_US-hfc_female-medium.onnx.json
```

Use only this voice with `$env:TTS_PROVIDER_ORDER = "local"`. Without any TTS key or voice file the agent cannot speak.

## Run

```powershell
.\.venv\Scripts\Activate.ps1
python -m clinic.db --reset                       # fresh synthetic data (booked slots come back)
$env:CLINIC_NOW_OVERRIDE = "2026-11-30T10:00"     # Monday 10:00 Eastern = open; use 2026-11-30T20:30 for after hours
lk agent dev agent.py --log-level INFO
```

Open the **Agent console** link it prints, click **Start a session**, allow the microphone. Without the override the real Eastern time
is used. `python -m clinic.db --show` prints saved requests, callbacks, appointments, simulated emails and the audit log.
Test callers: Sarah Johnson (1985-03-14, last four 3344), Michael Brown (1972-11-08, 1020), Maria Garcia (1990-07-22, 9081).

```powershell
python -m pytest -q -k "not live"      # offline tests: no network, about a minute; tests never touch data/clinic.db
```

## How identity verification works

* Existing patients give **full name, date of birth and the last four digits of their SSN**. Never the full SSN: it is not asked for,
  repeated or stored.
* It is a scripted conversation, one question per turn. The name is read back for confirmation, dates are accepted in natural speech, and
  the spoken digits are **removed from the transcript before the LLM or the logs see them**.
* The match happens in backend code (`clinic/services.py`): name, date of birth and last four must all match **the same** patient record;
  the last four are checked against a keyed hash and discarded after one check. The LLM cannot read or set the verified state.
* A mismatch always gets the same reply ("I'm sorry, I couldn't verify those details") and never says which detail was wrong. **Three
  failures lock verification for the call**; the agent then offers a (simulated) staff transfer or a callback message.
* Every patient tool refuses until verification has succeeded, so this does not depend on the prompt.
* After verification: next appointment (if none, offers booking or a callback), **lab status only** (never results), **balance due**
  only, and **refills** (the caller names the medication, which is matched among their own medications; the agent reads back
  medication and pharmacy, saves after a clear yes, and says a request goes to the doctor when no refills remain; it never says a refill
  is approved).

## Appointment booking flow

New patients (and existing ones, skipping details already on file) go through a scripted conversation, one question at a time:

`first visit? -> name (read back) -> date of birth -> phone -> email -> insurance provider + member ID, or self-pay -> reason (a few words) -> preferred doctor / day -> choose one of 2-3 real open slots -> read-back -> saved`

* Slots come from the database; the agent never invents one. The read-back says name, date and time, doctor and reason, and **spells the
  email**. Only after a clear yes does one transaction delete the slot, create the patient (`N001`...), the appointment, the request and a
  simulated confirmation email (saved in the database, never sent), and the agent gives the arrival instructions
  ("arrive 15 minutes early, bring a photo ID and your insurance card").
* Details are parsed by deterministic code (`clinic/speech.py`); an unrecognised insurer is read back for a yes, "book 1 for 10am" books
  the time that was said, and a "new patient" who matches an existing record is sent to staff without saying why.
* A caller who just registered as a new patient and asks about labs, balance or refills is told nothing is on file yet; they are never
  re-verified (a new patient has no stored last four digits).
* Every refill, booking, callback and handoff is saved in `requests` with patient or new-patient id, type, timestamp, short summary and
  status; messages go to `callback_messages`.

## After-hours calls

* Hours are Monday to Friday, 9 AM to 5 PM **Eastern Time** (daylight saving handled; weekends closed). The clock is checked on **every**
  backend call, so a call that crosses 5 PM loses patient access mid-call. `CLINIC_NOW_OVERRIDE` simulates any moment for testing.
* The greeting says the clinic is closed, gives the hours and says to call 911 in an emergency. **No patient information is shared, even
  with a verified caller**, and nothing is booked: the backend refuses, and a request for labs, balance, a refill or an appointment gets a
  fixed closed answer with no lookup.
* The agent takes a message (name, callback number, a few words of reason, no health details), reads it back, saves it after a yes and
  closes: "...our team will get back to you on the next business day. If this is an emergency, please hang up and call 911. Have a good
  night." Follow-up date is the next business day (Friday evening and weekends roll to Monday).

## Guardrails

| Guardrail | How |
|---|---|
| Emergencies | Detected in code **before** the LLM (chest pain, trouble breathing, suicidal thoughts, "emergency"...): scripted 911 / nearest-ER message (988 for suicide), all tools locked for the rest of the call, no normal flow |
| No medical advice | Prompt rules; the agent offers to send a message to the doctor; no tool returns anything to interpret |
| No third-party access | Phrases about a spouse, parent or friend are flagged; patient tools refuse for that turn; tools only ever act on the caller verified in this call |
| Prompt injection | Flagged and audited; the model is told not to comply; the backend rules apply whatever the model says. These last three rely partly on the LLM following the prompt, and the backend limits the damage |
| No made-up information | Tools are the only source of patient, appointment and slot data; unknown items get an offer of a staff callback |
| Stay on topic | Unrelated requests are declined |
| Human handoff | Offered when the caller asks for a person, is upset or confused, or cannot verify (simulated: a `handoff` request is saved; there is no real transfer) |

## HIPAA measures and vendor BAAs

| Principle | What the code does | Limit |
|---|---|---|
| Verify before disclosure | `services.operation()` refuses every patient tool until backend-held verification succeeds | Name + date of birth + 4 digits is weak identity proofing |
| Minimum necessary | Tools return one fact (status, balance, next appointment); medication lists are never volunteered; booking collects only the listed fields | The LLM could still say more than a tool returned |
| SSN protection | Only the last 4 are collected, hashed with a keyed HMAC, checked in code and discarded; digits are removed before the LLM and logs see them | **The audio still goes to the STT vendor**, which hears the digits. Production would use keypad (DTMF) entry |
| Matching in code | Lookup, verification and authorization are SQL / Python; the database is never put in a prompt | |
| Masking | Logs and audit rows hold no names, DOB, SSN or free text; helpers mask DOB / SSN / phone / email (`****-**-14`, `***-**-****`) | Run with `--log-level INFO` |
| Audit log | Every tool call, verification result, denial, emergency and booking: who / which patient id / action / outcome / time | Local SQLite, not tamper-evident |
| Secrets | `.env` is git-ignored; `.env.example` has placeholders only | |
| Recording notice | The greeting says the call "may be recorded and transcribed for quality purposes". No audio is recorded | Wording is a prototype choice, not legal advice |
| Synthetic data | Only fictional patients (`data/seed.json`) | |

**Transcripts.** Each call is saved as a redacted text file in `transcripts/` (git-ignored) **only after the recording notice was
delivered**: answers to the name, date of birth, last four, phone, email, insurance and reason questions are not stored; SSNs, phones,
emails, dates with a year, amounts, ids and known names are masked elsewhere. `TRANSCRIPTS=off` disables them (the greeting then says the
call is not recorded). `python -m clinic.transcript list` shows them.

**Vendor BAAs (checked 2026-11-24 / 25; I did not read a signed agreement, so none of this is a compliance claim).** The demo uses
synthetic data only, so none is needed for it.

| Vendor (role) | What I found | Covers the free tier used here? |
|---|---|---|
| Groq (STT, LLM) | Publishes a Business Associate Addendum for GroqCloud that excludes free-tier use and preview / beta models | **No** |
| Deepgram (TTS) | A BAA is available to Enterprise customers; PHI must not be sent without one | **No** |
| Cloudflare Workers AI (TTS, last LLM fallback) | Signs a BAA only with Enterprise customers; whether Workers AI is covered is unconfirmed | **No** |
| SambaNova (LLM fallback) | SOC 2 / ISO 27001 claimed; I found no HIPAA commitment or BAA | Unknown |
| LiveKit Cloud (voice platform) | A BAA can be considered on an eligible plan (Scale / Enterprise) | Not on a free project |
| Local Piper (TTS fallback) | Runs on the same machine; no vendor sees the text | No BAA needed |

**For production:** a signed BAA with every vendor that sees audio or text (STT, LLM, TTS, voice platform) covering the exact paid services
used, with no free-tier or preview component in the path (or self-host them); keypad entry for the SSN digits; stronger identity proofing
and rate limiting across calls (the 3-attempt limit here is per call); an encrypted database with access control and a real EHR
integration; tamper-evident central audit logs; a consent flow for recording; real staff transfer; retention policies; a security review.

## Demo scenarios

| # | Scenario | How |
|---|---|---|
| 1 | Verification, then a refill | "I need a refill" -> Sarah Johnson, March 14 1985, 3344 -> "Lisinopril" (Metformin has 0 refills: goes to the doctor) |
| 2 | Failed verification | Wrong date of birth or digits three times -> staff / callback offer |
| 3 | New patient booking | "I'd like to schedule an appointment" -> "first visit" -> answer the questions -> pick a slot -> yes |
| 4 | After-hours message | Clock `2026-11-30T20:30` -> "I want to leave a message" -> name, number, reason -> yes |
| 5 | Medical advice | "I have a headache, what should I take?" -> no advice, offers a message to the doctor |
| 6 | Emergency | "I'm having chest pain" -> 911 / nearest ER, nothing else continues |
| 7 | Third party | "What does my husband owe?" -> declined |

## Known limitations (free API tiers)

Every external service here runs on a free tier, and the limits caused real problems during development:

* **Groq text-to-speech (Orpheus):** about 3,600 characters a day (and 10 requests a minute), with its terms to be accepted per account.
  Calls went silent mid-sentence once it ran out, so it was replaced as the main voice.
* **Groq LLM:** a cap of about 8,000 tokens a minute per model. Prompts, tools and history are kept small and the LLM is skipped for the
  scripted flows; a router falls back to other models when one is limited.
* **Deepgram:** a one-time credit, and from the developer's PC its network path timed out or stalled for many seconds on some requests.
* **Cloudflare Workers AI:** a small free daily allowance (about 3,600 characters if the free 10,000 Neurons a day holds; unconfirmed),
  after which it answers HTTP 429.
* **What the agent does about it:** each voice clip is held until complete, a clip that misses its deadline goes to the next voice, a
  failed voice is skipped by later calls for a few minutes, and the local voice (no network, no quota) always finishes the call. The voice
  can therefore change mid-call. For a stall-free demo use `TTS_PROVIDER_ORDER=local`. Free tiers are for demos with synthetic data only.

## Layout

```
agent.py                entry point: LiveKit session, speech hooks, tools
clinic/                 backend: services.py (all rules), verify_flow / booking_flow / callback_flow / refill_flow, intent.py,
                        gate.py (SSN scrub, emergency, injection), speech.py, dates.py, db.py, security.py, clock.py,
                        llm_router.py, tts_chain.py (TTS fallback chain), transcript.py, prompts.py, tools.py
data/seed.json          synthetic patients, medications and slots
scripts/                TTS measurement tools (tts_samples, tts_probe, tts_chain_check)
tests/                  offline tests (+ live-LLM tests, deselected with -k "not live")
docs/                   TECHNICAL_NOTES.md, PROGRESS.md, VOICE_TEST_CHECKLIST.md
```
