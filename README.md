# VaaniTriage

Multilingual IT incident desk for the Great Indian AI Internship Challenge. A person records a technical problem, Gnani Prisma v3.0 transcribes it, the desk proposes a category, severity, and urgency, and Gnani Timbre v2.5 reads the confirmation aloud.

Speech-to-text and text-to-speech are live calls to `https://api.vachana.ai`. Category, severity, urgency, duplicate matching, and troubleshooting steps run locally on the transcript. Demonstration rows in the dashboard are labeled Sample and were not produced by Gnani.

## Live Demo

**Try VaaniTriage here:** [https://vaani-triage-ai.streamlit.app/](https://vaani-triage-ai.streamlit.app/)

The application is publicly accessible. No installation or local setup is required to try the live demo.

## What the desk does

- Record or upload a clip in a supported Indian language, or type the report.
- Transcribe with Prisma v3.0 (`POST /stt/v3`, model `prisma-v3.0`), including automatic language detection.
- Propose category, severity, urgency, affected service, and error codes.
- Ask follow-up questions when the service, impact, error text, or start time is missing.
- Compare the report with stored incidents using TF-IDF cosine similarity.
- Show troubleshooting checks, and read them with Timbre v2.5.
- Save the incident only after review. Timbre then speaks the incident number.
- Edit the record, change its status, and review the history.
- Dashboard and analytics for counts, severity, urgency, category, status, and daily volume.

## Project layout

```text
app.py                  Streamlit entry point
vaani/gnani_client.py   Prisma v3.0 and Timbre v2.5 client
vaani/interface.py      Pages
vaani/nlp.py            Category, severity, urgency, duplicates, suggestions
vaani/database.py       SQLite storage
vaani/languages.py      Language codes and Timbre v2.5 voices
vaani/audio_utils.py    Upload checks and WebM-to-WAV conversion
scripts/verify_gnani.py Live API check
tests/test_core.py      Local tests (no network)
data/vaani.db           Created on first launch
```

## Requirements

- Python 3.10 or newer
- A Gnani API key from the Vachana dashboard

## Setup

```powershell
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
copy .env.example .env
```

Edit `.env` and set `GNANI_API_KEY`. The application sends that value only in the `X-API-Key-ID` header. `.env` is listed in `.gitignore`. Do not put the key in source code or in the browser.

## Run

```powershell
.venv\Scripts\python -m streamlit run app.py
```

Open the URL Streamlit prints, usually `http://localhost:8501`.

## Database

SQLite is created automatically at `data/vaani.db`. No separate database server is required.

- Set `VAANI_SEED_SAMPLES=0` in `.env` to skip the labeled demonstration rows.
- Set `VAANI_DB_PATH` to store the file somewhere else.
- Delete `data/vaani.db` to start over. Uploaded clips in `data/uploads/` are separate.

Tables: `incidents` and `incident_events`.

## Check the Gnani APIs

```powershell
.venv\Scripts\python scripts\verify_gnani.py
```

The script builds a short English speech clip, posts it to Prisma v3.0, then posts a sentence to Timbre v2.5. It prints the HTTP result. It does not replace a failed call with a fake transcript or fake audio.

Local tests, which never call Gnani:

```powershell
.venv\Scripts\python -m pytest tests -q
```

## Record a public demo

1. Put a working key in `.env` and start the app.
2. On the dashboard, point out the Sample badge and the four-step flow.
3. Click **Verify Timbre connection** and let the English confirmation play.
4. Open **Report an incident**. Record a problem of under 30 seconds, or upload WAV, MP3, OGG, FLAC, AAC, or M4A. A browser WebM recording is converted to WAV before Prisma sees it.
5. Leave the language on Automatic detection. Prisma returns `detected_language_code`.
6. Review the transcript, category, severity, and urgency. Answer a follow-up question and click **Reassess**.
7. Show a duplicate match against a sample, then save a new incident.
8. On the details page, play the Timbre confirmation and **Read this incident aloud**.
9. Edit one field, change the status, then open **Incident history** and **Analytics**.
10. Say clearly which rows are samples and which row has a Prisma request id.

Keep each recording under 30 seconds. That is the Prisma REST limit.

## API behavior that is not faked

- Without `GNANI_API_KEY`, transcription and speech buttons show the missing-key error. Typed analysis still works and is labeled Typed.
- An empty Prisma transcript means the service heard no speech. The desk does not invent words.
- Troubleshooting text is a local playbook. Timbre only reads that text. It is not a confirmed fix.
