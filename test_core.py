"""Local checks for classification, storage, and the Gnani request shape.

These tests do not call api.vachana.ai. Live Prisma and Timbre checks are in
scripts/verify_gnani.py and only count as success when that script gets a real
HTTP success.
"""

from __future__ import annotations

import os
import tempfile
import wave
from pathlib import Path

os.environ["VAANI_DB_PATH"] = str(Path(tempfile.mkdtemp()) / "vaani-test.db")
os.environ["VAANI_SEED_SAMPLES"] = "0"
os.environ.pop("GNANI_API_KEY", None)

from vaani.audio_utils import AudioValidationError, prepare_audio, write_tone_wav
from vaani.database import create_incident, get_events, get_incident, init_db, list_incidents, update_incident, update_status
from vaani.gnani_client import GnaniAPIError, GnaniClient, parse_error
from vaani.nlp import analyze_incident, find_duplicates


def _wav_bytes(seconds: float = 0.4) -> bytes:
    folder = Path(tempfile.mkdtemp())
    path = folder / "clip.wav"
    write_tone_wav(path, seconds=seconds)
    return path.read_bytes()


def test_critical_email_is_immediate():
    result = analyze_incident(
        "The production email server is down. Nobody in the company can send mail. Outlook shows error 0x80040115.",
        "en-IN",
    )
    assert result["category"] == "Email & Collaboration"
    assert result["severity"] == "Critical"
    assert result["urgency"] == "Immediate"
    assert "0x80040115" in result["error_messages"]
    assert result["affected_service"] == "Outlook"


def test_hindi_email_and_follow_up_language():
    hindi = analyze_incident("प्रोडक्शन ईमेल सर्वर डाउन है। कंपनी के सभी यूजर मेल नहीं भेज पा रहे हैं।", "hi-IN")
    assert hindi["category"] == "Email & Collaboration"
    assert hindi["severity"] == "Critical"
    short = analyze_incident("प्रिंटर", "hi-IN")
    assert short["follow_ups"]
    assert short["follow_ups"][0]["text"] != short["follow_ups"][0]["text_en"]


def test_printer_jam_is_scheduled():
    result = analyze_incident("My laptop printer is jammed and the page will not come out.", "en-IN")
    assert result["category"] == "Hardware"
    assert result["severity"] == "Low"
    assert result["urgency"] == "Scheduled"


def test_duplicate_match_and_storage_round_trip():
    init_db(seed=False)
    first = analyze_incident(
        "The VPN is down for the entire office and nobody can connect to the corporate network.",
        "en-IN",
    )
    saved = create_incident({**first, "status": "Open", "source": "manual_text", "reporter_name": "Asha"})
    assert saved["reference"] == "VT-0001"
    assert saved["urgency"] == "Urgent"
    matches = find_duplicates(
        "The VPN is down for the entire office and nobody can connect to the corporate network today.",
        list_incidents(),
    )
    assert matches and matches[0]["reference"] == "VT-0001"
    edited = update_incident(saved["id"], {"title": "Office VPN outage", "urgency": "Immediate"})
    assert edited["title"] == "Office VPN outage"
    assert edited["urgency"] == "Immediate"
    assert any(event["event_type"] == "edited" for event in get_events(saved["id"]))
    update_status(saved["id"], "In Progress", "Desk is checking the concentrator.")
    again = get_incident(saved["id"])
    assert again is not None and again["status"] == "In Progress"


def test_audio_limits():
    name, data, mime = prepare_audio("note.wav", _wav_bytes(0.4))
    assert name == "incident.wav"
    assert mime == "audio/wav"
    assert data[:4] == b"RIFF"
    try:
        prepare_audio("empty.wav", b"")
    except AudioValidationError as exc:
        assert "empty" in str(exc).lower()
    else:
        raise AssertionError("empty audio should be rejected")
    try:
        prepare_audio("long.wav", _wav_bytes(31))
    except AudioValidationError as exc:
        assert "30" in str(exc)
    else:
        raise AssertionError("a 31 second clip should be rejected")


def test_client_sends_documented_prisma_and_timbre_fields():
    class Response:
        def __init__(self, status, body=None, content=b"", headers=None):
            self.status_code = status
            self._body = body
            self.content = content
            self.headers = headers or {}
            self.text = ""

        def json(self):
            if self._body is None:
                raise ValueError("not json")
            return self._body

    class Session:
        def __init__(self, response):
            self.response = response
            self.calls = []

        def request(self, method, url, timeout=None, **kwargs):
            self.calls.append((method, url, kwargs))
            return self.response

    stt_body = {
        "success": True,
        "request_id": "req-1",
        "transcript": "VPN down",
        "model": "prisma-v3.0",
        "detected_language_code": "en-IN",
        "processing_time": 0.2,
    }
    stt_session = Session(Response(200, stt_body))
    client = GnaniClient(api_key="test-key", base_url="https://api.vachana.ai", session=stt_session, sleeper=lambda _: None)
    result = client.transcribe(_wav_bytes(), "incident.wav", "audio/wav", "indic-auto", "transcribe")
    assert result["transcript"] == "VPN down"
    method, url, kwargs = stt_session.calls[0]
    assert method == "POST"
    assert url == "https://api.vachana.ai/stt/v3"
    assert kwargs["headers"]["X-API-Key-ID"] == "test-key"
    assert kwargs["data"]["model"] == "prisma-v3.0"
    assert kwargs["data"]["language_code"] == "indic-auto"
    assert kwargs["files"]["audio_file"][0] == "incident.wav"

    wav = b"RIFF" + b"\x00" * 80
    tts_session = Session(Response(200, content=wav, headers={"Content-Type": "audio/wav"}))
    speaker = GnaniClient(api_key="test-key", base_url="https://api.vachana.ai", session=tts_session, sleeper=lambda _: None)
    audio = speaker.synthesize("The incident is saved.", "en-IN")
    assert audio[:4] == b"RIFF"
    _method, tts_url, tts_kwargs = tts_session.calls[0]
    assert tts_url == "https://api.vachana.ai/api/v1/tts/inference"
    assert tts_kwargs["json"]["model"] == "timbre-v2.5"
    assert tts_kwargs["json"]["voice"] == "Kaveri"
    assert tts_kwargs["json"]["language"] == "en-IN"
    assert tts_kwargs["headers"]["X-API-Key-ID"] == "test-key"


def test_missing_key_does_not_invent_audio():
    client = GnaniClient(api_key="", base_url="https://api.vachana.ai", sleeper=lambda _: None)
    try:
        client.synthesize("Hello", "en-IN")
    except GnaniAPIError as exc:
        assert exc.error_type == "MISSING_API_KEY"
    else:
        raise AssertionError("a missing key must not return audio")


def test_parse_tts_error_shape():
    class Response:
        status_code = 400
        text = ""
        headers = {"Content-Type": "application/json"}

        def json(self):
            return {"success": False, "message": "Invalid request: Field required at body -> text", "status_code": 400}

    error = parse_error(Response())
    assert "Field required" in str(error)
    assert error.status_code == 400


def test_short_description_duplicate_detection():
    incidents = [
        {
            "id": 1,
            "reference": "VT-0001",
            "title": "VPN disruption",
            "description": "VPN is down for the entire office.",
            "category": "Network & Connectivity",
            "severity": "High",
            "status": "Open",
            "is_sample": False,
        }
    ]
    # "VPN is down" is 11 chars, previously skipped when threshold was < 12
    matches = find_duplicates("VPN is down", incidents)
    assert len(matches) == 1
    assert matches[0]["reference"] == "VT-0001"


def test_database_migration_backfill():
    import sqlite3
    db_file = Path(tempfile.mkdtemp()) / "migrate_test.db"
    conn = sqlite3.connect(db_file)
    conn.execute(
        """
        CREATE TABLE incidents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            reference TEXT UNIQUE NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            category TEXT NOT NULL,
            severity TEXT NOT NULL,
            status TEXT NOT NULL,
            source TEXT NOT NULL
        )
        """
    )
    conn.execute(
        "INSERT INTO incidents (reference, created_at, updated_at, title, description, category, severity, status, source) "
        "VALUES ('VT-0001', '2026-11-10T00:00:00Z', '2026-11-10T00:00:00Z', 'Test', 'Desc', 'General IT', 'Critical', 'Open', 'manual_text')"
    )
    conn.commit()

    from vaani.database import _migrate
    _migrate(conn)

    row = conn.execute("SELECT urgency, urgency_reason FROM incidents WHERE id = 1").fetchone()
    conn.close()
    assert row[0] == "Immediate"
    assert "Critical" in row[1] or "Respond now" in row[1]


def test_navigation_go_to_and_sidebar_state():
    import streamlit as st
    from vaani.interface import PAGES, go_to

    st.session_state["nav"] = "Dashboard"
    st.session_state.pop("nav_target", None)
    st.session_state.pop("selected_id", None)

    go_to("Incident details", 101)

    # go_to must defer navigation via nav_target and not mutate the widget key directly
    assert st.session_state.get("nav_target") == "Incident details"
    assert st.session_state.get("selected_id") == 101
    assert st.session_state.get("nav") == "Dashboard"

    # Simulate sidebar pre-widget lifecycle execution
    if "nav_target" in st.session_state:
        target = st.session_state.pop("nav_target")
        if target in PAGES:
            st.session_state["nav"] = target

    assert st.session_state["nav"] == "Incident details"
    assert "nav_target" not in st.session_state


def test_multilingual_classification_telugu():
    from vaani.languages import TIMBRE_VOICES, voice_for_language
    text = "ప్రొడక్షన్ డేటాబేస్ పూర్తిగా డౌన్ అయింది. కంపెనీ అంతటా వినియోగదారులందరికీ పని చేయడం లేదు."
    result = analyze_incident(text, "te-IN")
    assert result["category"] == "Database"
    assert result["severity"] == "Critical"
    assert result["urgency"] == "Immediate"
    assert len(result["suggestions"]) >= 2
    # Telugu suggestions should be returned
    assert any("డేటాబేస్" in s or "క్వెరీ" in s or "స్క్రీన్" in s for s in result["suggestions"])
    # Telugu urgency reason
    assert "స్పందించండి" in result["urgency_reason"]
    # Timbre voice check for Telugu
    voice, lang, model = voice_for_language("te-IN")
    assert voice == "Suhana"
    assert lang == "te-IN"
    assert model == "timbre-v2.5"


def test_multilingual_classification_tamil():
    from vaani.languages import voice_for_language
    text = "முழு அலுவலக விபிஎன் டவுன். பல பயனர்கள் இணைக்க முடியவில்லை."
    result = analyze_incident(text, "ta-IN")
    assert result["category"] == "Network & Connectivity"
    assert result["severity"] == "High"
    assert result["urgency"] == "Urgent"
    assert len(result["suggestions"]) >= 2
    # Tamil suggestions should be returned
    assert any("நெட்வொர்க்" in s or "இணைப்பு" in s or "விபிஎன்" in s for s in result["suggestions"])
    # Timbre voice check for Tamil
    voice, lang, model = voice_for_language("ta-IN")
    assert voice == "Asmita"
    assert lang == "ta-IN"
    assert model == "timbre-v2.5"


def test_multilingual_classification_hindi_low():
    text = "प्रिंटर जाम हो गया है और कागज फंस गया है।"
    result = analyze_incident(text, "hi-IN")
    assert result["category"] == "Hardware"
    assert result["severity"] == "Low"
    assert result["urgency"] == "Scheduled"
    # Hindi suggestions should be returned
    assert len(result["suggestions"]) >= 2
    assert any("प्रिंटर" in s or "केबल" in s or "कागज" in s for s in result["suggestions"])
    # Hindi urgency reason
    assert "रूटीन" in result["urgency_reason"] or "अनुरोध" in result["urgency_reason"] or "सामान्य" in result["urgency_reason"] or "शेड्यूल" in result["urgency_reason"] or "Schedule" in result["urgency_reason"]


def test_all_11_timbre_voices_and_support_check():
    from vaani.languages import TIMBRE_VOICES, is_timbre_supported, voice_for_language
    expected = {
        "en-IN": "Kaveri",
        "hi-IN": "Nalini",
        "hi-en": "Poorvi",
        "ta-IN": "Asmita",
        "te-IN": "Suhana",
        "kn-IN": "Saanvi",
        "ml-IN": "Reshma",
        "mr-IN": "Zahira",
        "bn-IN": "Kirra",
        "gu-IN": "Falak",
        "pa-IN": "Mehuli",
    }
    for code, expected_voice in expected.items():
        assert is_timbre_supported(code), f"Expected {code} to be supported by Timbre"
        voice, lang, model = voice_for_language(code)
        assert voice == expected_voice, f"Voice mismatch for {code}: got {voice}, expected {expected_voice}"
        assert lang == code
        assert model == "timbre-v2.5"

    assert not is_timbre_supported("fr-FR")
    assert not is_timbre_supported("es-ES")
    assert not is_timbre_supported("unsupported-lang")


def test_spoken_phrases_and_script_detection():
    from vaani.languages import (
        follow_up_speech,
        incident_summary_speech,
        language_for_speech,
        suggestion_speech,
    )

    # Telugu script detection
    te_text = "ల్యాప్‌టాప్ పనిచేయడం లేదు"
    assert language_for_speech(None, te_text) == "te-IN"

    # Tamil script detection
    ta_text = "மின்னஞ்சல் வேலை செய்யவில்லை"
    assert language_for_speech(None, ta_text) == "ta-IN"

    # Hinglish detection
    hinglish_text = "VPN connect nahi ho raha hai please help"
    assert language_for_speech(None, hinglish_text) == "hi-en"

    # Spoken suggestion check
    spoken_te = suggestion_speech(["స్టెప్ 1", "స్టెప్ 2"], "te-IN")
    assert "స్టెప్" in spoken_te or "పరిశోధన" in spoken_te

    spoken_ta = suggestion_speech(["படி 1", "படி 2"], "ta-IN")
    assert "படி" in spoken_ta or "பரிந்துரைகள்" in spoken_ta

    # Incident summary check
    summary_hi = incident_summary_speech(12, "ईमेल समस्या", "Critical", "Open", "hi-IN")
    assert "घटना क्रमांक" in summary_hi or "गंभीर" in summary_hi



