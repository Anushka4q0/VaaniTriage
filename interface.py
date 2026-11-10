"""Streamlit interface for the incident desk."""

from __future__ import annotations

import html
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from vaani import config
from vaani.analytics import format_timestamp, summarize
from vaani.audio_utils import AudioValidationError, prepare_audio
from vaani.database import (
    SEVERITIES,
    STATUSES,
    add_event,
    create_incident,
    get_events,
    get_incident,
    init_db,
    link_duplicate_report,
    list_incidents,
    update_incident,
    update_status,
)
from vaani.gnani_client import GnaniAPIError, GnaniClient
from vaani.languages import (
    LANGUAGE_NAMES,
    STT_LANGUAGE_OPTIONS,
    confirmation_text,
    follow_up_speech,
    incident_summary_speech,
    is_timbre_supported,
    language_for_speech,
    normalize_language_code,
    suggestion_disclaimer_text,
    suggestion_speech,
)
from vaani.nlp import SUGGESTION_DISCLAIMER, URGENCIES, analyze_incident, find_duplicates, urgency_for_severity

PAGES = [
    "Dashboard",
    "Report an incident",
    "Incident history",
    "Incident details",
    "Analytics",
]
CATEGORIES = [
    "Email & Collaboration",
    "Network & Connectivity",
    "Access & Identity",
    "Business Application",
    "Database",
    "Hardware",
    "Security",
    "Server & Infrastructure",
    "General IT",
]
BADGE_COLORS = {
    "Critical": ("#FDECEC", "#9F1239"),
    "High": ("#FEF4E6", "#9A3412"),
    "Medium": ("#FFF8E6", "#854D0E"),
    "Low": ("#E9F8EF", "#166534"),
    "Immediate": ("#FDECEC", "#9F1239"),
    "Urgent": ("#FEF4E6", "#9A3412"),
    "Today": ("#FFF8E6", "#854D0E"),
    "Scheduled": ("#E9F8EF", "#166534"),
    "Open": ("#E7F0FE", "#1D4ED8"),
    "In Progress": ("#F3E8FF", "#6B21A8"),
    "Waiting on User": ("#F3F4F6", "#374151"),
    "Resolved": ("#E9F8EF", "#166534"),
    "Closed": ("#E5E7EB", "#374151"),
    "Duplicate": ("#F3F4F6", "#374151"),
    "Sample": ("#F3F4F6", "#374151"),
    "Gnani": ("#E6F6F6", "#0F5F5F"),
    "Typed": ("#EEF2FF", "#3730A3"),
}


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1180px;}
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #d7e4e4;
            border-radius: 14px;
            padding: 0.7rem 0.9rem;
        }
        .vaani-hero {
            background: linear-gradient(135deg, #0f3d3e 0%, #14666a 100%);
            color: #f4fffe;
            border-radius: 16px;
            padding: 1.15rem 1.3rem 1rem;
            margin-bottom: 1rem;
        }
        .vaani-hero h1 {color: #ffffff; font-size: 1.7rem; margin: 0 0 0.25rem 0;}
        .vaani-hero p {margin: 0; color: #d7efef; font-size: 0.98rem;}
        .vaani-card {
            background: #ffffff;
            border: 1px solid #d7e4e4;
            border-radius: 14px;
            padding: 0.9rem 1rem;
            margin-bottom: 0.7rem;
        }
        .transcript-box {
            background: #f7fbfb;
            border-left: 4px solid #0f6e6e;
            border-radius: 8px;
            padding: 0.8rem 1rem;
            white-space: pre-wrap;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def badge(label: str) -> str:
    background, color = BADGE_COLORS.get(label, ("#F3F4F6", "#374151"))
    return (
        f"<span style='background:{background};color:{color};border-radius:999px;"
        f"padding:0.15rem 0.55rem;font-size:0.82rem;font-weight:600;'>{label}</span>"
    )


def hero(title: str, subtitle: str) -> None:
    st.markdown(
        f"<div class='vaani-hero'><h1>{title}</h1><p>{subtitle}</p></div>",
        unsafe_allow_html=True,
    )


def go_to(page: str, incident_id: int | None = None) -> None:
    st.session_state["nav_target"] = page
    if incident_id is not None:
        st.session_state["selected_id"] = incident_id


def render_sidebar() -> str:
    st.sidebar.markdown("### VaaniTriage")
    st.sidebar.caption("Multilingual voice desk for IT incidents")
    if "nav_target" in st.session_state:
        target = st.session_state.pop("nav_target")
        if target in PAGES:
            st.session_state["nav"] = target
    if "nav" not in st.session_state:
        st.session_state["nav"] = PAGES[0]
    page = st.sidebar.radio("Go to", PAGES, key="nav", label_visibility="collapsed")
    st.sidebar.divider()
    if config.api_key():
        st.sidebar.markdown("Gnani key: **loaded**")
    else:
        st.sidebar.markdown("Gnani key: **missing**")
    st.sidebar.caption("Speech-to-text uses Prisma v3.0. Spoken replies use Timbre v2.5. The key is read from the environment and is not shown here.")
    return page


def _client() -> GnaniClient:
    return GnaniClient()


def speak(text: str, language_code: str) -> bytes | None:
    norm = normalize_language_code(language_code) or language_code
    if not is_timbre_supported(norm):
        st.warning(f"Spoken audio is not supported by Timbre for language '{language_code}'.")
        return None
    try:
        return _client().synthesize(text, norm)
    except GnaniAPIError as exc:
        st.error(config.redact(str(exc)))
        return None


def page_dashboard() -> None:
    hero("Incident desk", "See what is open, what is critical, and which records came from a live Gnani transcription.")
    incidents = list_incidents()
    stats = summarize(incidents)
    first, second, third, fourth = st.columns(4)
    first.metric("Total incidents", stats["total"])
    second.metric("Open", stats["open"])
    third.metric("Critical, still open", stats["critical_open"])
    fourth.metric("High, still open", stats["high_open"])
    urgent_row = st.columns(4)
    urgency_counts = {row["urgency"]: int(row["incidents"]) for _, row in stats["by_urgency"].iterrows()} if stats["total"] else {}
    for column, label in zip(urgent_row, URGENCIES):
        column.metric(label, urgency_counts.get(label, 0))
    st.caption(
        f"{stats['samples']} demonstration sample(s) are included so the desk is not empty. "
        f"{stats['live']} record(s) were filed in this application. Samples are labeled and were not produced by Gnani."
    )
    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("Severity")
        if stats["total"]:
            chart = stats["by_severity"].set_index("severity")
            st.bar_chart(chart, color="#0F6E6E")
        else:
            st.info("No incidents yet.")
    with right:
        st.subheader("Recent reports")
        if stats["total"]:
            view = stats["recent"][["reference", "title", "severity", "status", "origin", "reported"]].head(6)
            st.dataframe(view, hide_index=True, use_container_width=True)
        else:
            st.info("Report an incident to see it here.")
    st.subheader("How a voice report is handled")
    one, two, three, four = st.columns(4)
    one.markdown("**1. Listen**  \nPrisma v3.0 turns the recording into text and names the language.")
    two.markdown("**2. Understand**  \nThe desk proposes a category, severity, and urgency for you to correct.")
    three.markdown("**3. Check**  \nFollow-up questions and duplicate matches appear before anything is stored.")
    four.markdown("**4. Confirm**  \nTimbre v2.5 reads the saved incident number back to you.")
    st.subheader("Gnani connection")
    st.caption("This plays a short English confirmation from Timbre v2.5. It does not create an incident.")
    if st.button("Verify Timbre connection", type="primary"):
        audio = speak("Vaani Triage is connected to Gnani.", "en-IN")
        if audio:
            st.success("Timbre v2.5 returned audio.")
            st.audio(audio, format="audio/wav")


def _audio_choice(recorded: Any, uploaded: Any) -> Any:
    if recorded is not None and uploaded is not None:
        choice = st.radio("Both a recording and a file are present. Which should be sent?", ["Recording", "Upload"])
        return recorded if choice == "Recording" else uploaded
    return recorded if recorded is not None else uploaded


def _store_audio(filename: str, data: bytes) -> str:
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    path = config.UPLOAD_DIR / f"{uuid.uuid4().hex}_{filename}"
    path.write_bytes(data)
    return str(path)


def _load_analysis_into_form(analysis: dict[str, Any]) -> None:
    st.session_state["field_title"] = analysis["title"]
    st.session_state["field_description"] = analysis["description"]
    st.session_state["field_category"] = analysis["category"] if analysis["category"] in CATEGORIES else "General IT"
    st.session_state["field_service"] = analysis["affected_service"]
    st.session_state["field_impact"] = analysis["user_impact"]
    st.session_state["field_severity"] = analysis["severity"]
    st.session_state["field_urgency"] = analysis["urgency"]
    st.session_state["field_reason"] = analysis["severity_reason"]
    st.session_state["field_details"] = analysis["technical_details"]


def _ensure_form(draft: dict[str, Any]) -> None:
    token = draft.get("token")
    if st.session_state.get("loaded_draft") == token:
        return
    st.session_state["loaded_draft"] = token
    _load_analysis_into_form(draft["analysis"])


def page_report() -> None:
    hero(
        "Report an incident",
        "Record the problem in an Indian language or English. Prisma v3.0 returns the transcript. You review it before anything is saved.",
    )
    if not config.api_key():
        st.warning("Add GNANI_API_KEY to the .env file before transcription or speech will work. Typed analysis still runs locally and is labeled as typed, not as Gnani output.")

    reporter = st.text_input("Your name (optional)", max_chars=80, key="reporter_name")
    language_label = st.selectbox(
        "Language hint for Prisma",
        [label for label, _code in STT_LANGUAGE_OPTIONS],
        help="Automatic detection sends language_code=indic-auto. A chosen language is a hint, not a lock. Prisma still returns the language it heard.",
    )
    language_code = dict(STT_LANGUAGE_OPTIONS)[language_label]
    with st.expander("Audio options"):
        output_format = st.selectbox(
            "Prisma output style",
            ["transcribe", "verbatim", "codemix", "transliterate"],
            help="transcribe keeps digits and error codes in conventional form and is the default for this desk.",
        )
        st.caption("REST transcription accepts up to 30 seconds of WAV, MP3, OGG, FLAC, AAC, or M4A.")

    left, right = st.columns(2)
    with left:
        recorded = st.audio_input("Record a description") if hasattr(st, "audio_input") else None
        if not hasattr(st, "audio_input"):
            st.info("This Streamlit build has no recorder. Upload a file instead.")
    with right:
        uploaded = st.file_uploader(
            "Or upload audio",
            type=["wav", "mp3", "ogg", "flac", "aac", "m4a"],
        )
    chosen = _audio_choice(recorded, uploaded)
    if st.button("Transcribe with Prisma v3.0", type="primary", disabled=chosen is None):
        _transcribe(chosen, language_code, output_format)

    with st.expander("Type a report instead of speaking"):
        st.caption("This path does not call Prisma. The saved record is marked Typed, not Gnani.")
        typed = st.text_area("What is going wrong?", height=120, key="typed_report")
        if st.button("Analyze typed text"):
            _analyze_typed(typed)

    draft = st.session_state.get("draft")
    if not draft:
        return
    _ensure_form(draft)
    _render_draft(draft, reporter)


def _transcribe(chosen: Any, language_code: str, output_format: str) -> None:
    data = chosen.getvalue()
    filename = getattr(chosen, "name", "recording.wav") or "recording.wav"
    try:
        safe_name, audio, mime = prepare_audio(filename, data)
        body = _client().transcribe(audio, safe_name, mime, language_code, output_format)
    except AudioValidationError as exc:
        st.error(str(exc))
        return
    except GnaniAPIError as exc:
        st.error(config.redact(str(exc)))
        return
    transcript = body.get("transcript", "")
    if not transcript.strip():
        st.warning("Prisma returned an empty transcript. No speech was detected in that clip. Record the problem again.")
        with st.expander("Gnani response"):
            st.json(_public_stt_body(body))
        return
    detected = body.get("detected_language_code")
    speech_language = language_for_speech(detected if isinstance(detected, str) else None, transcript)
    analysis = analyze_incident(transcript, speech_language)
    audio_path = _store_audio(safe_name, audio)
    st.session_state["draft"] = {
        "token": body.get("request_id") or str(uuid.uuid4()),
        "source": "gnani_prisma",
        "transcript": transcript,
        "language_code": speech_language,
        "detected_language_code": detected,
        "stt_model": body.get("model"),
        "stt_request_id": body.get("request_id"),
        "stt_processing_time": body.get("processing_time"),
        "truncated": bool(body.get("truncated")),
        "audio_path": audio_path,
        "analysis": analysis,
        "raw": _public_stt_body(body),
    }
    st.session_state["loaded_draft"] = None


def _analyze_typed(text: str) -> None:
    cleaned = " ".join(text.split())
    if len(cleaned) < 8:
        st.warning("Describe the problem in a little more detail.")
        return
    speech_language = language_for_speech(None, cleaned)
    st.session_state["draft"] = {
        "token": "typed-" + uuid.uuid4().hex,
        "source": "manual_text",
        "transcript": None,
        "language_code": speech_language,
        "detected_language_code": None,
        "stt_model": None,
        "stt_request_id": None,
        "stt_processing_time": None,
        "truncated": False,
        "audio_path": None,
        "analysis": analyze_incident(cleaned, speech_language),
        "raw": None,
    }
    st.session_state["loaded_draft"] = None


def _public_stt_body(body: dict[str, Any]) -> dict[str, Any]:
    allowed = (
        "success",
        "request_id",
        "transcript",
        "model",
        "detected_language_code",
        "processing_time",
        "end_to_end_latency",
        "truncated",
    )
    return {key: body.get(key) for key in allowed if key in body}


def _render_draft(draft: dict[str, Any], reporter: str) -> None:
    st.subheader("Review")
    if draft["source"] == "gnani_prisma":
        st.markdown(badge("Gnani") + " " + badge(LANGUAGE_NAMES.get(draft["language_code"], draft["language_code"] or "Language")), unsafe_allow_html=True)
        st.caption("The transcript below is the text returned by Gnani Prisma. It has not been rewritten.")
        st.markdown(
            f"<div class='transcript-box'>{html.escape(draft['transcript'])}</div>",
            unsafe_allow_html=True,
        )
        meta = f"Model: {draft.get('stt_model') or 'unknown'} · Request: {draft.get('stt_request_id') or 'n/a'}"
        if draft.get("stt_processing_time") is not None:
            meta += f" · Processing time: {draft['stt_processing_time']} s"
        if draft.get("detected_language_code"):
            meta += f" · Detected language: {draft['detected_language_code']}"
        st.caption(meta)
        if draft.get("truncated"):
            st.warning("Prisma marked this transcript as truncated. A shorter recording will be more complete.")
        if draft.get("raw"):
            with st.expander("Gnani response fields"):
                st.json(draft["raw"])
    else:
        st.markdown(badge("Typed"), unsafe_allow_html=True)
        st.caption("This text was typed. Prisma was not called.")

    supported_langs = [
        "en-IN", "hi-IN", "hi-en", "ta-IN", "te-IN",
        "kn-IN", "ml-IN", "mr-IN", "bn-IN", "gu-IN", "pa-IN",
    ]
    cur_lang = draft.get("language_code") or "en-IN"
    cur_idx = supported_langs.index(cur_lang) if cur_lang in supported_langs else 0
    chosen_lang = st.selectbox(
        "Active language for triage and audio",
        supported_langs,
        index=cur_idx,
        format_func=lambda code: f"{LANGUAGE_NAMES.get(code, code)} ({code})",
        key=f"lang_choice_{draft['token']}",
    )
    if chosen_lang != draft["language_code"]:
        draft["language_code"] = chosen_lang
        base_text = draft["transcript"] or st.session_state.get("field_description") or draft["analysis"]["description"]
        draft["analysis"] = analyze_incident(base_text, chosen_lang)
        st.session_state["draft"] = draft
        st.session_state["field_reason"] = draft["analysis"]["severity_reason"]
        st.rerun()

    analysis = draft["analysis"]
    st.markdown("**Why this severity**")
    st.write(st.session_state.get("field_reason") or analysis["severity_reason"])

    title = st.text_input("Title", key="field_title", max_chars=120)
    description = st.text_area("Description", key="field_description", height=140)
    col1, col2, col3 = st.columns(3)
    with col1:
        category = st.selectbox("Category", CATEGORIES, key="field_category")
    with col2:
        severity = st.selectbox("Severity", list(SEVERITIES), key="field_severity")
    with col3:
        urgency = st.selectbox("Urgency", list(URGENCIES), key="field_urgency")
    service = st.text_input("Affected service", key="field_service", max_chars=80)
    impact = st.text_input("User impact", key="field_impact", max_chars=120)
    details = st.text_input("Technical details", key="field_details", max_chars=240)

    combined = description
    answers: dict[str, str] = {}
    follow_ups = analysis["follow_ups"]
    if follow_ups:
        st.subheader("Follow-up questions")
        st.caption("These are asked because the report does not yet include that detail. You can save without answering.")
        for question in follow_ups:
            label = question["text_en"]
            if question["text"] != question["text_en"]:
                st.caption(question["text"])
            answers[question["id"]] = st.text_input(label, key=f"follow_{draft['token']}_{question['id']}", max_chars=400)
        if st.button("Reassess with these answers"):
            extra = "\n".join(value.strip() for value in answers.values() if value.strip())
            base = draft["transcript"] or description
            refreshed = analyze_incident(f"{base}\n{extra}".strip(), draft["language_code"])
            draft["analysis"] = refreshed
            draft["token"] = uuid.uuid4().hex
            st.session_state["loaded_draft"] = None
            st.session_state["draft"] = draft
            st.rerun()
        spoken_questions = follow_up_speech([item["id"] for item in follow_ups], draft["language_code"])
        if st.button("Read the questions aloud"):
            audio = speak(spoken_questions, draft["language_code"])
            if audio:
                st.audio(audio, format="audio/wav")
                lang_label = LANGUAGE_NAMES.get(draft["language_code"], draft["language_code"])
                st.caption(f"Spoken by Timbre v2.5 in {lang_label}: {spoken_questions}")

    st.subheader("Possible duplicates")
    st.caption("Similarity is TF-IDF cosine similarity against incidents already on the desk.")
    probe = " ".join(part for part in [description, *answers.values()] if part)
    matches = find_duplicates(probe or description, list_incidents())
    if not matches:
        st.info("No similar incident is above the match threshold. Filing a new record is appropriate.")
    for match in matches:
        sample = " · Sample data" if match["is_sample"] else ""
        st.markdown(
            f"<strong>{html.escape(str(match['reference']))}</strong> · {html.escape(str(match['title']))} · "
            f"{int(match['similarity'] * 100)}% similar{html.escape(sample)}<br>"
            f"{badge(match['severity'])} {badge(match['status'])}",
            unsafe_allow_html=True,
        )
        action_col, open_col = st.columns(2)
        with action_col:
            if st.button(f"Link to {match['reference']} instead of filing", key=f"link_{match['id']}"):
                try:
                    link_duplicate_report(int(match["id"]), draft["transcript"] or description, reporter)
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    st.success(f"The report was attached to {match['reference']}. A new incident was not created.")
                    go_to("Incident details", int(match["id"]))
                    st.rerun()
        with open_col:
            if st.button(f"Open {match['reference']}", key=f"open_{match['id']}"):
                go_to("Incident details", int(match["id"]))
                st.rerun()

    st.subheader("Troubleshooting suggestions")
    st.info(analysis.get("suggestion_disclaimer") or suggestion_disclaimer_text(draft["language_code"]))
    for step in analysis["suggestions"]:
        st.markdown(f"- {step}")
    if st.button("Read the suggestions aloud"):
        spoken_steps = suggestion_speech(analysis["suggestions"], draft["language_code"])
        audio = speak(spoken_steps, draft["language_code"])
        if audio:
            st.audio(audio, format="audio/wav")
            lang_label = LANGUAGE_NAMES.get(draft["language_code"], draft["language_code"])
            st.caption(f"Spoken by Timbre v2.5 in {lang_label}: {spoken_steps}")

    if st.button("Save incident", type="primary"):
        answered = [{"id": key, "answer": value.strip()} for key, value in answers.items() if value.strip()]
        extra_lines = [f"{key}: {value['answer']}" for key, value in ((item['id'], item) for item in answered)]
        full_description = description.strip()
        if extra_lines:
            full_description = full_description + "\n\nAdditional details:\n" + "\n".join(extra_lines)
        recommended_urgency, recommended_reason = urgency_for_severity(severity)
        urgency_reason = recommended_reason
        if urgency != recommended_urgency:
            urgency_reason = (
                f"Urgency set to {urgency} during review. The classifier had suggested {recommended_urgency}."
            )
        payload = {
            "title": title,
            "description": full_description,
            "transcript": draft["transcript"],
            "language_code": draft["language_code"],
            "category": category,
            "affected_service": service,
            "error_messages": analysis["error_messages"],
            "user_impact": impact,
            "technical_details": details,
            "severity": severity,
            "severity_reason": st.session_state.get("field_reason") or analysis["severity_reason"],
            "urgency": urgency,
            "urgency_reason": urgency_reason,
            "status": "Open",
            "source": draft["source"],
            "stt_request_id": draft.get("stt_request_id"),
            "stt_model": draft.get("stt_model"),
            "stt_processing_time": draft.get("stt_processing_time"),
            "reporter_name": reporter,
            "is_sample": False,
            "follow_ups": answered,
            "audio_path": draft.get("audio_path"),
        }
        try:
            incident = create_incident(payload)
        except ValueError as exc:
            st.error(str(exc))
            return
        st.session_state["draft"] = None
        st.session_state["saved_speech"] = {
            "text": confirmation_text(incident["id"], incident["severity"], incident["language_code"] or "en-IN"),
            "language": incident["language_code"] or "en-IN",
            "reference": incident["reference"],
            "id": incident["id"],
        }
        go_to("Incident details", incident["id"])
        st.rerun()


def page_history() -> None:
    hero("Incident history", "Search the desk and open a record. Sample rows stay labeled.")
    search = st.text_input("Search title, description, transcript, category, or reference")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        severity = st.selectbox("Severity", ["All", *SEVERITIES])
    with col2:
        urgency = st.selectbox("Urgency", ["All", *URGENCIES])
    with col3:
        status = st.selectbox("Status", ["All", *STATUSES])
    with col4:
        include_samples = st.checkbox("Include demonstration samples", value=True)
    rows = list_incidents(
        search=search,
        severity="" if severity == "All" else severity,
        status="" if status == "All" else status,
        urgency="" if urgency == "All" else urgency,
        include_samples=include_samples,
    )
    if not rows:
        st.info("No incidents match these filters.")
        return
    frame = pd.DataFrame(
        [
            {
                "reference": row["reference"],
                "title": row["title"],
                "category": row["category"],
                "severity": row["severity"],
                "urgency": row.get("urgency") or "",
                "status": row["status"],
                "language": LANGUAGE_NAMES.get(row.get("language_code") or "", row.get("language_code") or "English (India)"),
                "origin": "Sample" if row["is_sample"] else ("Gnani" if row["source"] == "gnani_prisma" else "Typed"),
                "reported": format_timestamp(row["created_at"]),
            }
            for row in rows
        ]
    )
    st.dataframe(frame, hide_index=True, use_container_width=True)
    labels = [f"{row['reference']} — {row['title']}" for row in rows]
    selected = st.selectbox("Open a record", labels)
    if st.button("View details", type="primary"):
        index = labels.index(selected)
        go_to("Incident details", int(rows[index]["id"]))
        st.rerun()


def page_details() -> None:
    hero("Incident details", "Review the transcript, the severity explanation, and the history of the record.")
    pending_speech = st.session_state.get("saved_speech")
    if pending_speech:
        st.success(f"{pending_speech['reference']} was saved.")
        audio = speak(pending_speech["text"], pending_speech["language"])
        if audio:
            st.audio(audio, format="audio/wav")
            st.caption(pending_speech["text"])
        st.session_state["saved_speech"] = None

    incident_id = st.session_state.get("selected_id")
    if not incident_id:
        st.info("Open a record from Incident history.")
        return
    incident = get_incident(int(incident_id))
    if incident is None:
        st.warning("That incident is no longer in the database.")
        return
    origin = "Sample" if incident["is_sample"] else ("Gnani" if incident["source"] == "gnani_prisma" else "Typed")
    st.markdown(
        f"### {html.escape(incident['reference'])} &nbsp; {badge(incident['severity'])} {badge(incident.get('urgency') or 'Today')} {badge(incident['status'])} {badge(origin)}",
        unsafe_allow_html=True,
    )
    st.caption(f"Reported {format_timestamp(incident['created_at'])} · Updated {format_timestamp(incident['updated_at'])}")
    if incident["is_sample"]:
        st.warning("This is demonstration data. It was classified locally and was not transcribed by Gnani.")

    left, right = st.columns(2)
    with left:
        st.markdown(f"**Title**  \n{incident['title']}")
        st.markdown(f"**Category**  \n{incident['category']}")
        st.markdown(f"**Affected service**  \n{incident['affected_service'] or 'Not recorded'}")
        st.markdown(f"**User impact**  \n{incident['user_impact'] or 'Not recorded'}")
    with right:
        st.markdown(f"**Reporter**  \n{incident['reporter_name'] or 'Not recorded'}")
        language = LANGUAGE_NAMES.get(incident.get("language_code") or "", incident.get("language_code") or "Not recorded")
        st.markdown(f"**Language**  \n{language}")
        st.markdown(f"**Urgency**  \n{incident.get('urgency') or 'Today'}")
        st.caption(incident.get("urgency_reason") or "")
        st.markdown(f"**Technical details**  \n{incident['technical_details'] or 'None captured'}")
        errors = incident["error_messages"] or []
        st.markdown("**Error messages**  \n" + (", ".join(errors) if errors else "None captured"))

    st.markdown("**Description**")
    st.write(incident["description"])
    if incident.get("transcript") and incident["source"] == "gnani_prisma":
        st.markdown("**Prisma transcript**")
        st.markdown(
            f"<div class='transcript-box'>{html.escape(incident['transcript'])}</div>",
            unsafe_allow_html=True,
        )
        st.caption(f"Model {incident.get('stt_model') or 'prisma'} · Request {incident.get('stt_request_id') or 'n/a'}")
    elif incident.get("transcript") and incident["is_sample"]:
        st.markdown("**Sample narrative**")
        st.write(incident["transcript"])

    audio_path = incident.get("audio_path")
    if audio_path:
        audio_file = Path(audio_path)
        if audio_file.is_file():
            st.markdown("**Original voice recording**")
            suffix = audio_file.suffix.lower()
            mime = {
                ".wav": "audio/wav",
                ".mp3": "audio/mpeg",
                ".ogg": "audio/ogg",
                ".flac": "audio/flac",
                ".aac": "audio/aac",
                ".m4a": "audio/mp4",
            }.get(suffix, "audio/wav")
            try:
                audio_bytes = audio_file.read_bytes()
                if audio_bytes:
                    st.audio(audio_bytes, format=mime)
                    st.caption(f"Original audio file: {audio_file.name} ({len(audio_bytes):,} bytes)")
                else:
                    st.caption("The recorded audio file is empty.")
            except Exception as exc:
                st.caption(f"Could not load audio recording: {exc}")
        else:
            st.caption(f"Original audio recording ({audio_file.name}) is no longer on disk.")

    st.markdown("**Severity explanation**")
    st.write(incident["severity_reason"])
    language_code = incident.get("language_code") or language_for_speech(None, incident["description"]) or "en-IN"
    st.info(suggestion_disclaimer_text(language_code))
    refreshed = analyze_incident(incident["description"], language_code)
    for step in refreshed["suggestions"]:
        st.markdown(f"- {step}")
    if st.button("Read the suggestions aloud", key="details_read_steps"):
        spoken_steps = suggestion_speech(refreshed["suggestions"], language_code)
        audio = speak(spoken_steps, language_code)
        if audio:
            st.audio(audio, format="audio/wav")
            lang_label = LANGUAGE_NAMES.get(language_code, language_code)
            st.caption(f"Spoken by Timbre v2.5 in {lang_label}: {spoken_steps}")

    language_code = incident.get("language_code") or language_for_speech(None, incident["description"])
    summary = incident_summary_speech(incident["id"], incident["title"], incident["severity"], incident["status"], language_code)
    if st.button("Read this incident aloud"):
        audio = speak(summary, language_code)
        if audio:
            st.audio(audio, format="audio/wav")
            st.caption(summary)

    st.subheader("Edit the record")
    st.caption("Changes are stored in SQLite and written to the history. Sample rows can be edited too, and they stay labeled as samples.")
    with st.form(f"edit_{incident['id']}"):
        edit_title = st.text_input("Title", value=incident["title"], max_chars=120)
        edit_description = st.text_area("Description", value=incident["description"], height=140)
        edit_left, edit_mid, edit_right = st.columns(3)
        with edit_left:
            edit_category = st.selectbox(
                "Category",
                CATEGORIES,
                index=CATEGORIES.index(incident["category"]) if incident["category"] in CATEGORIES else CATEGORIES.index("General IT"),
            )
        with edit_mid:
            sev_list = list(SEVERITIES)
            sev_idx = sev_list.index(incident["severity"]) if incident.get("severity") in sev_list else 0
            edit_severity = st.selectbox("Severity", sev_list, index=sev_idx)
        with edit_right:
            urg_list = list(URGENCIES)
            current_urgency = incident.get("urgency") or urgency_for_severity(incident["severity"])[0]
            urg_idx = urg_list.index(current_urgency) if current_urgency in urg_list else 0
            edit_urgency = st.selectbox("Urgency", urg_list, index=urg_idx)
        edit_service = st.text_input("Affected service", value=incident.get("affected_service") or "", max_chars=80)
        edit_impact = st.text_input("User impact", value=incident.get("user_impact") or "", max_chars=120)
        edit_details = st.text_input("Technical details", value=incident.get("technical_details") or "", max_chars=240)
        save_edit = st.form_submit_button("Save edits", type="primary")
    if save_edit:
        recommended_urgency, recommended_reason = urgency_for_severity(edit_severity)
        urgency_reason = recommended_reason
        if edit_urgency != recommended_urgency:
            urgency_reason = (
                f"Urgency set to {edit_urgency} during review. The classifier had suggested {recommended_urgency}."
            )
        severity_reason = incident["severity_reason"]
        if edit_severity != incident["severity"]:
            severity_reason = f"Severity changed to {edit_severity} by the reviewer. Previous note: {incident['severity_reason']}"
        try:
            update_incident(
                incident["id"],
                {
                    "title": edit_title,
                    "description": edit_description,
                    "category": edit_category,
                    "severity": edit_severity,
                    "severity_reason": severity_reason,
                    "urgency": edit_urgency,
                    "urgency_reason": urgency_reason,
                    "affected_service": edit_service,
                    "user_impact": edit_impact,
                    "technical_details": edit_details,
                },
            )
        except ValueError as exc:
            st.error(str(exc))
        else:
            st.success("Record saved.")
            st.rerun()

    st.subheader("Update status")
    note = st.text_input("Note for the history", max_chars=500, key="status_note")
    status_list = list(STATUSES)
    status_idx = status_list.index(incident["status"]) if incident.get("status") in status_list else 0
    new_status = st.selectbox("Status", status_list, index=status_idx)
    if st.button("Save status"):
        try:
            update_status(incident["id"], new_status, note)
        except ValueError as exc:
            st.error(str(exc))
        else:
            st.success("Status saved.")
            st.rerun()

    st.subheader("History")
    events = get_events(incident["id"])
    if not events:
        st.caption("No history yet.")
    for event in events:
        st.markdown(f"**{format_timestamp(event['created_at'])}** · {event['event_type']}  \n{event['message']}")


def page_analytics() -> None:
    hero("Analytics", "Counts and trends from the incidents stored in SQLite.")
    include_samples = st.checkbox("Include demonstration samples", value=True, key="analytics_samples")
    stats = summarize(list_incidents(include_samples=include_samples))
    first, second, third, fourth = st.columns(4)
    first.metric("Total", stats["total"])
    second.metric("Open", stats["open"])
    third.metric("Resolved or closed", stats["resolved"])
    fourth.metric("Critical + high, open", stats["critical_open"] + stats["high_open"])
    if not stats["total"]:
        st.info("There are no incidents in this view.")
        return
    left, right = st.columns(2)
    with left:
        st.subheader("Severity distribution")
        st.bar_chart(stats["by_severity"].set_index("severity"), color="#0F6E6E")
    with right:
        st.subheader("Status")
        st.bar_chart(stats["by_status"].set_index("status"), color="#1D4ED8")
    st.subheader("Urgency")
    st.bar_chart(stats["by_urgency"].set_index("urgency"), color="#9A3412")
    st.subheader("Categories")
    st.bar_chart(stats["by_category"].set_index("category"), color="#0F6E6E", horizontal=True)
    st.subheader("Reports by day")
    if stats["trend"].empty:
        st.caption("No dated incidents.")
    else:
        st.line_chart(stats["trend"].set_index("day")[["incidents"]], color="#9A3412")
    st.caption("Days are shown in India Standard Time.")


def main() -> None:
    st.set_page_config(page_title="VaaniTriage", page_icon="☎", layout="wide")
    init_db()
    inject_css()
    if "selected_id" not in st.session_state:
        st.session_state["selected_id"] = None
    page = render_sidebar()
    pages = {
        "Dashboard": page_dashboard,
        "Report an incident": page_report,
        "Incident history": page_history,
        "Incident details": page_details,
        "Analytics": page_analytics,
    }
    pages[page]()
