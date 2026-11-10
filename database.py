"""SQLite storage for incidents and their history."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any

from vaani import config
from vaani.languages import language_for_speech
from vaani.nlp import URGENCIES, analyze_incident, urgency_for_severity
from vaani.sample_data import SAMPLES

STATUSES = ("Open", "In Progress", "Waiting on User", "Resolved", "Closed", "Duplicate")
SEVERITIES = ("Critical", "High", "Medium", "Low")
SOURCES = ("gnani_prisma", "manual_text", "sample")


def _connect() -> sqlite3.Connection:
    path = config.db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def init_db(seed: bool | None = None) -> None:
    with _connect() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS incidents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                reference TEXT UNIQUE NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                transcript TEXT,
                language_code TEXT,
                category TEXT NOT NULL,
                affected_service TEXT,
                error_messages TEXT,
                user_impact TEXT,
                technical_details TEXT,
                severity TEXT NOT NULL,
                severity_reason TEXT NOT NULL,
                urgency TEXT NOT NULL DEFAULT 'Today',
                urgency_reason TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL,
                source TEXT NOT NULL,
                stt_request_id TEXT,
                stt_model TEXT,
                stt_processing_time REAL,
                duplicate_of INTEGER,
                reporter_name TEXT,
                is_sample INTEGER NOT NULL DEFAULT 0,
                follow_up_json TEXT,
                audio_path TEXT,
                FOREIGN KEY (duplicate_of) REFERENCES incidents(id)
            );

            CREATE TABLE IF NOT EXISTS incident_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                incident_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                event_type TEXT NOT NULL,
                message TEXT NOT NULL,
                FOREIGN KEY (incident_id) REFERENCES incidents(id)
            );

            CREATE INDEX IF NOT EXISTS idx_incidents_status ON incidents(status);
            CREATE INDEX IF NOT EXISTS idx_incidents_severity ON incidents(severity);
            CREATE INDEX IF NOT EXISTS idx_incidents_created ON incidents(created_at);
            """
        )
        _migrate(connection)
    should_seed = config.seed_samples_enabled() if seed is None else seed
    if should_seed and not list_incidents(include_samples=True):
        seed_samples()


def _migrate(connection: sqlite3.Connection) -> None:
    columns = {row[1] for row in connection.execute("PRAGMA table_info(incidents)")}
    migrated = False
    if "urgency" not in columns:
        connection.execute("ALTER TABLE incidents ADD COLUMN urgency TEXT NOT NULL DEFAULT 'Today'")
        migrated = True
    if "urgency_reason" not in columns:
        connection.execute("ALTER TABLE incidents ADD COLUMN urgency_reason TEXT NOT NULL DEFAULT ''")
        migrated = True
    if not migrated:
        return
    for severity in SEVERITIES:
        label, reason = urgency_for_severity(severity)
        connection.execute(
            """
            UPDATE incidents
            SET urgency = ?, urgency_reason = ?
            WHERE severity = ? AND (urgency IS NULL OR urgency = '' OR urgency = 'Today' OR urgency_reason IS NULL OR urgency_reason = '')
            """,
            (label, reason, severity),
        )


def _row_to_incident(row: sqlite3.Row) -> dict[str, Any]:
    item = dict(row)
    try:
        item["error_messages"] = json.loads(item.get("error_messages") or "[]")
    except json.JSONDecodeError:
        item["error_messages"] = []
    try:
        item["follow_ups"] = json.loads(item.get("follow_up_json") or "[]")
    except json.JSONDecodeError:
        item["follow_ups"] = []
    item["is_sample"] = bool(item.get("is_sample"))
    return item


def list_incidents(
    search: str = "",
    severity: str = "",
    status: str = "",
    urgency: str = "",
    include_samples: bool = True,
) -> list[dict[str, Any]]:
    clauses = ["1 = 1"]
    params: list[Any] = []
    if severity:
        clauses.append("severity = ?")
        params.append(severity)
    if urgency:
        clauses.append("urgency = ?")
        params.append(urgency)
    if status:
        clauses.append("status = ?")
        params.append(status)
    if not include_samples:
        clauses.append("is_sample = 0")
    needle = search.strip()
    if needle:
        escaped = needle.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        like = f"%{escaped}%"
        clauses.append(
            """(
                title LIKE ? ESCAPE '\\' OR description LIKE ? ESCAPE '\\' OR
                IFNULL(transcript, '') LIKE ? ESCAPE '\\' OR reference LIKE ? ESCAPE '\\' OR
                category LIKE ? ESCAPE '\\' OR IFNULL(affected_service, '') LIKE ? ESCAPE '\\'
            )"""
        )
        params.extend([like] * 6)
    query = f"SELECT * FROM incidents WHERE {' AND '.join(clauses)} ORDER BY created_at DESC, id DESC"
    with _connect() as connection:
        rows = connection.execute(query, params).fetchall()
    return [_row_to_incident(row) for row in rows]


def get_incident(incident_id: int) -> dict[str, Any] | None:
    with _connect() as connection:
        row = connection.execute("SELECT * FROM incidents WHERE id = ?", (incident_id,)).fetchone()
    return _row_to_incident(row) if row else None


def get_events(incident_id: int) -> list[dict[str, Any]]:
    with _connect() as connection:
        rows = connection.execute(
            "SELECT * FROM incident_events WHERE incident_id = ? ORDER BY created_at ASC, id ASC",
            (incident_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def add_event(incident_id: int, event_type: str, message: str, when: datetime | None = None) -> None:
    with _connect() as connection:
        connection.execute(
            "INSERT INTO incident_events (incident_id, created_at, event_type, message) VALUES (?, ?, ?, ?)",
            (incident_id, _iso(when or _now()), event_type, message[:2000]),
        )


def _insert(payload: dict[str, Any], when: datetime, event_message: str) -> dict[str, Any]:
    if payload["severity"] not in SEVERITIES:
        raise ValueError("Severity must be Critical, High, Medium, or Low.")
    urgency = payload.get("urgency") or urgency_for_severity(payload["severity"])[0]
    if urgency not in URGENCIES:
        raise ValueError("Urgency must be Immediate, Urgent, Today, or Scheduled.")
    payload = {**payload, "urgency": urgency}
    if payload["status"] not in STATUSES:
        raise ValueError("Unknown incident status.")
    if payload["source"] not in SOURCES:
        raise ValueError("Unknown incident source.")
    title = " ".join(str(payload.get("title") or "").split())
    description = str(payload.get("description") or "").strip()
    if len(title) < 3:
        raise ValueError("A title of at least 3 characters is required.")
    if not description:
        raise ValueError("A description is required.")
    created = _iso(when)
    with _connect() as connection:
        cursor = connection.execute(
            """
            INSERT INTO incidents (
                reference, created_at, updated_at, title, description, transcript, language_code,
                category, affected_service, error_messages, user_impact, technical_details,
                severity, severity_reason, urgency, urgency_reason, status, source, stt_request_id, stt_model,
                stt_processing_time, duplicate_of, reporter_name, is_sample, follow_up_json, audio_path
            ) VALUES (
                'PENDING', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            (
                created,
                created,
                title[:120],
                description,
                payload.get("transcript"),
                payload.get("language_code"),
                payload.get("category") or "General IT",
                payload.get("affected_service") or "",
                json.dumps(payload.get("error_messages") or []),
                payload.get("user_impact") or "",
                payload.get("technical_details") or "",
                payload["severity"],
                payload.get("severity_reason") or "",
                payload["urgency"],
                payload.get("urgency_reason") or urgency_for_severity(payload["severity"])[1],
                payload["status"],
                payload["source"],
                payload.get("stt_request_id"),
                payload.get("stt_model"),
                payload.get("stt_processing_time"),
                payload.get("duplicate_of"),
                (payload.get("reporter_name") or "")[:80],
                1 if payload.get("is_sample") else 0,
                json.dumps(payload.get("follow_ups") or []),
                payload.get("audio_path"),
            ),
        )
        incident_id = int(cursor.lastrowid)
        reference = f"VT-{incident_id:04d}"
        connection.execute("UPDATE incidents SET reference = ? WHERE id = ?", (reference, incident_id))
    add_event(incident_id, "created", event_message, when)
    incident = get_incident(incident_id)
    if incident is None:
        raise RuntimeError("The incident was saved but could not be read back.")
    return incident


def create_incident(payload: dict[str, Any]) -> dict[str, Any]:
    message = "Incident filed from a Gnani Prisma transcription."
    if payload.get("source") == "manual_text":
        message = "Incident filed from typed text. It was not transcribed by Gnani."
    elif payload.get("source") == "sample":
        message = "Demonstration incident loaded locally. It was not transcribed by Gnani."
    return _insert(payload, _now(), message)


def update_status(incident_id: int, status: str, note: str = "") -> dict[str, Any]:
    if status not in STATUSES:
        raise ValueError("Unknown incident status.")
    current = get_incident(incident_id)
    if current is None:
        raise ValueError("Incident not found.")
    if current["status"] == status and not note.strip():
        return current
    now = _iso(_now())
    with _connect() as connection:
        connection.execute(
            "UPDATE incidents SET status = ?, updated_at = ? WHERE id = ?",
            (status, now, incident_id),
        )
    detail = f"Status changed from {current['status']} to {status}."
    if note.strip():
        detail = f"{detail} Note: {note.strip()[:500]}"
    add_event(incident_id, "status", detail)
    updated = get_incident(incident_id)
    if updated is None:
        raise RuntimeError("Incident disappeared after update.")
    return updated


def update_incident(incident_id: int, changes: dict[str, Any]) -> dict[str, Any]:
    current = get_incident(incident_id)
    if current is None:
        raise ValueError("Incident not found.")
    title = " ".join(str(changes.get("title", current["title"]) or "").split())
    description = str(changes.get("description", current["description"]) or "").strip()
    category = str(changes.get("category", current["category"]) or "General IT").strip() or "General IT"
    severity = str(changes.get("severity", current["severity"]))
    urgency = str(changes.get("urgency", current.get("urgency") or "Today"))
    service = str(changes.get("affected_service", current.get("affected_service") or "") or "")[:80]
    impact = str(changes.get("user_impact", current.get("user_impact") or "") or "")[:120]
    details = str(changes.get("technical_details", current.get("technical_details") or "") or "")[:240]
    severity_reason = str(changes.get("severity_reason", current.get("severity_reason") or "") or "")
    urgency_reason = str(changes.get("urgency_reason", current.get("urgency_reason") or "") or "")
    if severity not in SEVERITIES:
        raise ValueError("Severity must be Critical, High, Medium, or Low.")
    if urgency not in URGENCIES:
        raise ValueError("Urgency must be Immediate, Urgent, Today, or Scheduled.")
    if len(title) < 3:
        raise ValueError("A title of at least 3 characters is required.")
    if not description:
        raise ValueError("A description is required.")
    proposed = {
        "title": title[:120],
        "description": description,
        "category": category[:80],
        "affected_service": service,
        "user_impact": impact,
        "technical_details": details,
        "severity": severity,
        "severity_reason": severity_reason,
        "urgency": urgency,
        "urgency_reason": urgency_reason,
    }
    changed = [name for name, value in proposed.items() if value != (current.get(name) or "")]
    if not changed:
        return current
    now = _iso(_now())
    with _connect() as connection:
        connection.execute(
            """
            UPDATE incidents
            SET title = ?, description = ?, category = ?, affected_service = ?, user_impact = ?,
                technical_details = ?, severity = ?, severity_reason = ?, urgency = ?,
                urgency_reason = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                proposed["title"],
                proposed["description"],
                proposed["category"],
                proposed["affected_service"],
                proposed["user_impact"],
                proposed["technical_details"],
                proposed["severity"],
                proposed["severity_reason"],
                proposed["urgency"],
                proposed["urgency_reason"],
                now,
                incident_id,
            ),
        )
    add_event(incident_id, "edited", "Updated " + ", ".join(changed) + ".")
    updated = get_incident(incident_id)
    if updated is None:
        raise RuntimeError("Incident disappeared after update.")
    return updated


def link_duplicate_report(existing_id: int, transcript: str, reporter_name: str = "") -> None:
    existing = get_incident(existing_id)
    if existing is None:
        raise ValueError("The existing incident was not found.")
    who = f" Reporter: {reporter_name.strip()}." if reporter_name.strip() else ""
    message = (
        "A similar new report was matched and was not filed as another incident."
        + who
        + " Transcript: "
        + " ".join(transcript.split())
    )
    add_event(existing_id, "duplicate_report", message)
    now = _iso(_now())
    with _connect() as connection:
        connection.execute("UPDATE incidents SET updated_at = ? WHERE id = ?", (now, existing_id))


def seed_samples() -> None:
    now = _now()
    for sample in SAMPLES:
        when = now - timedelta(days=float(sample["age_days"]))
        text = str(sample["text"])
        language_code = language_for_speech(None, text)
        analysis = analyze_incident(text, language_code)
        payload = {
            **analysis,
            "transcript": text,
            "description": text,
            "language_code": language_code,
            "status": sample["status"],
            "source": "sample",
            "reporter_name": sample["reporter"],
            "is_sample": True,
            "stt_model": None,
            "stt_request_id": None,
        }
        incident = _insert(
            payload,
            when,
            "Demonstration incident loaded locally. It was not transcribed by Gnani.",
        )
        if sample["status"] != "Open":
            add_event(
                incident["id"],
                "status",
                f"Demonstration status set to {sample['status']}.",
                when + timedelta(hours=2),
            )
