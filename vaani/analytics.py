"""Aggregate incident records for the dashboard."""

from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd

IST = ZoneInfo("Asia/Kolkata")
OPEN_STATUSES = {"Open", "In Progress", "Waiting on User"}


def format_timestamp(value: str | None) -> str:
    if not value:
        return ""
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return value
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(IST).strftime("%d %b %Y, %H:%M IST")


def _frame(incidents: list[dict]) -> pd.DataFrame:
    if not incidents:
        return pd.DataFrame(
            columns=["id", "reference", "created_at", "title", "category", "severity", "urgency", "status", "source", "is_sample", "affected_service"]
        )
    return pd.DataFrame(incidents)


def summarize(incidents: list[dict]) -> dict:
    frame = _frame(incidents)
    total = int(len(frame))
    if total == 0:
        empty = pd.DataFrame({"count": []})
        return {
            "total": 0,
            "samples": 0,
            "live": 0,
            "open": 0,
            "resolved": 0,
            "critical_open": 0,
            "high_open": 0,
            "by_severity": empty,
            "by_urgency": empty,
            "by_category": empty,
            "by_status": empty,
            "trend": empty,
            "recent": frame,
        }
    open_mask = frame["status"].isin(OPEN_STATUSES)
    resolved_mask = frame["status"].isin(["Resolved", "Closed"])
    stamped = pd.to_datetime(frame["created_at"], utc=True, errors="coerce").dt.tz_convert(IST)
    trend = (
        pd.DataFrame({"day": stamped.dt.strftime("%d %b"), "incidents": 1, "sort_day": stamped.dt.date})
        .dropna()
        .groupby(["sort_day", "day"], as_index=False)["incidents"]
        .sum()
        .sort_values("sort_day")
    )
    severity_order = ["Critical", "High", "Medium", "Low"]
    by_severity = (
        frame["severity"].value_counts().reindex(severity_order, fill_value=0).rename_axis("severity").reset_index(name="incidents")
    )
    urgency_order = ["Immediate", "Urgent", "Today", "Scheduled"]
    if "urgency" not in frame.columns:
        frame["urgency"] = "Today"
    by_urgency = (
        frame["urgency"].value_counts().reindex(urgency_order, fill_value=0).rename_axis("urgency").reset_index(name="incidents")
    )
    by_category = (
        frame["category"].value_counts().rename_axis("category").reset_index(name="incidents").sort_values("incidents", ascending=False)
    )
    by_status = frame["status"].value_counts().rename_axis("status").reset_index(name="incidents")
    recent = frame.copy()
    recent["reported"] = recent["created_at"].map(format_timestamp)
    recent["origin"] = recent.apply(lambda row: "Sample" if row["is_sample"] else ("Gnani" if row["source"] == "gnani_prisma" else "Typed"), axis=1)
    return {
        "total": total,
        "samples": int(frame["is_sample"].sum()),
        "live": int((~frame["is_sample"]).sum()),
        "open": int(open_mask.sum()),
        "resolved": int(resolved_mask.sum()),
        "critical_open": int(((frame["severity"] == "Critical") & open_mask).sum()),
        "high_open": int(((frame["severity"] == "High") & open_mask).sum()),
        "by_severity": by_severity,
        "by_urgency": by_urgency,
        "by_category": by_category,
        "by_status": by_status,
        "trend": trend,
        "recent": recent,
    }
