"""Demonstration incidents.

These rows are created by the local classifier so the desk is not empty on first
launch. They are stored with source='sample' and are not Gnani transcriptions.
"""

from __future__ import annotations

SAMPLES: list[dict[str, object]] = [
    {
        "age_days": 0,
        "status": "Open",
        "reporter": "Sample desk",
        "text": "The production email server is down. Nobody in the company can send or receive mail. Outlook shows error 0x80040115.",
    },
    {
        "age_days": 1,
        "status": "Open",
        "reporter": "Sample desk",
        "text": "The VPN is down for the entire office and nobody can connect to the corporate network.",
    },
    {
        "age_days": 1,
        "status": "In Progress",
        "reporter": "Sample desk",
        "text": "Several users cannot open SAP. It shows HTTP 500 on the portal.",
    },
    {
        "age_days": 2,
        "status": "Open",
        "reporter": "Sample desk",
        "text": "My laptop keeps disconnecting from the office Wi-Fi every few minutes.",
    },
    {
        "age_days": 2,
        "status": "Open",
        "reporter": "Sample desk",
        "text": "प्रोडक्शन ईमेल सर्वर डाउन है। कंपनी के सभी यूजर मेल नहीं भेज पा रहे हैं। Outlook में error 0x80040115 आ रहा है।",
    },
    {
        "age_days": 3,
        "status": "Waiting on User",
        "reporter": "Sample desk",
        "text": "I cannot sign in to Outlook. It asks for the password again but this is only happening for me.",
    },
    {
        "age_days": 3,
        "status": "Open",
        "reporter": "Sample desk",
        "text": "மின்னஞ்சல் சர்வர் வேலை செய்யவில்லை. அலுவலகத்தில் யாரும் மின்னஞ்சல் அனுப்ப முடியவில்லை.",
    },
    {
        "age_days": 4,
        "status": "Open",
        "reporter": "Sample desk",
        "text": "My laptop printer is jammed and the page will not come out.",
    },
    {
        "age_days": 5,
        "status": "Resolved",
        "reporter": "Sample desk",
        "text": "I need a password reset for my own account. This is a minor request.",
    },
    {
        "age_days": 6,
        "status": "Open",
        "reporter": "Sample desk",
        "text": "Possible ransomware on the finance file server. Files are encrypted and this is a data loss incident for all customers.",
    },
    {
        "age_days": 8,
        "status": "Resolved",
        "reporter": "Sample desk",
        "text": "The reporting database is unavailable for the sales team, so several users cannot run queries.",
    },
    {
        "age_days": 9,
        "status": "Closed",
        "reporter": "Sample desk",
        "text": "The dock on my desk sometimes loses the monitor signal.",
    },
]
