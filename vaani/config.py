"""Paths and environment settings. The API key stays in the environment."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"

STT_URL_PATH = "/stt/v3"
TTS_URL_PATH = "/api/v1/tts/inference"
STT_MODEL = "prisma-v3.0"
TTS_MODEL = "timbre-v2.5"
DEFAULT_API_BASE = "https://api.vachana.ai"


def load_env() -> None:
    load_dotenv(ROOT / ".env", override=False)


def api_base() -> str:
    load_env()
    return os.getenv("GNANI_API_BASE", DEFAULT_API_BASE).strip().rstrip("/") or DEFAULT_API_BASE


# def api_key() -> str:
#     load_env()
#     return os.getenv("GNANI_API_KEY", "").strip()

def api_key() -> str:
    load_env()

    # First check environment variables
    key = os.getenv("GNANI_API_KEY", "").strip()

    if key:
        return key

    # If not found, check Streamlit Secrets
    try:
        import streamlit as st
        return str(st.secrets.get("GNANI_API_KEY", "")).strip()
    except Exception:
        return ""



def db_path() -> Path:
    load_env()
    override = os.getenv("VAANI_DB_PATH", "").strip()
    if override:
        return Path(override)
    return DATA_DIR / "vaani.db"


def seed_samples_enabled() -> bool:
    load_env()
    return os.getenv("VAANI_SEED_SAMPLES", "1").strip() != "0"


def key_status() -> str:
    """Safe status for the interface. Does not reveal any part of the key."""
    key = api_key()
    if not key:
        return "missing"
    return f"loaded ({len(key)} characters)"


def redact(text: str) -> str:
    key = api_key()
    if key and key in text:
        return text.replace(key, "[redacted]")
    return text
