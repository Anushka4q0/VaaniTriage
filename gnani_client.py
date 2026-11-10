"""Live clients for Gnani Prisma v3.0 and Timbre v2.5.

Request shapes follow the official docs:
https://docs.gnani.ai/api/STT/speech-to-text
https://docs.gnani.ai/api/STT/models/prisma-v3
https://docs.gnani.ai/api/TTS/tts-inference
https://docs.gnani.ai/api/TTS/available-voices
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Callable

import requests

from vaani import config
from vaani.languages import voice_for_language

STT_KEYTERMS = [
    "Outlook",
    "VPN",
    "SAP",
    "Wi-Fi",
    "Active Directory",
    "server",
    "outage",
    "password",
    "database",
    "printer",
    "error",
]
STT_TOPIC = "IT helpdesk incident about software, network, access, email, or infrastructure problems."

TTS_AUDIO_CONFIG = {
    "sample_rate": 48000,
    "num_channels": 1,
    "sample_width": 2,
    "encoding": "linear_pcm",
    "container": "wav",
}

FRIENDLY_ERRORS = {
    "MISSING_API_KEY": "The Gnani API key was not accepted. Set GNANI_API_KEY in the .env file.",
    "INVALID_API_KEY": "Gnani rejected the API key. Check GNANI_API_KEY in the .env file.",
    "SCOPE_NOT_ALLOWED": "This API key is not allowed to call that Gnani service.",
    "RATE_LIMITED": "Gnani is limiting requests. Wait a moment and try again.",
    "MAX_AUDIO_DURATION_EXCEEDED": "Prisma accepts clips of up to 30 seconds. Trim the recording and try again.",
    "UNSUPPORTED_AUDIO_FORMAT": "Prisma could not accept that file type. Use WAV, MP3, OGG, FLAC, AAC, or M4A.",
    "EMPTY_AUDIO": "Prisma received an empty audio file.",
    "AUDIO_CONVERSION_ERROR": "Prisma could not decode that audio file.",
    "INVALID_LANGUAGE_CODE": "That language code is not supported by Prisma.",
    "MODEL_UNSUPPORTED_LANGUAGE": "Prisma v3.0 does not support that language.",
    "UPSTREAM_TIMEOUT": "Prisma did not finish in time. Try a shorter clip.",
}


class GnaniAPIError(Exception):
    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        error_type: str | None = None,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_type = error_type
        self.request_id = request_id


def _friendly(error_type: str | None, message: str) -> str:
    lead = FRIENDLY_ERRORS.get(error_type or "")
    cleaned = config.redact(message or "The Gnani request failed.")
    if lead and lead not in cleaned:
        return f"{lead} {cleaned}".strip()
    return cleaned


def parse_error(response: requests.Response) -> GnaniAPIError:
    status = response.status_code
    body: Any
    try:
        body = response.json()
    except ValueError:
        snippet = config.redact(response.text[:300] if response.text else "")
        return GnaniAPIError(
            snippet or f"Gnani returned HTTP {status} without a JSON body.",
            status,
        )

    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            error_type = str(error.get("type") or "")
            message = str(error.get("message") or "Speech request failed.")
            return GnaniAPIError(
                _friendly(error_type, message),
                status,
                error_type or None,
                body.get("request_id"),
            )
        if body.get("code"):
            error_type = str(body.get("code"))
            message = str(body.get("message") or body.get("error") or "Gnani request failed.")
            return GnaniAPIError(_friendly(error_type, message), status, error_type)
        detail = body.get("detail")
        if isinstance(detail, dict):
            error_type = str(detail.get("error_code") or "")
            message = str(detail.get("message") or "Gnani request failed.")
            return GnaniAPIError(_friendly(error_type or None, message), status, error_type or None)
        if isinstance(detail, list):
            parts = []
            for item in detail:
                if isinstance(item, dict):
                    parts.append(str(item.get("msg") or item))
                else:
                    parts.append(str(item))
            return GnaniAPIError(_friendly("VALIDATION_ERROR", "; ".join(parts) or "Invalid request."), status, "VALIDATION_ERROR")
        if body.get("message"):
            return GnaniAPIError(_friendly(None, str(body.get("message"))), status, None)
        if body.get("success") is False:
            return GnaniAPIError(_friendly(None, "Gnani reported that the request failed."), status, None)
    return GnaniAPIError(f"Gnani returned HTTP {status}.", status)


class GnaniClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        session: requests.Session | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        timeout: float = 60,
    ) -> None:
        self.api_key = config.api_key() if api_key is None else api_key.strip()
        self.base_url = (base_url or config.api_base()).rstrip("/")
        self.session = session or requests.Session()
        self.sleeper = sleeper
        self.timeout = timeout

    def _require_key(self) -> str:
        if not self.api_key:
            raise GnaniAPIError(
                "GNANI_API_KEY is not set. Add it to the .env file in the project folder.",
                error_type="MISSING_API_KEY",
            )
        return self.api_key

    def _headers(self, request_id: str, json_body: bool = False) -> dict[str, str]:
        headers = {
            "X-API-Key-ID": self._require_key(),
            "X-API-Request-ID": request_id,
        }
        if json_body:
            headers["Content-Type"] = "application/json"
        return headers

    def _send(self, method: str, url: str, request_id: str, **kwargs: Any) -> requests.Response:
        attempts = 2
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                response = self.session.request(method, url, timeout=self.timeout, **kwargs)
            except requests.Timeout as exc:
                last_error = exc
                if attempt + 1 == attempts:
                    raise GnaniAPIError("The Gnani request timed out. Check the network and try again.") from exc
                self.sleeper(1.5)
                continue
            except requests.ConnectionError as exc:
                last_error = exc
                if attempt + 1 == attempts:
                    raise GnaniAPIError("Could not reach the Gnani API. Check the network and try again.") from exc
                self.sleeper(1.5)
                continue
            if response.status_code in {429, 503, 504} and attempt + 1 < attempts:
                self.sleeper(1.5)
                continue
            return response
        raise GnaniAPIError("The Gnani request failed.") from last_error

    def transcribe(
        self,
        audio_bytes: bytes,
        filename: str,
        mime_type: str,
        language_code: str = "indic-auto",
        output_format: str = "transcribe",
    ) -> dict[str, Any]:
        """POST /stt/v3 with model=prisma-v3.0. Returns the parsed JSON body."""
        if not audio_bytes:
            raise GnaniAPIError("Cannot transcribe an empty audio file.", error_type="EMPTY_AUDIO")
        request_id = str(uuid.uuid4())
        data: dict[str, str] = {
            "model": config.STT_MODEL,
            "format": output_format,
            "topic": STT_TOPIC,
            "keyterms": json.dumps(STT_KEYTERMS),
        }
        # language_code is required except when Prisma v3 transliterates.
        if not (output_format == "transliterate" and language_code in {"", "indic-auto"}):
            data["language_code"] = language_code
        url = self.base_url + config.STT_URL_PATH
        response = self._send(
            "POST",
            url,
            request_id,
            headers=self._headers(request_id),
            files={"audio_file": (filename, audio_bytes, mime_type)},
            data=data,
        )
        if response.status_code != 200:
            raise parse_error(response)
        try:
            body = response.json()
        except ValueError as exc:
            raise GnaniAPIError("Prisma returned a response that was not JSON.", response.status_code) from exc
        if not isinstance(body, dict) or body.get("success") is not True:
            raise parse_error(response)
        transcript = body.get("transcript")
        if not isinstance(transcript, str):
            raise GnaniAPIError("Prisma did not return a transcript string.", response.status_code, request_id=body.get("request_id"))
        body["client_request_id"] = request_id
        return body

    def synthesize(self, text: str, language_code: str) -> bytes:
        """POST /api/v1/tts/inference with Timbre v2.5. Returns WAV bytes."""
        spoken = " ".join((text or "").split())
        if not spoken:
            raise GnaniAPIError("There is no text to read aloud.")
        if len(spoken) > 600:
            spoken = spoken[:600].rsplit(" ", 1)[0]
        voice, language, model = voice_for_language(language_code)
        request_id = str(uuid.uuid4())
        payload = {
            "text": spoken,
            "voice": voice,
            "model": model,
            "language": language,
            "speed": 1.0,
            "audio_config": TTS_AUDIO_CONFIG,
        }
        url = self.base_url + config.TTS_URL_PATH
        response = self._send(
            "POST",
            url,
            request_id,
            headers=self._headers(request_id, json_body=True),
            json=payload,
        )
        content_type = response.headers.get("Content-Type", "")
        if response.status_code == 200 and (
            content_type.startswith("audio/") or response.content[:4] in {b"RIFF", b"ID3", b"OggS"}
        ):
            if len(response.content) < 44:
                raise GnaniAPIError("Timbre returned an audio response that was too small to play.")
            return response.content
        raise parse_error(response)
