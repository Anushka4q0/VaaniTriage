"""Call the live Gnani Prisma v3.0 and Timbre v2.5 endpoints.

This script never substitutes a fake transcript or a fake audio file.
A non-200 response is printed and the process exits with an error.
The API key is read from the environment and is not printed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vaani import config
from vaani.audio_utils import write_tone_wav
from vaani.gnani_client import GnaniAPIError, GnaniClient


def _speech_wav(path: Path) -> str:
    """Write a short spoken WAV with the Windows voice, if that voice is available."""
    sentence = "The VPN is down for the entire office and nobody can connect to the corporate network."
    try:
        import win32com.client  # type: ignore
    except ImportError:
        win32com = None
    else:
        try:
            voice = win32com.client.Dispatch("SAPI.SpVoice")
            stream = win32com.client.Dispatch("SAPI.SpFileStream")
            stream.Open(str(path), 3)
            voice.AudioOutputStream = stream
            voice.Speak(sentence)
            stream.Close()
            if path.exists() and path.stat().st_size > 1000:
                return "windows-sapi"
        except Exception:
            pass
    command = (
        "Add-Type -AssemblyName System.Speech; "
        "$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$synth.SetOutputToWaveFile('{path}'); "
        f"$synth.Speak('{sentence}'); "
        "$synth.Dispose()"
    )
    import subprocess

    completed = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode == 0 and path.exists() and path.stat().st_size > 1000:
        return "windows-speech"
    write_tone_wav(path, seconds=1.2)
    return "tone-not-speech"


def _print_http(label: str, response: requests.Response) -> None:
    body = config.redact(response.text[:500] if response.text else "")
    print(f"{label} HTTP {response.status_code}")
    print(body or "(empty body)")


def _probe_without_key(base: str, wav: bytes) -> int:
    print("GNANI_API_KEY is not set. Sending the documented requests without a key.")
    print("These calls are expected to fail. No transcript or audio will be invented.")
    stt = requests.post(
        base + config.STT_URL_PATH,
        files={"audio_file": ("incident.wav", wav, "audio/wav")},
        data={"model": config.STT_MODEL, "language_code": "en-IN", "format": "transcribe"},
        timeout=60,
    )
    _print_http("Prisma", stt)
    tts = requests.post(
        base + config.TTS_URL_PATH,
        json={
            "text": "Vaani Triage is connected to Gnani.",
            "voice": "Kaveri",
            "model": config.TTS_MODEL,
            "language": "en-IN",
            "speed": 1.0,
        },
        timeout=60,
    )
    _print_http("Timbre", tts)
    return 2


def main() -> int:
    out_dir = config.DATA_DIR / "api_tests"
    out_dir.mkdir(parents=True, exist_ok=True)
    sample = out_dir / "prisma_input.wav"
    source = _speech_wav(sample)
    wav = sample.read_bytes()
    base = config.api_base()
    print(f"Audio sample: {source}, {len(wav)} bytes, {sample}")
    print(f"API base: {base}")
    print(f"Key status: {config.key_status()}")
    if not config.api_key():
        return _probe_without_key(base, wav)

    client = GnaniClient()
    report: dict[str, object] = {"audio_source": source, "prisma": None, "timbre": None}
    try:
        body = client.transcribe(wav, sample.name, "audio/wav", "en-IN", "transcribe")
    except GnaniAPIError as exc:
        print(f"Prisma FAILED HTTP {exc.status_code} type={exc.error_type}")
        print(config.redact(str(exc)))
        return 1
    public = {
        "success": body.get("success"),
        "request_id": body.get("request_id"),
        "model": body.get("model"),
        "transcript": body.get("transcript"),
        "detected_language_code": body.get("detected_language_code"),
        "processing_time": body.get("processing_time"),
        "truncated": body.get("truncated"),
    }
    report["prisma"] = public
    print("Prisma SUCCESS")
    print(json.dumps(public, ensure_ascii=False))
    if source == "tone-not-speech" and not str(body.get("transcript") or "").strip():
        print("Prisma accepted the file and returned an empty transcript because the sample was a tone, not speech.")

    phrase = "Your incident has been saved. The severity is high."
    try:
        audio = client.synthesize(phrase, "en-IN")
    except GnaniAPIError as exc:
        print(f"Timbre FAILED HTTP {exc.status_code} type={exc.error_type}")
        print(config.redact(str(exc)))
        (out_dir / "live_result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return 1
    target = out_dir / "timbre_result.wav"
    target.write_bytes(audio)
    report["timbre"] = {
        "success": True,
        "model": config.TTS_MODEL,
        "voice": "Kaveri",
        "language": "en-IN",
        "bytes": len(audio),
        "header": audio[:4].decode("latin1"),
        "file": str(target),
    }
    print("Timbre SUCCESS")
    print(json.dumps(report["timbre"]))
    (out_dir / "live_result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
