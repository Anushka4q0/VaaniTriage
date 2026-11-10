"""Live multilingual verification for Gnani Prisma v3.0 and Timbre v2.5 across languages.

Tests real synthesis requests for Telugu (Suhana), Tamil (Asmita), Hindi (Nalini),
and English (Kaveri), plus Prisma v3.0 STT language detection.
Never prints or logs API keys.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vaani import config
from vaani.gnani_client import GnaniAPIError, GnaniClient
from vaani.languages import (
    confirmation_text,
    follow_up_speech,
    suggestion_speech,
    voice_for_language,
)
from vaani.nlp import analyze_incident


def main() -> int:
    print(f"API base: {config.api_base()}")
    print(f"Key status: {config.key_status()}")
    if not config.api_key():
        print("ERROR: GNANI_API_KEY is not configured.")
        return 1

    client = GnaniClient()
    out_dir = config.DATA_DIR / "api_tests"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Test Timbre v2.5 across 4 key languages
    test_cases = [
        ("te-IN", "మీ ఇన్సిడెంట్ విజయవంతంగా నమోదయింది.", "Suhana"),
        ("ta-IN", "உங்கள் சம்பவம் வெற்றிகரமாக பதிவாகிவிட்டது.", "Asmita"),
        ("hi-IN", "आपकी घटना सफलतापूर्वक दर्ज कर ली गई है।", "Nalini"),
        ("en-IN", "Your incident has been saved successfully.", "Kaveri"),
    ]

    tts_results: dict[str, dict[str, object]] = {}
    for lang_code, phrase, expected_voice in test_cases:
        voice, lang, model = voice_for_language(lang_code)
        assert voice == expected_voice, f"Expected voice {expected_voice} for {lang_code}, got {voice}"
        try:
            audio_bytes = client.synthesize(phrase, lang_code)
        except GnaniAPIError as exc:
            print(f"Timbre FAILED for {lang_code} ({expected_voice}): {config.redact(str(exc))}")
            return 1
        
        target_file = out_dir / f"timbre_{lang_code}.wav"
        target_file.write_bytes(audio_bytes)
        
        header = audio_bytes[:4].decode("latin1", errors="replace")
        assert header in ("RIFF", "ID3", "OggS"), f"Invalid audio header: {header}"
        assert len(audio_bytes) > 44, "Audio file too small"
        
        tts_results[lang_code] = {
            "success": True,
            "voice": voice,
            "language": lang,
            "model": model,
            "bytes": len(audio_bytes),
            "header": header,
            "file": str(target_file),
        }
        print(f"Timbre SUCCESS for {lang_code} | Voice: {voice} | Size: {len(audio_bytes)} bytes | Header: {header}")

    # 2. Test Prisma v3.0 STT with actual speech audio
    sample_file = out_dir / "prisma_input.wav"
    if not sample_file.exists():
        sample_file = config.DATA_DIR / "uploads" / "test_run_audio.wav"

    if sample_file.exists():
        wav_bytes = sample_file.read_bytes()
        print(f"\nTesting Prisma STT with {sample_file.name} ({len(wav_bytes)} bytes)...")
        try:
            stt_body = client.transcribe(wav_bytes, sample_file.name, "audio/wav", "indic-auto", "transcribe")
        except GnaniAPIError as exc:
            print(f"Prisma FAILED: {config.redact(str(exc))}")
            return 1
        
        transcript = stt_body.get("transcript", "")
        detected_lang = stt_body.get("detected_language_code") or "en-IN"
        print(f"Prisma SUCCESS | Model: {stt_body.get('model')} | Request ID: {stt_body.get('request_id')}")
        print(f"Transcript: {transcript}")
        print(f"Detected language code: {detected_lang}")

        # 3. Test end-to-end analysis on the transcript
        analysis = analyze_incident(transcript, detected_lang)
        print(f"\nIncident Analysis:")
        print(f"  Category: {analysis['category']}")
        print(f"  Severity: {analysis['severity']} ({analysis['severity_reason']})")
        print(f"  Urgency: {analysis['urgency']} ({analysis['urgency_reason']})")
        print(f"  Follow-ups count: {len(analysis['follow_ups'])}")
        print(f"  Suggestions count: {len(analysis['suggestions'])}")

    report = {
        "tts_results": tts_results,
        "stt_tested": sample_file.exists(),
    }
    (out_dir / "multilingual_live_results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nAll live multilingual checks completed successfully!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
