"""Validate short recordings before they are sent to Prisma."""

from __future__ import annotations

import io
import subprocess
import tempfile
import wave
from pathlib import Path

MAX_BYTES = 8 * 1024 * 1024
MAX_SECONDS = 30.0
MIN_SECONDS = 0.15

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".aac", ".m4a"}
MIME_TYPES = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    ".aac": "audio/aac",
    ".m4a": "audio/mp4",
}


class AudioValidationError(ValueError):
    pass


def _wav_duration(data: bytes) -> float | None:
    try:
        with wave.open(io.BytesIO(data), "rb") as handle:
            rate = handle.getframerate()
            channels = handle.getnchannels()
            sampwidth = handle.getsampwidth()
            if rate <= 0 or channels <= 0 or sampwidth <= 0:
                return None
            bytes_per_sec = rate * channels * sampwidth
            frames = handle.getnframes()
            if frames * channels * sampwidth > len(data):
                return max(0.0, (len(data) - 44) / float(bytes_per_sec))
            return frames / float(rate)
    except (wave.Error, EOFError):
        return None


def _mutagen_duration(data: bytes, suffix: str) -> float | None:
    try:
        from mutagen import File as MutagenFile
    except ImportError:
        return None
    buffer = io.BytesIO(data)
    buffer.name = "clip" + suffix
    try:
        parsed = MutagenFile(buffer)
    except Exception:
        return None
    if parsed is None or not getattr(parsed, "info", None):
        return None
    length = getattr(parsed.info, "length", None)
    if length is None:
        return None
    return float(length)


def sniff_extension(filename: str, data: bytes) -> str:
    suffix = Path(filename or "").suffix.lower()
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return ".wav"
    if data[:4] == b"fLaC":
        return ".flac"
    if data[:4] == b"OggS":
        return ".ogg"
    if data[:3] == b"ID3" or (len(data) > 2 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0):
        return ".mp3"
    if suffix in SUPPORTED_EXTENSIONS:
        return suffix
    return suffix


def prepare_audio(filename: str, data: bytes) -> tuple[str, bytes, str]:
    """Return a safe filename, the original bytes, and a MIME type.

    Prisma REST accepts WAV, MP3, OGG, FLAC, AAC, and M4A, up to 30 seconds.
    """
    if not data:
        raise AudioValidationError("The audio file is empty. Record or choose a clip that contains speech.")
    if len(data) > MAX_BYTES:
        raise AudioValidationError("The audio file is larger than 8 MB. Use a shorter clip.")
    if data[:4] == b"\x1a\x45\xdf\xa3":
        data = _webm_to_wav(data)
        filename = "recording.wav"

    suffix = sniff_extension(filename, data)
    if suffix not in SUPPORTED_EXTENSIONS:
        raise AudioValidationError(
            "Unsupported audio format. Upload WAV, MP3, OGG, FLAC, AAC, or M4A (30 seconds or less)."
        )

    duration = _wav_duration(data) if suffix == ".wav" else _mutagen_duration(data, suffix)
    if duration is not None and duration < MIN_SECONDS:
        raise AudioValidationError("The recording is too short. Describe the problem in a few sentences.")
    if duration is not None and duration > MAX_SECONDS:
        raise AudioValidationError(
            f"The clip is {duration:.1f} seconds. Prisma REST accepts up to 30 seconds. Trim it and try again."
        )

    safe_name = "incident" + suffix
    return safe_name, data, MIME_TYPES[suffix]


def _webm_to_wav(data: bytes) -> bytes:
    """Browser recordings are often WebM. Prisma REST does not accept WebM, so convert to 16 kHz mono WAV."""
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise AudioValidationError(
            "This browser recording is WebM, which Prisma does not accept, and the WAV converter is not installed. "
            "Upload WAV, MP3, OGG, FLAC, AAC, or M4A instead."
        ) from exc
    try:
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise AudioValidationError(
            "The browser recording is WebM and could not be converted to WAV. Upload WAV, MP3, OGG, FLAC, AAC, or M4A."
        ) from exc
    with tempfile.TemporaryDirectory() as folder:
        source = Path(folder) / "clip.webm"
        target = Path(folder) / "clip.wav"
        source.write_bytes(data)
        completed = subprocess.run(
            [ffmpeg, "-y", "-i", str(source), "-ac", "1", "-ar", "16000", str(target)],
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0 or not target.exists() or target.stat().st_size < 44:
            raise AudioValidationError(
                "The browser recording could not be converted to WAV. Upload WAV, MP3, OGG, FLAC, AAC, or M4A."
            )
        return target.read_bytes()


def write_tone_wav(path: Path, seconds: float = 1.0, rate: int = 16000) -> None:
    """Write a quiet mono WAV. Used to exercise the real speech endpoint."""
    frames = int(seconds * rate)
    # A soft 440 Hz tone so the file is valid speech-shaped audio, not digital silence.
    import math
    import struct

    samples = bytearray()
    for index in range(frames):
        value = int(1200 * math.sin(2 * math.pi * 440 * index / rate))
        samples.extend(struct.pack("<h", value))
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(bytes(samples))
