"""
sarvam_voice.py — Speech-to-Text and Text-to-Speech via Sarvam AI API
"""

import base64
import io
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def transcribe_audio(audio_bytes: bytes, api_key: str, language_code: str = "unknown") -> Optional[str]:
    """
    Send audio bytes to Sarvam AI STT and return transcribed text.

    Args:
        audio_bytes: Raw audio file bytes (wav, mp3, webm, ogg, etc.)
        api_key: Sarvam AI subscription key
        language_code: BCP-47 language code or 'unknown' for auto-detect

    Returns:
        Transcribed text string, or None on failure.
    """
    try:
        from sarvamai import SarvamAI
        client = SarvamAI(api_subscription_key=api_key)

        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = "recording.wav"

        response = client.speech_to_text.transcribe(
            file=audio_file,
            model="saarika:v2.5",
            language_code=language_code,
        )

        transcript = response.transcript
        if transcript:
            logger.info(f"STT success: {len(transcript)} chars")
            return transcript.strip()
        return None

    except Exception as e:
        logger.error(f"STT error: {e}")
        raise


def synthesize_speech(
    text: str,
    api_key: str,
    speaker: str = "anushka",
    language_code: str = "en-IN",
    pace: float = 1.0,
    pitch: float = 0.0,
) -> Optional[bytes]:
    """
    Convert text to speech via Sarvam AI TTS.

    Args:
        text: Text to synthesize (max ~500 chars per call recommended)
        api_key: Sarvam AI subscription key
        speaker: Voice speaker name
        language_code: Target language BCP-47 code
        pace: Speaking pace multiplier (0.5–2.0)
        pitch: Pitch shift (-0.5 to 0.5)

    Returns:
        MP3 audio bytes, or None on failure.
    """
    try:
        from sarvamai import SarvamAI
        client = SarvamAI(api_subscription_key=api_key)

        # Sarvam TTS works best with chunks ≤ 500 chars
        chunks = _split_text(text, max_chars=490)
        audio_parts: list[bytes] = []

        for chunk in chunks:
            if not chunk.strip():
                continue
            response = client.text_to_speech.convert(
                text=chunk,
                target_language_code=language_code,
                speaker=speaker,
                pace=pace,
                pitch=pitch,
                model="bulbul:v2",
                output_audio_codec="mp3",
            )
            # audios is a list of base64-encoded audio strings
            if response.audios:
                for audio_b64 in response.audios:
                    audio_parts.append(base64.b64decode(audio_b64))

        if audio_parts:
            logger.info(f"TTS success: {len(audio_parts)} chunk(s)")
            return b"".join(audio_parts)
        return None

    except Exception as e:
        logger.error(f"TTS error: {e}")
        raise


def _split_text(text: str, max_chars: int = 490) -> list[str]:
    """Split long text into sentence-aware chunks."""
    if len(text) <= max_chars:
        return [text]

    chunks = []
    sentences = text.replace("\n", " ").split(". ")
    current = ""
    for sent in sentences:
        candidate = (current + ". " + sent).strip() if current else sent
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            # If single sentence is too long, hard-split
            while len(sent) > max_chars:
                chunks.append(sent[:max_chars])
                sent = sent[max_chars:]
            current = sent
    if current:
        chunks.append(current)
    return chunks
