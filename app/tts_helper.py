import os
import sys
import hashlib
import asyncio
import logging
from typing import Optional

logger = logging.getLogger("cv_fit_tts")

AUDIO_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio_cache")
os.makedirs(AUDIO_CACHE_DIR, exist_ok=True)

# Top tier neural male gym coach voices
MALE_VOICE_MAP = {
    "ar": "ar-EG-ShakirNeural",        # Natural, athletic, masculine Egyptian Arabic voice
    "en": "en-US-ChristopherNeural",   # Energetic, masculine American English coach voice
}

def get_cache_path(text: str, lang: str) -> str:
    clean_text = text.strip()
    key = hashlib.md5(f"{lang}:{clean_text}".encode("utf-8")).hexdigest()
    return os.path.join(AUDIO_CACHE_DIR, f"{key}.mp3")

async def synthesize_speech_async(text: str, lang: str = "ar") -> Optional[bytes]:
    """Synthesize text to MP3 bytes using Edge-TTS with a real neural male coach voice."""
    clean_text = text.strip()
    if not clean_text:
        return None

    cache_file = get_cache_path(clean_text, lang)
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                return f.read()
        except Exception:
            pass

    voice = MALE_VOICE_MAP.get(lang, MALE_VOICE_MAP["en"])
    try:
        import edge_tts
        # Slightly faster, confident athletic cadence (+6%)
        communicate = edge_tts.Communicate(clean_text, voice, rate="+6%")
        chunks = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                chunks.append(chunk["data"])
        audio_bytes = b"".join(chunks)

        if audio_bytes:
            with open(cache_file, "wb") as f:
                f.write(audio_bytes)
            return audio_bytes
    except Exception as e:
        logger.warning(f"Failed to synthesize voice via edge-tts: {e}")
        return None

def get_speech_audio(text: str, lang: str = "ar") -> Optional[bytes]:
    """Synchronous / cached retrieval of speech audio."""
    clean_text = text.strip()
    cache_file = get_cache_path(clean_text, lang)
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                return f.read()
        except Exception:
            pass
    try:
        return asyncio.run(synthesize_speech_async(clean_text, lang))
    except Exception:
        return None
