"""Text-to-speech using gTTS (needs internet access)."""
import hashlib
from pathlib import Path

from gtts import gTTS

from app.config import AUDIO_DIR


def text_to_speech(text: str, lang: str = "en") -> Path:
    """Convert text to an mp3 file. Identical text reuses the cached file."""
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    filename = hashlib.md5(f"{lang}:{text}".encode("utf-8")).hexdigest() + ".mp3"
    path = AUDIO_DIR / filename
    if not path.exists():
        gTTS(text=text, lang=lang).save(str(path))
    return path
