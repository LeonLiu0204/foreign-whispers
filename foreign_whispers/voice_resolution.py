"""Voice resolution for Chatterbox speaker cloning.

Resolves which reference WAV to use for a given target language
and optional speaker ID. The Chatterbox container expects a filename
relative to its /app/voices/ mount point.
"""

from pathlib import Path


def resolve_speaker_wav(
    speakers_dir: Path,
    target_language: str,
    speaker_id: str | None = None,
) -> str:
    """Resolve the reference WAV path for voice cloning.

    Resolution order:
    1. speakers/{lang}/{speaker_id}.wav  (if speaker_id given and file exists)
    2. speakers/{lang}/default.wav       (language-specific default)
    3. speakers/default.wav              (global fallback)

    Args:
        speakers_dir: Absolute path to the speakers directory.
        target_language: Language code (e.g. "es", "fr").
        speaker_id: Optional speaker identifier (e.g. "SPEAKER_00").

    Returns:
        Relative path string for the Chatterbox container (e.g. "es/default.wav").
    """
    # ---- YOUR CODE HERE ----
    speakers_dir = Path(speakers_dir)

    lang = (target_language or "").strip()
    speaker = (speaker_id or "").strip() if speaker_id else None

    if speaker:
        speaker_specific = speakers_dir / lang / f"{speaker}.wav"
        if speaker_specific.exists():
            return f"{lang}/{speaker}.wav"

    language_default = speakers_dir / lang / "default.wav"
    if language_default.exists():
        return f"{lang}/default.wav"

    global_default = speakers_dir / "default.wav"
    if global_default.exists():
        return "default.wav"

    raise FileNotFoundError(
        f"No reference WAV found in {speakers_dir} for "
        f"target_language={target_language!r}, speaker_id={speaker_id!r}"
    )
    # ---- END YOUR CODE ----
