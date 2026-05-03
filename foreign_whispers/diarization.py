"""Speaker diarization using pyannote.audio.

Extracted from notebooks/foreign_whispers_pipeline.ipynb (M2-align).

Optional dependency: pyannote.audio
    pip install pyannote.audio
Requires accepting the pyannote/speaker-diarization-3.1 licence on HuggingFace
and providing an HF token.  Returns empty list with a warning if the dep is
absent or the token is missing.
"""
import logging

logger = logging.getLogger(__name__)


def diarize_audio(audio_path: str, hf_token: str | None = None) -> list[dict]:
    """Return speaker-labeled intervals for *audio_path*.

    Returns:
        List of ``{start_s: float, end_s: float, speaker: str}``.
        Empty list when pyannote.audio is absent, token is missing, or diarization fails.
    """
    if not hf_token:
        logger.warning("No HF token provided — diarization skipped.")
        return []

    try:
        from pyannote.audio import Pipeline
    except (ImportError, TypeError):
        logger.warning("pyannote.audio not installed — returning empty diarization.")
        return []

    try:
        pipeline    = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )
        diarization = pipeline(audio_path)
        return [
            {"start_s": turn.start, "end_s": turn.end, "speaker": speaker}
            for turn, _, speaker in diarization.itertracks(yield_label=True)
        ]
    except Exception as exc:
        logger.warning("Diarization failed for %s: %s", audio_path, exc)
        return []


def assign_speakers(
    segments: list[dict],
    diarization: list[dict],
    default_speaker: str = "UNKNOWN",
) -> list[dict]:
    """Assign speaker labels to transcript segments.

    Each transcript segment is assigned the speaker whose diarization interval
    has the largest time overlap with that segment.
    """
    assigned = []

    for seg in segments:
        seg_start = float(seg.get("start", seg.get("start_s", 0.0)))
        seg_end = float(seg.get("end", seg.get("end_s", seg_start)))

        best_speaker = default_speaker
        best_overlap = 0.0

        for turn in diarization:
            turn_start = float(turn.get("start_s", turn.get("start", 0.0)))
            turn_end = float(turn.get("end_s", turn.get("end", turn_start)))

            overlap = max(0.0, min(seg_end, turn_end) - max(seg_start, turn_start))

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = turn.get("speaker", default_speaker)

        new_seg = dict(seg)
        new_seg["speaker"] = best_speaker
        assigned.append(new_seg)

    return assigned