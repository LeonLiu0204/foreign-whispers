"""POST /api/diarize/{video_id} — speaker diarization (issue fw-lua)."""

import asyncio
import json
import subprocess

from fastapi import APIRouter, HTTPException

from api.src.core.config import settings
from api.src.core.dependencies import resolve_title
from api.src.schemas.diarize import DiarizeResponse
from api.src.services.alignment_service import AlignmentService

from foreign_whispers.diarization import assign_speakers

router = APIRouter(prefix="/api")

_alignment_service = AlignmentService(settings=settings)


@router.post("/diarize/{video_id}", response_model=DiarizeResponse)
async def diarize_endpoint(video_id: str):
    """Run speaker diarization on a video's audio track.

    Steps:
    1. Extract audio from video via ffmpeg
    2. Run pyannote diarization
    3. Cache and return speaker segments
    """
    title = resolve_title(video_id)
    if title is None:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found")

    diar_dir = settings.diarizations_dir
    diar_dir.mkdir(parents=True, exist_ok=True)
    diar_path = diar_dir / f"{title}.json"

    # Return cached result
    if diar_path.exists():
        data = json.loads(diar_path.read_text())
        return DiarizeResponse(
            video_id=video_id,
            speakers=data.get("speakers", []),
            segments=data.get("segments", []),
            skipped=True,
        )

    # ---- YOUR CODE HERE ----
    video_path = settings.videos_dir / f"{title}.mp4"
    if not video_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Video file not found: {video_path}",
        )

    audio_path = diar_dir / f"{title}.wav"

    try:
        await asyncio.to_thread(
            subprocess.run,
            [
                "ffmpeg",
                "-y",
                "-i",
                str(video_path),
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                "16000",
                "-ac",
                "1",
                str(audio_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Audio extraction failed: {exc.stderr}",
        ) from exc
        
    diar_segments = await asyncio.to_thread(
        _alignment_service.diarize,
        str(audio_path),
    )
    
    speakers = sorted({seg["speaker"] for seg in diar_segments if "speaker" in seg})
    
    # Merge speaker labels into the cached transcription JSON.
    transcript_path = settings.transcriptions_dir / f"{title}.json"
    if transcript_path.exists():
        transcript_data = json.loads(transcript_path.read_text())
        transcript_segments = transcript_data.get("segments", [])
    
        transcript_data["segments"] = assign_speakers(
            transcript_segments,
            diar_segments,
        )
    
        transcript_path.write_text(json.dumps(transcript_data, indent=2))
    
    result = {
        "speakers": speakers,
        "segments": diar_segments,
    }
    diar_path.write_text(json.dumps(result, indent=2))
    
    return DiarizeResponse(
        video_id=video_id,
        speakers=speakers,
        segments=diar_segments,
        skipped=False,
    )