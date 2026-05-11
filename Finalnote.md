# Foreign Whispers Dubbing Pipeline

## Team

Xingjian Liu

NetID: xl6081

## Overview

This project implements an extended AI video dubbing pipeline: **Download → Transcribe → Diarize → Translate → TTS → Stitch**. 

**Primary Submission:**

- **GitHub Repo:** https://github.com/LeonLiu0204/foreign-whispers.git
- **Final Output Videos (Google Drive):**  https://drive.google.com/drive/folders/1jX8gE8kxNDTS5u_asYEncNYl2jMnXO1s?usp=drive_link

------

## Outputs

The Google Drive folder contains two distinct pipeline outputs to demonstrate system stability across different transcription granularities:

### 1. Output Variant 1: Whisper-based Primary Output (98 segments)

- **Folder:** output_by_98

- **Files:** Strait of Hormuz disruption threatens to shake global economy.mp4

  Strait of Hormuz disruption threatens to shake global economy.vtt

- **Detail:** This is the primary final output. It uses OpenAI Whisper transcription with 98 clean segments. The stable segment boundaries produced the most natural prosody and timing in the final Spanish dub.

### 2. Output Variant 2: YouTube-caption Fallback Experiment (336 segments)

- **Folder:** output_by_336

- **Files:** caption_336_dubbed_video.mp4

  caption_336_dubbed_captions.vtt

- **Detail:** This experimental path used manually downloaded YouTube .vtt captions. Because YouTube's auto-captions use "rolling" segments, it resulted in 336 fragmented segments. This output serves as a stress test, proving the pipeline can successfully process and stitch even highly granular data without failure.

------

## Environment

The project was executed on a Lambda Cloud GPU instance using Docker Compose. Due to the high-performance requirements of the Whisper and TTS models, the following environment setup was used:

#### 1. Start the Service Stack

Launch the API and frontend containers with the NVIDIA profile:

```bash
docker compose --profile nvidia up -d
```

#### 2. Health & Connectivity Checks

Verify the backend API:

```bash
curl http://localhost:8080/healthz
```

Verify the Streamlit frontend:

```bash
curl -I http://localhost:8501
```

#### 3. Local Development Setup

The local environment and Python dependencies are managed via **uv**:

```bash
uv sync
```

------

## End-to-End Pipeline

The complete pipeline was executed through the integration notebook: notebooks/pipeline_end_to_end/pipeline_end_to_end.ipynb`

### 1. Manual P1 Fallback

Since the Lambda Cloud instance was blocked by YouTube’s bot-detection (yt-dlp challenges), the P1 Download stage was adapted to a robust manual fallback. This ensured the downstream stages could proceed without being halted by infrastructure restrictions.

**Workflow:** Manual source MP4 + Downloaded VTT → JSONL Caption Conversion → Downstream API Pipeline

### 2. Execution Flow

The final notebook successfully executed the following stages:

- **P1 Manual Fallback:** Verified source files and prepared transcription data.
- **P2 Transcribe:** Generated 98 high-quality segments (Primary) or 336 segments (Experimental).
- **P2.5 Diarize: (Added Feature)** Applied speaker identification logic.
- **P3 Translate:** Translated segments into target Spanish.
- **P4 TTS:** Generated dubbed audio based on speaker-aware resolution.
- **P5 Stitch:** Merged dubbed audio with source video and generated VTT captions.

------

## Implemented Features

### 1. Manual P1 fallback

The original YouTube download route was blocked by YouTube bot/challenge checks on the cloud GPU instance. To keep the pipeline running, I added a manual fallback in the notebook that verifies the source MP4 and optionally converts a downloaded `.vtt` subtitle file into the pipeline’s expected JSONL caption format.

The fallback supports:

```
GYQ5yGV_-Oc.en.vtt
→ pipeline_data/api/youtube_captions/<title>.txt
```

The generated `.txt` file is a JSONL caption file, not a plain text subtitle file. Each line has the format:

```
{"start": 2.32, "duration": 4.15, "text": "60 Minutes overtime."}
```

------

### 2. Segment-speaker merge function

Implemented in:

```
foreign_whispers/diarization.py
```

Function:

```
assign_speakers(segments, diarization)
```

This function assigns a speaker label to each transcript segment by finding the diarization interval with the largest temporal overlap. It does not mutate the original input. If no speaker interval matches, it defaults to:

```
SPEAKER_00
```

Tests were added in:

```
tests/test_diarization.py
```

The test result was:

```
7 passed, 1 skipped
```

------

### 3. Diarization API endpoint

Implemented:

```
POST /api/diarize/{video_id}
```

Files changed:

```
api/src/schemas/diarize.py
api/src/routers/diarize.py
api/src/main.py
api/src/core/config.py
docker-compose.yml
```

The endpoint:

- extracts audio from the source video,
- calls the diarization service,
- caches results in `pipeline_data/api/diarizations/`,
- returns a structured response,
- supports cache reuse with `skipped: true`.

The endpoint is available and callable from the notebook/API pipeline.

------

### 4. Speaker-label merge into transcription

After diarization runs, the endpoint updates the transcription JSON so each segment can contain a `speaker` field.

Example:

```
{
  "start": 0.0,
  "end": 3.6,
  "text": "60 minutes over time.",
  "speaker": "SPEAKER_00"
}
```

This prepares the downstream translation and TTS stages for speaker-aware dubbing.

------

### 5. Per-speaker TTS voice resolution

Implemented voice resolution logic so that TTS can choose a reference voice based on the speaker label.

Resolution order:

```
pipeline_data/speakers/{lang}/{speaker_id}.wav
pipeline_data/speakers/{lang}/default.wav
pipeline_data/speakers/default.wav
```

The TTS align report was also extended to include:

```
speaker
speaker_wav
```

This makes the backend ready for speaker-aware TTS when diarization labels and reference voice files are available.

------

### 6. Improved TTS duration estimation

Implemented in:

```
foreign_whispers/alignment.py
```

The duration estimate was improved from a simple syllable-rate heuristic to a more stable speech heuristic using:

- syllable count,
- punctuation pause penalties,
- minimum short-utterance duration.

Example checks:

```
Hola. => 0.69s
60 minutos con el tiempo. => 1.81s
¿Cuál es el peor escenario que te preocupa? => 3.14s
```

------

### 7. Dubbing quality scorecard

Implemented in:

```
foreign_whispers/evaluation.py
```

Function:

```
dubbing_quality_scorecard(...)
```

The scorecard reports:

```
timing_accuracy
intelligibility
semantic_fidelity
naturalness
overall_score
```

Example output:

```
{
  "mean_abs_duration_error_s": 1.528,
  "pct_severe_stretch": 0.0,
  "n_gap_shifts": 0,
  "n_translation_retries": 0,
  "total_cumulative_drift_s": 0.0,
  "timing_accuracy": 81.7,
  "intelligibility": 100.0,
  "semantic_fidelity": 100.0,
  "naturalness": 100.0,
  "overall_score": 95.4
}
```

------

### 8. Stitch-stage caption generation

The stitch endpoint was updated so that P5 produces both:

```
pipeline_data/api/dubbed_videos/{config}/<title>.mp4
pipeline_data/api/dubbed_captions/<title>.vtt
```

Endpoint:

```
POST /api/stitch/{video_id}?config=c-fb1074a
```

This makes dubbed captions a formal P5 pipeline artifact instead of a separate manual post-processing step.

------

## Validation

The main validation path used the integration notebook and API endpoints.

Key API checks:

```
curl http://localhost:8080/healthz

curl -X POST "http://localhost:8080/api/transcribe/GYQ5yGV_-Oc"

curl -X POST "http://localhost:8080/api/translate/GYQ5yGV_-Oc?target_language=es"

curl -X POST "http://localhost:8080/api/tts/GYQ5yGV_-Oc?config=c-fb1074a&alignment=true"

curl -X POST "http://localhost:8080/api/stitch/GYQ5yGV_-Oc?config=c-fb1074a"
```

The TTS output was checked for failed or silent segments:

```
total segments: 98
failed/silence segments: 0
```

This check was important because the TTS endpoint can return successfully even if individual segments fail and are replaced with silence.

------

## Known Limitations

### YouTube download limitation

The original YouTube `yt-dlp` download path was blocked on the Lambda Cloud GPU instance by bot checks. To complete the project, I manually uploaded the source MP4 and validated the downstream stages through the API pipeline.

I also implemented and tested a VTT-to-JSONL fallback path for downloaded YouTube captions.

------

### Pyannote diarization runtime limitation

The backend diarization endpoint, caching logic, speaker merge, and TTS voice-resolution code paths are implemented.

However, full pyannote diarization could not be fully validated in the provided Docker API runtime. I attempted to install and run `pyannote.audio`, but the dependency chain across:

```
pyannote.audio
torch
torchaudio
torchcodec
FFmpeg shared libraries
```

was not stable in the current container.

Observed blockers included:

```
torchaudio.AudioMetaData missing
torchaudio backend API compatibility issues
PyTorch weights_only checkpoint loading issues
torchcodec / FFmpeg / PyTorch ABI loading failures
```

Therefore, `/api/diarize/{video_id}` falls back gracefully instead of crashing. The downstream speaker-aware code remains implemented, but the final stable video uses the default TTS voice.

------

### Frontend diarization stage

The backend `/api/diarize/{video_id}` endpoint is implemented and validated through the API pipeline.

Full frontend integration of the diarization stage was not completed. The frontend pipeline starts from the YouTube download stage, which was blocked in the cloud environment, so the primary validation path was the integration notebook and FastAPI endpoints.

------

### Global alignment optimizer

The current system keeps the original greedy global alignment scheduler. I improved duration prediction and added a dubbing quality scorecard, but a full DP/beam-search optimizer for minimizing total drift, severe stretch count, and overlap count remains future work.

------

## Submission Notes

The project delivery is organized into two primary output variants within the submitted Google Drive folder to demonstrate pipeline stability:

### 1. Primary Output: Whisper-based

This folder contains the results from the most stable pipeline run:

- Strait of Hormuz disruption threatens to shake global economy.mp4: The final Spanish dubbed video using 98 high-quality segments.
- Strait of Hormuz disruption threatens to shake global economy.vtt: The corresponding dubbed captions.

### 2. Experimental Output: YouTube-VTT 

This folder contains the results from the stress-test run using rolling captions:

- caption_336_dubbed_video.mp4: The dubbed video generated from 336 fragmented segments.
- caption_336_dubbed_captions.vtt: The granular dubbed captions.
