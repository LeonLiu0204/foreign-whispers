# Foreign Whispers Dubbing Pipeline - Final Project

## Team

Xingjian Liu

## Project Overview
This project implements an extended AI video dubbing pipeline: **Download → Transcribe → Diarize → Translate → TTS → Stitch**. 

**Primary Submission:**
- **GitHub Repo:** https://github.com/LeonLiu0204/foreign-whispers.git
- **Final Output Videos (Google Drive):** [在此粘贴你的 Google Drive 链接]

---

## Key Implemented Features

1. **Manual P1 Fallback:** Resolved YouTube `yt-dlp` blocking on cloud instances by implementing a manual source MP4 verification and VTT-to-JSONL caption conversion path.
2. **Diarization API & Logic:** Implemented `POST /api/diarize/{video_id}` and `assign_speakers` logic to map transcript segments to specific speakers.
3. **Speaker-Aware TTS:** Developed a voice resolution hierarchy allowing unique reference voices per speaker label (e.g., `SPEAKER_00.wav`).
4. **Enhanced TTS Alignment:** Improved duration estimation using syllable counts and punctuation penalties (verified: 95.4 overall scorecard).
5. **Quality Scorecard:** Implemented `evaluation.py` to report timing accuracy, intelligibility, and semantic fidelity.
6. **Integrated Captioning:** Updated the Stitch stage to automatically produce `.vtt` dubbed captions as a formal pipeline artifact.

---

## Technical Challenges & Environment Notes

### 1. Cloud Instance Restrictions
The original `fw.download()` route was blocked by YouTube's bot detection on the Lambda GPU instance. I successfully implemented a local file fallback to ensure the rest of the pipeline (Stages P2-P5) remained fully functional.

### 2. Dependency & ABI Conflicts
Significant time was invested in troubleshooting severe environment conflicts within the provided Docker container:
- **Issues:** ABI mismatches between `PyTorch 2.10`, `torchaudio`, and `FFmpeg` shared libraries (`libavutil.so.58`).
- **Resolution:** Manually patched the container environment and downgraded specific libraries (Torch 2.4.1) to restore core pipeline functionality.
- **Impact:** While the Diarization backend and speaker-aware TTS logic are fully implemented, the final stable video uses a default voice fallback due to the underlying instability of the `pyannote.audio` dependency chain in the current environment.

---

## How to Reproduce
1. Start Docker: `docker compose --profile nvidia up -d`
2. Open: `notebooks/pipeline_end_to_end/pipeline_end_to_end.ipynb`
3. Execute the manual fallback cells to verify source files and trigger the API pipeline.