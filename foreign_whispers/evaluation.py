"""Clip-level alignment quality metrics.

Extracted from notebooks/foreign_whispers_pipeline.ipynb (M8-align).
Imports from foreign_whispers.alignment — no other dependencies.
"""
import statistics as _stats

from foreign_whispers.alignment import (
    AlignAction,
    AlignedSegment,
    SegmentMetrics,
    decide_action,
)


def clip_evaluation_report(
    metrics: list[SegmentMetrics],
    aligned: list[AlignedSegment],
) -> dict:
    """Return a summary dict of alignment quality metrics for one clip.

    Keys:
        mean_abs_duration_error_s: Mean |predicted_tts_s - source_duration_s| per segment.
        pct_severe_stretch: % of aligned segments with stretch_factor > 1.4.
        n_gap_shifts: Number of segments resolved via gap-shift.
        n_translation_retries: Number of segments that required re-ranking.
        total_cumulative_drift_s: End-to-end drift introduced by gap-shifts.
    """
    if not metrics:
        return {
            "mean_abs_duration_error_s": 0.0,
            "pct_severe_stretch":        0.0,
            "n_gap_shifts":              0,
            "n_translation_retries":     0,
            "total_cumulative_drift_s":  0.0,
        }

    errors    = [abs(m.predicted_tts_s - m.source_duration_s) for m in metrics]
    n_severe  = sum(1 for a in aligned if a.stretch_factor > 1.4)
    n_shifted = sum(1 for a in aligned if a.action == AlignAction.GAP_SHIFT)
    n_retry   = sum(1 for m in metrics if decide_action(m) == AlignAction.REQUEST_SHORTER)
    drift     = (
        aligned[-1].scheduled_end - aligned[-1].original_end
        if aligned else 0.0
    )

    return {
        "mean_abs_duration_error_s": round(_stats.mean(errors), 3),
        "pct_severe_stretch":        round(100 * n_severe / max(len(metrics), 1), 1),
        "n_gap_shifts":              n_shifted,
        "n_translation_retries":     n_retry,
        "total_cumulative_drift_s":  round(drift, 3),
    }


def _clamp_score(value: float) -> float:
    """Clamp a score to the [0, 100] range."""
    return round(max(0.0, min(100.0, value)), 1)


def dubbing_quality_scorecard(
    metrics: list[SegmentMetrics],
    aligned: list[AlignedSegment],
    *,
    transcription_segments: list[dict] | None = None,
    translation_segments: list[dict] | None = None,
) -> dict:
    """Return a lightweight multi-dimensional dubbing quality scorecard.

    This scorecard is intentionally heuristic-based. It is designed to provide
    a runnable engineering framework for clip-level evaluation rather than a
    human-level perceptual judgment.

    Dimensions:
        timing_accuracy: Penalizes duration mismatch, severe stretch, and drift.
        intelligibility: Proxy based on non-empty translated segments.
        semantic_fidelity: Proxy based on source/target segment count agreement.
        naturalness: Penalizes aggressive stretch and gap shifts.
        overall_score: Mean of the four dimension scores.
    """
    base = clip_evaluation_report(metrics, aligned)

    mean_err = base["mean_abs_duration_error_s"]
    severe_pct = base["pct_severe_stretch"]
    drift = abs(base["total_cumulative_drift_s"])
    n_gap_shifts = base["n_gap_shifts"]

    timing_accuracy = _clamp_score(
        100.0
        - mean_err * 12.0
        - severe_pct * 0.5
        - drift * 8.0
    )

    if translation_segments:
        non_empty = sum(
            1 for seg in translation_segments
            if str(seg.get("text", "")).strip()
        )
        intelligibility = _clamp_score(100.0 * non_empty / max(len(translation_segments), 1))
    else:
        intelligibility = 100.0 if metrics else 0.0

    if transcription_segments is not None and translation_segments is not None:
        src_n = len(transcription_segments)
        tgt_n = len(translation_segments)
        count_gap = abs(src_n - tgt_n)
        semantic_fidelity = _clamp_score(100.0 - count_gap * 5.0)
    else:
        semantic_fidelity = 100.0 if metrics else 0.0

    naturalness = _clamp_score(
        100.0
        - severe_pct * 0.7
        - n_gap_shifts * 3.0
        - base["n_translation_retries"] * 2.0
    )

    overall_score = round(
        (
            timing_accuracy
            + intelligibility
            + semantic_fidelity
            + naturalness
        ) / 4.0,
        1,
    )

    return {
        **base,
        "timing_accuracy": timing_accuracy,
        "intelligibility": intelligibility,
        "semantic_fidelity": semantic_fidelity,
        "naturalness": naturalness,
        "overall_score": overall_score,
    }
    