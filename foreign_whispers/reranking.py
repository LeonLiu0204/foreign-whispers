"""Deterministic failure analysis and translation re-ranking stubs.

The failure analysis function uses simple threshold rules derived from
SegmentMetrics.  The translation re-ranking function is a **student assignment**
— see the docstring for inputs, outputs, and implementation guidance.
"""

import dataclasses
import logging

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class TranslationCandidate:
    """A candidate translation that fits a duration budget.

    Attributes:
        text: The translated text.
        char_count: Number of characters in *text*.
        brevity_rationale: Short explanation of what was shortened.
    """
    text: str
    char_count: int
    brevity_rationale: str = ""


@dataclasses.dataclass
class FailureAnalysis:
    """Diagnostic summary of the dominant failure mode in a clip.

    Attributes:
        failure_category: One of "duration_overflow", "cumulative_drift",
            "stretch_quality", or "ok".
        likely_root_cause: One-sentence description.
        suggested_change: Most impactful next action.
    """
    failure_category: str
    likely_root_cause: str
    suggested_change: str


def analyze_failures(report: dict) -> FailureAnalysis:
    """Classify the dominant failure mode from a clip evaluation report.

    Pure heuristic — no LLM needed.  The thresholds below match the policy
    bands defined in ``alignment.decide_action``.

    Args:
        report: Dict returned by ``clip_evaluation_report()``.  Expected keys:
            ``mean_abs_duration_error_s``, ``pct_severe_stretch``,
            ``total_cumulative_drift_s``, ``n_translation_retries``.

    Returns:
        A ``FailureAnalysis`` dataclass.
    """
    mean_err = report.get("mean_abs_duration_error_s", 0.0)
    pct_severe = report.get("pct_severe_stretch", 0.0)
    drift = abs(report.get("total_cumulative_drift_s", 0.0))
    retries = report.get("n_translation_retries", 0)

    if pct_severe > 20:
        return FailureAnalysis(
            failure_category="duration_overflow",
            likely_root_cause=(
                f"{pct_severe:.0f}% of segments exceed the 1.4x stretch threshold — "
                "translated text is consistently too long for the available time window."
            ),
            suggested_change="Implement duration-aware translation re-ranking (P8).",
        )

    if drift > 3.0:
        return FailureAnalysis(
            failure_category="cumulative_drift",
            likely_root_cause=(
                f"Total drift is {drift:.1f}s — small per-segment overflows "
                "accumulate because gaps between segments are not being reclaimed."
            ),
            suggested_change="Enable gap_shift in the global alignment optimizer (P9).",
        )

    if mean_err > 0.8:
        return FailureAnalysis(
            failure_category="stretch_quality",
            likely_root_cause=(
                f"Mean duration error is {mean_err:.2f}s — segments fit within "
                "stretch limits but the stretch distorts audio quality."
            ),
            suggested_change="Lower the mild_stretch ceiling or shorten translations.",
        )

    return FailureAnalysis(
        failure_category="ok",
        likely_root_cause="No dominant failure mode detected.",
        suggested_change="Review individual outlier segments if any remain.",
    )


def get_shorter_translations(
    source_text: str,
    baseline_es: str,
    target_duration_s: float,
    context_prev: str = "",
    context_next: str = "",
) -> list[TranslationCandidate]:
    """Return shorter translation candidates that fit *target_duration_s*.

    .. admonition:: Student Assignment — Duration-Aware Translation Re-ranking

       This function is intentionally a **stub that returns an empty list**.
       Your task is to implement a strategy that produces shorter
       target-language translations when the baseline translation is too long
       for the time budget.

       **Inputs**

       ============== ======== ==================================================
       Parameter      Type     Description
       ============== ======== ==================================================
       source_text    str      Original source-language segment text
       baseline_es    str      Baseline target-language translation (from argostranslate)
       target_duration_s float Time budget in seconds for this segment
       context_prev   str      Text of the preceding segment (for coherence)
       context_next   str      Text of the following segment (for coherence)
       ============== ======== ==================================================

       **Outputs**

       A list of ``TranslationCandidate`` objects, sorted shortest first.
       Each candidate has:

       - ``text``: the shortened target-language translation
       - ``char_count``: ``len(text)``
       - ``brevity_rationale``: short note on what was changed

       **Duration heuristic**: target-language TTS produces ~15 characters/second
       (or ~4.5 syllables/second for Romance languages).  So a 3-second budget
       ≈ 45 characters.

       **Approaches to consider** (pick one or combine):

       1. **Rule-based shortening** — strip filler words, use shorter synonyms
          from a lookup table, contract common phrases
          (e.g. "en este momento" → "ahora").
       2. **Multiple translation backends** — call argostranslate with
          paraphrased input, or use a second translation model, then pick
          the shortest output that preserves meaning.
       3. **LLM re-ranking** — use an LLM (e.g. via an API) to generate
          condensed alternatives.  This was the previous approach but adds
          latency, cost, and a runtime dependency.
       4. **Hybrid** — rule-based first, fall back to LLM only for segments
          that still exceed the budget.

       **Evaluation criteria**: the caller selects the candidate whose
       ``len(text) / 15.0`` is closest to ``target_duration_s``.

    Returns:
        Empty list (stub).  Implement to return ``TranslationCandidate`` items.
    """
    # Rule-based duration-aware re-ranking.
    # We estimate Spanish TTS duration with the notebook heuristic:
    # about 15 characters per second.
    source_text = (source_text or "").strip()
    baseline_es = (baseline_es or "").strip()

    if not baseline_es:
        return []

    char_budget = max(1, int(target_duration_s * 15))

    def normalize_spaces(text: str) -> str:
        text = " ".join(text.split())
        for punct in [".", ",", ":", ";", "?", "!", ")"]:
            text = text.replace(f" {punct}", punct)
        text = text.replace("( ", "(")
        return text.strip()

    def add_candidate(
        candidates: list[TranslationCandidate],
        seen: set[str],
        text: str,
        rationale: str,
    ) -> None:
        text = normalize_spaces(text).strip(" ,;:")

        if not text:
            return

        if baseline_es.endswith((".", "?", "!")) and not text.endswith((".", "?", "!")):
            text += baseline_es[-1]

        key = text.lower()
        if key in seen:
            return

        seen.add(key)
        candidates.append(
            TranslationCandidate(
                text=text,
                char_count=len(text),
                brevity_rationale=rationale,
            )
        )

    candidates: list[TranslationCandidate] = []
    seen: set[str] = set()

    baseline_clean = normalize_spaces(baseline_es)

    # If the baseline already fits the duration budget, return it.
    if len(baseline_clean) <= char_budget:
        add_candidate(
            candidates,
            seen,
            baseline_clean,
            "Baseline translation already fits the duration budget.",
        )
        return candidates

    # 1. Remove common filler / low-information Spanish phrases.
    filler_phrases = [
        "en realidad",
        "realmente",
        "básicamente",
        "simplemente",
        "por supuesto",
        "desde luego",
        "en este momento",
        "en estos momentos",
        "en el momento actual",
        "el hecho de que",
        "lo que es",
        "lo que está",
        "que es lo que",
        "de alguna manera",
        "de forma significativa",
        "de manera significativa",
        "muy",
        "tan",
    ]

    shortened = baseline_clean
    for phrase in filler_phrases:
        shortened = shortened.replace(f" {phrase} ", " ")
        shortened = shortened.replace(f" {phrase},", ",")
        shortened = shortened.replace(f"{phrase} ", "")
        shortened = shortened.replace(phrase, "")

    add_candidate(
        candidates,
        seen,
        shortened,
        "Removed filler or low-information phrases.",
    )

    # 2. Replace verbose Spanish phrases with shorter equivalents.
    replacements = {
        "en este momento": "ahora",
        "en estos momentos": "ahora",
        "en el momento actual": "ahora",
        "con el fin de": "para",
        "a fin de": "para",
        "debido a que": "porque",
        "ya que": "porque",
        "puesto que": "porque",
        "a pesar de que": "aunque",
        "por el hecho de que": "porque",
        "tener la capacidad de": "poder",
        "tiene la capacidad de": "puede",
        "tienen la capacidad de": "pueden",
        "es necesario que": "debe",
        "ser capaz de": "poder",
        "llevar a cabo": "hacer",
        "dar lugar a": "causar",
        "como resultado": "así",
        "por esa razón": "por eso",
        "un gran número de": "muchos",
        "una gran cantidad de": "mucho",
        "cada uno de": "cada",
    }

    replaced = baseline_clean
    for long_phrase, short_phrase in replacements.items():
        replaced = replaced.replace(long_phrase, short_phrase)

    add_candidate(
        candidates,
        seen,
        replaced,
        "Replaced verbose Spanish phrases with shorter equivalents.",
    )

    # 3. Combine filler removal with phrase replacement.
    combined = shortened
    for long_phrase, short_phrase in replacements.items():
        combined = combined.replace(long_phrase, short_phrase)

    add_candidate(
        candidates,
        seen,
        combined,
        "Combined filler removal with shorter phrase substitutions.",
    )

    # 4. Fallback: truncate at word boundaries to fit the character budget.
    # This is less semantically ideal, so it is only used when rule-based
    # shortening still exceeds the target budget.
    best_so_far = min(
        [baseline_clean] + [candidate.text for candidate in candidates],
        key=len,
    )

    if len(best_so_far) > char_budget:
        words = best_so_far.split()
        kept_words: list[str] = []
        current_len = 0

        for word in words:
            extra_len = len(word) + (1 if kept_words else 0)
            if current_len + extra_len <= char_budget:
                kept_words.append(word)
                current_len += extra_len
            else:
                break

        if kept_words:
            truncated = " ".join(kept_words)
            add_candidate(
                candidates,
                seen,
                truncated,
                f"Truncated at word boundaries to fit the {char_budget}-character budget.",
            )

    if not candidates:
        add_candidate(
            candidates,
            seen,
            baseline_clean,
            "Fallback to normalized baseline translation.",
        )

    # Sort fitting candidates first, then by closeness to the duration budget.
    candidates.sort(
        key=lambda candidate: (
            candidate.char_count > char_budget,
            abs(candidate.char_count - char_budget),
            candidate.char_count,
        )
    )

    logger.info(
        "Generated %d shorter translation candidates for %.1fs budget "
        "(budget=%d chars, baseline=%d chars).",
        len(candidates),
        target_duration_s,
        char_budget,
        len(baseline_es),
    )

    return candidates