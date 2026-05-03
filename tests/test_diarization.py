# tests/test_diarization.py
import pytest
from foreign_whispers.diarization import diarize_audio, assign_speakers


def test_returns_empty_without_token():
    result = diarize_audio("/any/path.wav", hf_token=None)
    assert result == []


def test_returns_empty_with_empty_token():
    result = diarize_audio("/any/path.wav", hf_token="")
    assert result == []


def test_returns_empty_when_pyannote_absent(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "pyannote.audio", None)
    result = diarize_audio("/any/path.wav", hf_token="fake-token")
    assert result == []


@pytest.mark.requires_pyannote
def test_real_diarization_returns_speaker_labels(tmp_path):
    """Integration test — requires pyannote.audio and FW_HF_TOKEN env var."""
    import os
    token = os.environ.get("FW_HF_TOKEN")
    if not token:
        pytest.skip("FW_HF_TOKEN not set")
    result = diarize_audio("/path/to/sample.wav", hf_token=token)
    assert isinstance(result, list)
    for r in result:
        assert "start_s" in r and "end_s" in r and "speaker" in r


def test_assigns_speaker_by_direct_overlap():
    segments = [
        {"start": 0.0, "end": 3.0, "text": "hello"},
    ]
    diarization = [
        {"start_s": 0.0, "end_s": 3.0, "speaker": "SPEAKER_00"},
    ]

    result = assign_speakers(segments, diarization)

    assert result[0]["speaker"] == "SPEAKER_00"
    assert result[0]["text"] == "hello"


def test_assigns_speaker_with_largest_overlap():
    segments = [
        {"start": 2.0, "end": 6.0, "text": "mixed segment"},
    ]
    diarization = [
        {"start_s": 0.0, "end_s": 3.0, "speaker": "SPEAKER_00"},
        {"start_s": 3.0, "end_s": 7.0, "speaker": "SPEAKER_01"},
    ]

    result = assign_speakers(segments, diarization)

    assert result[0]["speaker"] == "SPEAKER_01"


def test_assigns_default_when_no_overlap():
    segments = [
        {"start": 10.0, "end": 12.0, "text": "no speaker here"},
    ]
    diarization = [
        {"start_s": 0.0, "end_s": 3.0, "speaker": "SPEAKER_00"},
    ]

    result = assign_speakers(segments, diarization)

    assert result[0]["speaker"] == "SPEAKER_00"


def test_assigns_default_when_diarization_empty():
    segments = [
        {"start": 0.0, "end": 2.0, "text": "hello"},
        {"start": 2.0, "end": 4.0, "text": "world"},
    ]

    result = assign_speakers(segments, [])

    assert [seg["speaker"] for seg in result] == ["SPEAKER_00", "SPEAKER_00"]