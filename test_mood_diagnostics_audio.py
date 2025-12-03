import pytest


np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
from scipy.io import wavfile

import mood_debug_tools


def test_audio_file_analysis_returns_structure(tmp_path):
    if not mood_debug_tools.ENHANCED_AVAILABLE:
        pytest.skip("Enhanced components not available for diagnostics")

    samplerate = 16000
    duration_seconds = 0.25
    t = np.linspace(0, duration_seconds, int(samplerate * duration_seconds), endpoint=False)
    audio = 0.3 * np.sin(2 * np.pi * 440 * t)

    audio_path = tmp_path / "tone.wav"
    wavfile.write(audio_path, samplerate, (audio * 32767).astype(np.int16))

    analyzer = mood_debug_tools.DiagnosticAnalyzer()
    results = analyzer._analyze_audio_file(str(audio_path))

    assert 'file_path' in results
    assert results.get('frames_analyzed', 0) > 0
    assert 'feature_stats' in results
    assert 'mood_summary' in results
    assert results['mood_summary'].get('mood_counts')
