# Enhanced Mood Detection Design Document

## Overview

This design enhances the existing LED matrix mood detection system by implementing more sophisticated audio analysis, configurable parameters, smooth mood transitions, and environmental adaptability. The solution builds upon the current RMS/ZCR/spectral centroid foundation while adding advanced features for reliable cosplay performance.

## Architecture

### Current System Analysis
The existing system in `led.py` already has:
- Audio capture via sounddevice (44.1kHz, 1024 block size)
- Basic feature extraction: `extract_features(pcm_block, samplerate)`
- Simple threshold-based mood detection: `detect_mood(rms, zcr, centroid)`
- Mood-specific frame loading from `ascii_frames/{mood}/` directories

### Enhanced Architecture Components

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Audio Input   │───▶│  Feature Engine  │───▶│  Mood Detector  │
│   (Microphone)  │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │                          │
                              ▼                          ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Config Manager  │    │ Noise Filter     │    │ Transition      │
│                 │    │                  │    │ Smoother        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
                                               ┌─────────────────┐
                                               │ Frame Selector  │
                                               │                 │
                                               └─────────────────┘
```

## Components and Interfaces

### 1. Enhanced Feature Engine

**Purpose**: Extract comprehensive audio features for mood analysis

**Interface**:
```python
class EnhancedFeatureExtractor:
    def extract_features(self, pcm_block: np.ndarray, samplerate: int) -> AudioFeatures
    def get_baseline_features(self, duration_seconds: int) -> AudioFeatures
    def update_noise_profile(self, pcm_block: np.ndarray) -> None
```

**Features to Extract**:
- **Energy Features**: RMS energy, peak energy, energy variance
- **Spectral Features**: Spectral centroid, spectral rolloff, spectral flux, MFCCs (first 4 coefficients)
- **Temporal Features**: Zero-crossing rate, tempo estimation, voice activity detection
- **Pitch Features**: Fundamental frequency (F0), pitch stability, pitch range

### 2. Advanced Mood Detector

**Purpose**: Classify moods using multiple features with confidence scoring

**Interface**:
```python
class AdvancedMoodDetector:
    def detect_mood(self, features: AudioFeatures) -> MoodResult
    def calibrate_for_user(self, calibration_data: List[AudioFeatures]) -> None
    def update_thresholds(self, config: MoodConfig) -> None
```

**Mood Categories**:
- **Calm**: Low energy, stable pitch, low ZCR
- **Neutral**: Moderate energy, normal speech patterns
- **Energetic**: High energy, variable pitch, higher ZCR
- **Excited/Angry**: Very high energy, rapid changes, high spectral flux

**Detection Algorithm**:
```python
def detect_mood_advanced(features):
    # Multi-dimensional scoring
    energy_score = calculate_energy_score(features.rms, features.peak_energy)
    spectral_score = calculate_spectral_score(features.centroid, features.rolloff)
    temporal_score = calculate_temporal_score(features.zcr, features.tempo)
    pitch_score = calculate_pitch_score(features.f0, features.pitch_stability)
    
    # Weighted combination
    mood_scores = {
        'calm': weight_calm * [energy_score, spectral_score, temporal_score, pitch_score],
        'neutral': weight_neutral * [energy_score, spectral_score, temporal_score, pitch_score],
        'energetic': weight_energetic * [energy_score, spectral_score, temporal_score, pitch_score],
        'excited': weight_excited * [energy_score, spectral_score, temporal_score, pitch_score]
    }
    
    return max(mood_scores, key=mood_scores.get)
```

### 3. Configuration Manager

**Purpose**: Handle configurable parameters and user calibration

**Configuration File Structure** (`mood_config.json`):
```json
{
    "thresholds": {
        "energy": {
            "calm_max": 0.02,
            "neutral_range": [0.02, 0.08],
            "energetic_min": 0.08,
            "excited_min": 0.15
        },
        "spectral": {
            "calm_centroid_max": 2000,
            "bright_centroid_min": 3000,
            "rolloff_thresholds": [1500, 3000, 5000]
        },
        "temporal": {
            "calm_zcr_max": 0.05,
            "energetic_zcr_min": 0.15
        }
    },
    "smoothing": {
        "transition_time": 2.0,
        "minimum_duration": 5.0,
        "confidence_threshold": 0.7
    },
    "noise_filtering": {
        "noise_gate_threshold": 0.01,
        "adaptive_gain": true,
        "background_learning_rate": 0.1
    }
}
```

### 4. Transition Smoother

**Purpose**: Prevent jarring mood changes and implement smooth transitions

**Interface**:
```python
class MoodTransitionSmoother:
    def smooth_mood_transition(self, new_mood: str, confidence: float) -> str
    def should_transition(self, current_mood: str, new_mood: str, confidence: float) -> bool
    def get_transition_progress(self) -> float
```

**Smoothing Algorithm**:
- **Confidence Threshold**: Only transition if confidence > 70%
- **Minimum Duration**: Hold each mood for at least 5 seconds
- **Transition Buffer**: Collect 3-5 consecutive detections before changing
- **Gradual Transition**: Fade between frame sets over 1-3 seconds

### 5. Noise Filter

**Purpose**: Improve signal quality and adapt to environmental conditions

**Features**:
- **Spectral Subtraction**: Remove consistent background noise
- **Voice Activity Detection**: Only analyze during speech
- **Adaptive Gain Control**: Adjust sensitivity based on ambient levels
- **Multi-band Filtering**: Focus on human voice frequency range (80Hz-8kHz)

## Data Models

### AudioFeatures Class
```python
@dataclass
class AudioFeatures:
    # Energy features
    rms: float
    peak_energy: float
    energy_variance: float
    
    # Spectral features
    spectral_centroid: float
    spectral_rolloff: float
    spectral_flux: float
    mfccs: List[float]  # First 4 coefficients
    
    # Temporal features
    zero_crossing_rate: float
    tempo: float
    voice_activity: bool
    
    # Pitch features
    fundamental_freq: float
    pitch_stability: float
    pitch_range: float
    
    # Metadata
    timestamp: float
    confidence: float
```

### MoodResult Class
```python
@dataclass
class MoodResult:
    mood: str  # 'calm', 'neutral', 'energetic', 'excited'
    confidence: float  # 0.0 to 1.0
    features_used: AudioFeatures
    transition_recommended: bool
    debug_scores: Dict[str, float]
```

## Error Handling

### Audio Processing Errors
- **Microphone Disconnection**: Graceful fallback to neutral mood
- **Audio Buffer Underrun**: Skip frame and continue with last known mood
- **Feature Extraction Failure**: Use simplified fallback detection

### Configuration Errors
- **Invalid Config File**: Load default parameters and log warning
- **Calibration Failure**: Use generic thresholds with reduced confidence
- **Threshold Out of Range**: Clamp to valid ranges and warn user

### Performance Optimization
- **CPU Usage Monitoring**: Scale feature complexity based on available resources
- **Memory Management**: Circular buffers for audio history
- **Real-time Constraints**: Ensure processing completes within audio block time (23ms for 1024 samples at 44.1kHz)

## Testing Strategy

### Unit Tests
- **Feature Extraction**: Test with synthetic audio signals of known characteristics
- **Mood Detection**: Verify correct classification with controlled input features
- **Configuration Loading**: Test various config file scenarios
- **Transition Smoothing**: Verify timing and confidence thresholds

### Integration Tests
- **End-to-End Pipeline**: Test complete audio → mood → frame selection flow
- **Performance Tests**: Verify real-time performance on Pi Zero 2 W
- **Calibration Tests**: Test user calibration process with different voice types

### Hardware Tests
- **Microphone Quality**: Test with different USB microphones
- **Environmental Conditions**: Test in various noise environments
- **Long-term Stability**: Run for extended periods to check for memory leaks or drift

### User Acceptance Tests
- **Mood Accuracy**: Have users speak in different emotional states and verify detection
- **Transition Quality**: Evaluate smoothness of mood changes during natural speech
- **Calibration Effectiveness**: Test improvement after user-specific calibration