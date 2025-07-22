# Configuration Tuning Guide

## Overview

The enhanced mood detection system uses a comprehensive configuration file (`mood_config.json`) to control detection sensitivity, smoothing behavior, and noise filtering. This guide explains each parameter and how to tune them for optimal performance.

## Configuration File Structure

The configuration file is organized into four main sections:

```json
{
    "thresholds": { /* Mood detection sensitivity */ },
    "smoothing": { /* Transition behavior */ },
    "noise_filtering": { /* Audio preprocessing */ },
    "performance": { /* System optimization */ }
}
```

## Threshold Parameters

### Energy Thresholds

Controls how voice energy levels map to different moods:

```json
"energy": {
    "calm_max": 0.02,
    "neutral_range": [0.02, 0.08],
    "energetic_min": 0.08,
    "excited_min": 0.15
}
```

**Parameter Explanations:**
- `calm_max`: Maximum RMS energy for calm mood (0.01-0.05)
- `neutral_range`: [min, max] energy range for neutral mood
- `energetic_min`: Minimum energy to trigger energetic mood
- `excited_min`: Minimum energy for excited/angry mood

**Tuning Tips:**
- **Quiet speakers**: Lower all thresholds by 30-50%
- **Loud speakers**: Increase thresholds by 20-40%
- **Noisy environments**: Increase all thresholds to avoid false triggers

### Spectral Thresholds

Controls frequency-based mood detection:

```json
"spectral": {
    "calm_centroid_max": 2000,
    "bright_centroid_min": 3000,
    "rolloff_thresholds": [1500, 3000, 5000],
    "mfcc_weights": [1.0, 0.8, 0.6, 0.4]
}
```

**Parameter Explanations:**
- `calm_centroid_max`: Maximum spectral centroid (Hz) for calm detection
- `bright_centroid_min`: Minimum centroid for bright/excited speech
- `rolloff_thresholds`: Frequency rolloff points for mood classification
- `mfcc_weights`: Importance weights for first 4 MFCC coefficients

**Tuning Tips:**
- **Deep voices**: Lower centroid thresholds by 500-1000 Hz
- **High voices**: Increase centroid thresholds by 500-1000 Hz
- **Poor microphone quality**: Reduce MFCC weights

### Temporal Thresholds

Controls timing-based features:

```json
"temporal": {
    "calm_zcr_max": 0.05,
    "energetic_zcr_min": 0.15,
    "tempo_thresholds": [60, 120, 180],
    "voice_activity_threshold": 0.01
}
```

**Parameter Explanations:**
- `calm_zcr_max`: Maximum zero-crossing rate for calm speech
- `energetic_zcr_min`: Minimum ZCR for energetic detection
- `tempo_thresholds`: Speech rate boundaries (words per minute equivalent)
- `voice_activity_threshold`: Minimum energy to consider as speech

## Smoothing Parameters

Controls mood transition behavior:

```json
"smoothing": {
    "transition_time": 2.0,
    "minimum_duration": 5.0,
    "confidence_threshold": 0.7,
    "buffer_size": 5,
    "hysteresis_factor": 0.1
}
```

**Parameter Explanations:**
- `transition_time`: Seconds to fade between moods (1.0-5.0)
- `minimum_duration`: Minimum time to hold each mood (3.0-10.0)
- `confidence_threshold`: Minimum confidence to trigger transition (0.5-0.9)
- `buffer_size`: Number of consecutive detections needed (3-10)
- `hysteresis_factor`: Prevents oscillation between similar moods (0.05-0.2)

**Tuning Tips:**
- **Reduce flickering**: Increase minimum_duration and buffer_size
- **Faster response**: Decrease transition_time and confidence_threshold
- **Stable performance**: Increase hysteresis_factor

## Noise Filtering Parameters

Controls audio preprocessing:

```json
"noise_filtering": {
    "noise_gate_threshold": 0.01,
    "adaptive_gain": true,
    "background_learning_rate": 0.1,
    "spectral_subtraction_factor": 2.0,
    "voice_frequency_range": [80, 8000]
}
```

**Parameter Explanations:**
- `noise_gate_threshold`: Minimum signal level to process (0.005-0.05)
- `adaptive_gain`: Enable automatic gain adjustment
- `background_learning_rate`: How quickly to adapt to noise changes (0.01-0.5)
- `spectral_subtraction_factor`: Noise reduction strength (1.0-3.0)
- `voice_frequency_range`: [low, high] Hz to focus on human voice

## Performance Parameters

Controls system optimization:

```json
"performance": {
    "max_cpu_usage": 0.8,
    "feature_complexity": "auto",
    "buffer_size": 1024,
    "processing_threads": 1
}
```

**Parameter Explanations:**
- `max_cpu_usage`: Maximum CPU utilization before scaling back (0.5-0.9)
- `feature_complexity`: "simple", "standard", "advanced", or "auto"
- `buffer_size`: Audio buffer size in samples (512, 1024, 2048)
- `processing_threads`: Number of processing threads (1-2 for Pi Zero)

## Common Tuning Scenarios

### Scenario 1: Quiet Speaker
```json
{
    "energy": {
        "calm_max": 0.01,
        "neutral_range": [0.01, 0.04],
        "energetic_min": 0.04,
        "excited_min": 0.08
    },
    "noise_filtering": {
        "noise_gate_threshold": 0.005,
        "adaptive_gain": true
    }
}
```

### Scenario 2: Noisy Environment
```json
{
    "energy": {
        "calm_max": 0.04,
        "neutral_range": [0.04, 0.12],
        "energetic_min": 0.12,
        "excited_min": 0.25
    },
    "noise_filtering": {
        "spectral_subtraction_factor": 3.0,
        "background_learning_rate": 0.2
    }
}
```

### Scenario 3: Performance-Critical (Pi Zero)
```json
{
    "performance": {
        "feature_complexity": "simple",
        "buffer_size": 512,
        "max_cpu_usage": 0.7
    },
    "smoothing": {
        "buffer_size": 3
    }
}
```

## Testing Your Configuration

1. **Start with default configuration**
2. **Run calibration mode** to establish baseline
3. **Test in target environment** with typical speech patterns
4. **Adjust one parameter group at a time**
5. **Verify mood detection accuracy** with known emotional states
6. **Check system performance** with monitoring tools

## Configuration Validation

The system automatically validates configuration parameters:
- **Range checking**: Ensures values are within acceptable bounds
- **Type validation**: Confirms correct data types
- **Dependency checking**: Verifies parameter relationships
- **Performance impact**: Warns about resource-intensive settings

Invalid configurations will fall back to safe defaults with console warnings.
#
# Performance Optimization Tips

### Hardware-Specific Optimizations

#### Raspberry Pi Zero 2 W
- **CPU**: ARM Cortex-A53 quad-core @ 1GHz
- **RAM**: 512MB
- **Optimization Strategy**: Minimize computational complexity

**Recommended Settings:**
```json
{
    "performance": {
        "feature_complexity": "simple",
        "buffer_size": 512,
        "max_cpu_usage": 0.6,
        "processing_threads": 1
    },
    "smoothing": {
        "buffer_size": 3
    },
    "noise_filtering": {
        "adaptive_gain": false,
        "spectral_subtraction_factor": 1.5
    }
}
```

#### Standard Desktop/Laptop
- **Optimization Strategy**: Balance quality and performance

**Recommended Settings:**
```json
{
    "performance": {
        "feature_complexity": "standard",
        "buffer_size": 1024,
        "max_cpu_usage": 0.8,
        "processing_threads": 2
    }
}
```

#### High-Performance Systems
- **Optimization Strategy**: Maximum quality and features

**Recommended Settings:**
```json
{
    "performance": {
        "feature_complexity": "advanced",
        "buffer_size": 2048,
        "max_cpu_usage": 0.9,
        "processing_threads": 4
    }
}
```

### Memory Optimization

#### Reduce Memory Usage
```json
{
    "performance": {
        "buffer_size": 512,
        "history_length": 5,
        "memory_limit_mb": 64
    },
    "smoothing": {
        "buffer_size": 3
    }
}
```

#### Memory Monitoring
```python
# Add to main loop for memory tracking
import psutil
import gc

def monitor_memory():
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    if memory_mb > MEMORY_LIMIT:
        gc.collect()
        return True
    return False
```

### CPU Optimization

#### Dynamic Performance Scaling
```python
# Automatically adjust complexity based on CPU usage
def adjust_performance(cpu_usage):
    if cpu_usage > 0.8:
        return "simple"
    elif cpu_usage > 0.6:
        return "standard"
    else:
        return "advanced"
```

#### Processing Pipeline Optimization
1. **Skip unnecessary features** when CPU is high
2. **Reduce buffer sizes** for faster processing
3. **Use simpler algorithms** under load
4. **Cache frequently used calculations**

### Audio Processing Optimization

#### Buffer Size Selection
- **512 samples**: Lowest latency, highest CPU usage
- **1024 samples**: Balanced performance (recommended)
- **2048 samples**: Lower CPU usage, higher latency

#### Sample Rate Optimization
```python
# Reduce sample rate for performance
SAMPLE_RATE = 22050  # Instead of 44100
# Note: May reduce detection accuracy
```

#### Feature Extraction Optimization
```python
# Disable expensive features under load
if cpu_usage > 0.7:
    features.disable_mfcc()
    features.disable_pitch_detection()
```

### Real-Time Performance Monitoring

#### Performance Metrics to Track
```python
metrics = {
    'cpu_usage': psutil.cpu_percent(),
    'memory_usage': psutil.virtual_memory().percent,
    'processing_time': time.time() - start_time,
    'buffer_underruns': audio_underrun_count,
    'detection_accuracy': mood_accuracy_score
}
```

#### Automatic Performance Adjustment
```python
def auto_optimize_performance(metrics):
    if metrics['cpu_usage'] > 80:
        reduce_feature_complexity()
    if metrics['processing_time'] > 0.02:  # 20ms limit
        increase_buffer_size()
    if metrics['buffer_underruns'] > 5:
        simplify_processing()
```

### Battery Life Optimization (Mobile/Pi)

#### Power-Saving Features
```json
{
    "performance": {
        "sleep_when_idle": true,
        "idle_timeout": 30,
        "reduce_sampling_rate": true,
        "disable_advanced_features": true
    }
}
```

#### Adaptive Processing
- **Reduce processing frequency** when no voice detected
- **Lower sample rates** during idle periods
- **Disable LED updates** when no mood changes
- **Use interrupt-driven processing** instead of polling

### Network and I/O Optimization

#### Minimize File I/O
```python
# Cache configuration in memory
config_cache = load_config_once()

# Batch log writes
log_buffer = []
if len(log_buffer) > 100:
    write_logs_batch(log_buffer)
```

#### Reduce Network Usage
- **Local processing only** (no cloud APIs)
- **Minimal telemetry** data
- **Compress log files** before transmission

### Debugging Performance Issues

#### Performance Profiling
```bash
# Profile Python code
python -m cProfile -o profile_output.prof led.py

# Analyze profile
python -c "
import pstats
p = pstats.Stats('profile_output.prof')
p.sort_stats('cumulative').print_stats(20)
"
```

#### Real-Time Monitoring
```bash
# Monitor system resources
htop

# Monitor specific process
top -p $(pgrep -f led.py)

# Monitor audio performance
python performance_monitor.py --real-time
```

#### Performance Benchmarking
```bash
# Run performance tests
python test_performance_integration.py

# Benchmark different configurations
python performance_monitor.py --benchmark-configs
```

### Configuration Examples by Use Case

#### Gaming/Streaming (Low Latency)
```json
{
    "performance": {
        "buffer_size": 512,
        "feature_complexity": "standard",
        "max_cpu_usage": 0.9
    },
    "smoothing": {
        "transition_time": 1.0,
        "minimum_duration": 3.0
    }
}
```

#### Background Operation (Low Resource)
```json
{
    "performance": {
        "buffer_size": 2048,
        "feature_complexity": "simple",
        "max_cpu_usage": 0.3
    },
    "smoothing": {
        "transition_time": 5.0,
        "minimum_duration": 10.0
    }
}
```

#### Professional Performance (High Quality)
```json
{
    "performance": {
        "buffer_size": 1024,
        "feature_complexity": "advanced",
        "max_cpu_usage": 0.8
    },
    "noise_filtering": {
        "spectral_subtraction_factor": 3.0,
        "adaptive_gain": true
    }
}
```