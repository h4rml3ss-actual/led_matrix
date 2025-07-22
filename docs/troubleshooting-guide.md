# Troubleshooting Guide

## Overview

This guide helps diagnose and resolve common issues with the enhanced mood detection system. Issues are organized by category with step-by-step solutions.

## Quick Diagnostic Commands

```bash
# System health check
python demo_debug_diagnostic.py --full-check

# Audio system test
python demo_debug_diagnostic.py --audio-test

# Performance monitoring
python demo_performance_monitoring.py --real-time

# Configuration validation
python mood_config.py --validate
```

## Audio Input Issues

### Problem: No Audio Input Detected

**Symptoms:**
- System shows "No microphone detected"
- Audio levels always zero
- No mood changes detected

**Diagnosis:**
```bash
# List available audio devices
python -c "import sounddevice as sd; print(sd.query_devices())"

# Test microphone access
python -c "
import sounddevice as sd
import numpy as np
try:
    data = sd.rec(1024, samplerate=44100, channels=1)
    sd.wait()
    print(f'Audio captured: RMS = {np.sqrt(np.mean(data**2)):.4f}')
except Exception as e:
    print(f'Audio error: {e}')
"
```

**Solutions:**
1. **Check microphone connection**
   - Verify USB microphone is plugged in
   - Try different USB port
   - Check microphone LED (if present)

2. **Verify permissions**
   ```bash
   # On macOS
   sudo chmod 755 /dev/audio*
   
   # Check system audio permissions
   # System Preferences > Security & Privacy > Microphone
   ```

3. **Update audio configuration**
   ```bash
   # Set default audio device
   export PULSE_DEVICE="your_microphone_name"
   
   # Or specify in code
   python led.py --audio-device "USB Audio Device"
   ```

### Problem: Poor Audio Quality

**Symptoms:**
- Crackling or distorted audio
- Inconsistent mood detection
- High background noise

**Diagnosis:**
```bash
# Check audio quality metrics
python demo_debug_diagnostic.py --audio-quality

# Monitor signal-to-noise ratio
python noise_filter.py --analyze-noise
```

**Solutions:**
1. **Adjust microphone gain**
   ```bash
   # Reduce gain if clipping
   alsamixer  # Linux
   # Or use system audio settings
   ```

2. **Improve microphone positioning**
   - Position 6-12 inches from mouth
   - Avoid breathing directly on microphone
   - Use pop filter if available

3. **Enable noise filtering**
   ```json
   // In mood_config.json
   "noise_filtering": {
       "spectral_subtraction_factor": 2.5,
       "adaptive_gain": true,
       "noise_gate_threshold": 0.02
   }
   ```

## Mood Detection Issues

### Problem: Inaccurate Mood Detection

**Symptoms:**
- Wrong moods detected consistently
- No mood changes despite voice changes
- Random mood switching

**Diagnosis:**
```bash
# Test with known emotional samples
python test_enhanced_integration.py --mood-accuracy

# Check feature extraction
python demo_debug_diagnostic.py --feature-analysis

# Validate thresholds
python advanced_mood_detector.py --threshold-test
```

**Solutions:**
1. **Run user calibration**
   ```bash
   python demo_user_calibration.py --interactive
   ```

2. **Adjust detection thresholds**
   ```json
   // For quiet speakers
   "energy": {
       "calm_max": 0.01,
       "neutral_range": [0.01, 0.04],
       "energetic_min": 0.04
   }
   
   // For loud speakers  
   "energy": {
       "calm_max": 0.04,
       "neutral_range": [0.04, 0.12],
       "energetic_min": 0.12
   }
   ```

3. **Check environmental factors**
   - Test in quieter environment
   - Verify consistent microphone distance
   - Update noise profile

### Problem: Mood Flickering/Instability

**Symptoms:**
- Rapid switching between moods
- Unstable mood detection
- Jerky LED animations

**Diagnosis:**
```bash
# Monitor transition behavior
python mood_transition_smoother.py --debug-transitions

# Check confidence scores
python demo_debug_diagnostic.py --confidence-analysis
```

**Solutions:**
1. **Increase smoothing parameters**
   ```json
   "smoothing": {
       "minimum_duration": 7.0,
       "confidence_threshold": 0.8,
       "buffer_size": 7,
       "hysteresis_factor": 0.15
   }
   ```

2. **Improve signal quality**
   - Reduce background noise
   - Use better microphone
   - Adjust microphone gain

3. **Update calibration**
   ```bash
   python demo_user_calibration.py --recalibrate
   ```

## Performance Issues

### Problem: High CPU Usage

**Symptoms:**
- System becomes slow/unresponsive
- Audio dropouts or delays
- Overheating on Raspberry Pi

**Diagnosis:**
```bash
# Monitor CPU usage
python demo_performance_monitoring.py --cpu-monitor

# Check processing times
python performance_monitor.py --timing-analysis
```

**Solutions:**
1. **Reduce feature complexity**
   ```json
   "performance": {
       "feature_complexity": "simple",
       "max_cpu_usage": 0.7,
       "buffer_size": 512
   }
   ```

2. **Optimize audio settings**
   ```python
   # Reduce sample rate if possible
   SAMPLE_RATE = 22050  # Instead of 44100
   BLOCK_SIZE = 512     # Instead of 1024
   ```

3. **Disable unnecessary features**
   ```json
   "noise_filtering": {
       "adaptive_gain": false,
       "spectral_subtraction_factor": 1.0
   }
   ```

### Problem: Memory Issues

**Symptoms:**
- System crashes after extended use
- "Out of memory" errors
- Gradual performance degradation

**Diagnosis:**
```bash
# Monitor memory usage
python demo_performance_monitoring.py --memory-monitor

# Check for memory leaks
python test_performance_integration.py --memory-leak-test
```

**Solutions:**
1. **Reduce buffer sizes**
   ```json
   "performance": {
       "buffer_size": 512,
       "history_length": 10
   }
   ```

2. **Enable garbage collection**
   ```python
   import gc
   gc.collect()  # Add to main loop
   ```

3. **Restart periodically**
   ```bash
   # Add to cron for automatic restart
   0 */6 * * * /usr/bin/pkill -f led.py && /usr/bin/python3 /path/to/led.py
   ```

## Configuration Issues

### Problem: Configuration Not Loading

**Symptoms:**
- System uses default settings
- Configuration changes ignored
- "Config file not found" errors

**Diagnosis:**
```bash
# Check file existence and permissions
ls -la mood_config.json

# Validate JSON syntax
python -m json.tool mood_config.json

# Test configuration loading
python mood_config.py --test-load
```

**Solutions:**
1. **Verify file location**
   ```bash
   # Ensure config file is in correct location
   ls -la mood_config.json
   
   # Check current working directory
   pwd
   ```

2. **Fix JSON syntax**
   ```bash
   # Validate and fix JSON
   python -c "
   import json
   with open('mood_config.json', 'r') as f:
       config = json.load(f)
   print('Configuration valid')
   "
   ```

3. **Reset to defaults**
   ```bash
   # Create default configuration
   python mood_config.py --create-default
   ```

### Problem: Invalid Configuration Values

**Symptoms:**
- System warnings about invalid parameters
- Unexpected behavior
- Fallback to default values

**Diagnosis:**
```bash
# Validate all parameters
python mood_config.py --validate --verbose

# Check parameter ranges
python mood_config.py --check-ranges
```

**Solutions:**
1. **Fix parameter ranges**
   ```json
   // Ensure values are within valid ranges
   "energy": {
       "calm_max": 0.02,        // 0.01-0.05
       "energetic_min": 0.08    // Must be > calm_max
   }
   ```

2. **Use configuration wizard**
   ```bash
   python mood_config.py --wizard
   ```

## Hardware-Specific Issues

### Raspberry Pi Zero 2 W Issues

**Problem: Performance Too Slow**
```json
// Optimized Pi Zero configuration
{
    "performance": {
        "feature_complexity": "simple",
        "buffer_size": 512,
        "max_cpu_usage": 0.6,
        "processing_threads": 1
    },
    "smoothing": {
        "buffer_size": 3
    }
}
```

**Problem: USB Audio Issues**
```bash
# Add to /boot/config.txt
dtoverlay=dwc2
dwc_otg.fiq_fix_enable=1

# Increase USB current
max_usb_current=1
```

### macOS-Specific Issues

**Problem: Microphone Permission Denied**
1. System Preferences > Security & Privacy > Privacy > Microphone
2. Add Terminal or Python to allowed applications
3. Restart terminal/application

**Problem: Audio Device Selection**
```bash
# List audio devices
system_profiler SPAudioDataType

# Set specific device
export PULSE_DEVICE="Built-in Microphone"
```

## Error Recovery

### Automatic Recovery Features

The system includes several automatic recovery mechanisms:

1. **Audio Input Recovery**
   - Automatically reconnects if microphone disconnects
   - Falls back to neutral mood if audio fails
   - Retries audio initialization

2. **Configuration Recovery**
   - Uses default values for invalid parameters
   - Automatically creates missing config files
   - Validates and corrects parameter ranges

3. **Performance Recovery**
   - Reduces feature complexity under high CPU load
   - Increases buffer sizes if processing can't keep up
   - Disables non-essential features when needed

### Manual Recovery Commands

```bash
# Reset everything to defaults
python mood_config.py --reset-all

# Clear calibration data
rm -rf calibration_data/*

# Restart with minimal configuration
python led.py --safe-mode

# Force audio device reset
python led.py --reset-audio
```

## Getting Help

### Debug Information Collection

When reporting issues, collect this information:

```bash
# System information
python demo_debug_diagnostic.py --system-info > debug_info.txt

# Configuration dump
python mood_config.py --dump >> debug_info.txt

# Recent error logs
tail -n 50 mood_detection_errors.log >> debug_info.txt

# Performance metrics
python performance_monitor.py --report >> debug_info.txt
```

### Log File Locations

- **Error logs**: `mood_detection_errors.log`
- **Performance logs**: `performance_monitor.log`
- **Debug logs**: `debug_output.log`
- **Calibration logs**: `calibration_data/calibration.log`

### Common Log Messages

**"Feature extraction failed"**
- Audio input issue or processing overload
- Try reducing feature complexity

**"Configuration validation failed"**
- Invalid parameter values in config file
- Run configuration validator

**"Calibration data corrupted"**
- Calibration file damaged
- Re-run user calibration

**"Performance threshold exceeded"**
- System overloaded
- Reduce processing complexity

### Support Resources

1. **Configuration examples**: `mood_config_example.json`
2. **Test scripts**: `test_*.py` files
3. **Demo applications**: `demo_*.py` files
4. **Performance monitoring**: `performance_monitor.py`

For persistent issues, run the full diagnostic suite and review the generated reports.