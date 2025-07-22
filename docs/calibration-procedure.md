# User Calibration Procedure

## Overview

User calibration personalizes the mood detection system to your specific voice characteristics, improving accuracy by 20-40%. The calibration process records baseline voice patterns and adjusts detection thresholds accordingly.

## When to Calibrate

**Required Calibration:**
- First-time system setup
- New user or performer
- Significant voice changes (illness, fatigue)
- Poor mood detection accuracy

**Optional Recalibration:**
- Different microphone or audio setup
- New performance environment
- Seasonal voice changes
- After system updates

## Pre-Calibration Setup

### 1. Environment Preparation
- **Quiet space**: Minimize background noise
- **Consistent positioning**: Maintain same distance from microphone
- **Stable setup**: Ensure microphone won't move during calibration
- **Normal voice condition**: Don't calibrate when sick or tired

### 2. Equipment Check
```bash
# Test microphone input
python -c "import sounddevice as sd; print(sd.query_devices())"

# Verify audio levels
python demo_debug_diagnostic.py --audio-test
```

### 3. Time Requirements
- **Quick calibration**: 2-3 minutes (basic)
- **Standard calibration**: 5-7 minutes (recommended)
- **Comprehensive calibration**: 10-15 minutes (optimal)

## Calibration Process

### Step 1: Start Calibration Mode

```bash
# Interactive calibration
python demo_user_calibration.py --interactive

# Or programmatic calibration
python -c "
from user_calibration import UserCalibration
calibrator = UserCalibration()
calibrator.start_calibration_session('your_username')
"
```

### Step 2: Baseline Recording

**Neutral Speech (60 seconds):**
- Read provided text in normal conversational tone
- Maintain consistent volume and pace
- Avoid emotional inflection

**Sample Text:**
> "The quick brown fox jumps over the lazy dog. This pangram contains every letter of the alphabet. I am speaking in my normal, everyday voice without any particular emotion. The weather today is neither particularly good nor bad. I am simply providing a baseline sample for voice calibration purposes."

### Step 3: Emotional Range Sampling

**Calm Voice (30 seconds each):**
- Speak softly and slowly
- Use soothing, relaxed tone
- Lower energy and pitch

**Sample Text:**
> "I feel peaceful and relaxed. The gentle breeze flows through the trees. Everything is calm and serene. I am speaking in a quiet, soothing voice."

**Energetic Voice (30 seconds each):**
- Speak with enthusiasm and energy
- Increase pace and volume
- Use animated tone

**Sample Text:**
> "I'm excited about this project! This is going to be amazing! I can't wait to see how well this works! The energy is fantastic!"

**Excited/Intense Voice (30 seconds each):**
- Speak with high energy and intensity
- Use dramatic emphasis
- Vary pitch and volume

**Sample Text:**
> "This is incredible! I can't believe how well this is working! The results are absolutely amazing! This is the best thing ever!"

### Step 4: Environmental Adaptation

**Background Noise Sampling (30 seconds):**
- Record ambient noise without speaking
- Capture typical environmental sounds
- Include any consistent background noise

### Step 5: Validation Testing

**Mixed Emotional Samples (2 minutes):**
- Alternate between different emotional states
- Test transition detection
- Verify calibration effectiveness

## Calibration Data Analysis

The system analyzes your voice patterns and creates a personalized profile:

### Voice Characteristics Measured
- **Energy Distribution**: Your typical volume range
- **Spectral Profile**: Frequency characteristics of your voice
- **Temporal Patterns**: Speaking rate and rhythm
- **Pitch Range**: Fundamental frequency variations
- **Emotional Markers**: Features that indicate mood changes

### Threshold Adjustments
```json
{
    "user_profile": {
        "baseline_energy": 0.045,
        "energy_range": [0.01, 0.18],
        "spectral_centroid_mean": 2150,
        "pitch_range": [120, 280],
        "speaking_rate": 145
    },
    "adjusted_thresholds": {
        "energy": {
            "calm_max": 0.025,
            "neutral_range": [0.025, 0.09],
            "energetic_min": 0.09,
            "excited_min": 0.16
        }
    }
}
```

## Calibration Quality Assessment

### Automatic Quality Checks
- **Sample Duration**: Sufficient recording time for each mood
- **Signal Quality**: Adequate signal-to-noise ratio
- **Consistency**: Stable voice characteristics within each mood
- **Range Coverage**: Adequate emotional range representation

### Quality Indicators
```
Calibration Quality Report:
✓ Baseline samples: 98% quality
✓ Calm voice range: 95% quality  
✓ Energetic range: 92% quality
✓ Excited range: 89% quality
✓ Environmental noise: 94% quality
Overall calibration quality: 94% (Excellent)
```

### Quality Improvement Tips
- **Low baseline quality**: Re-record in quieter environment
- **Poor emotional range**: Exaggerate emotional differences
- **Inconsistent samples**: Maintain steady voice characteristics
- **Environmental issues**: Reduce background noise

## Post-Calibration Testing

### 1. Immediate Validation
```bash
# Test calibrated system
python demo_enhanced_features.py --test-calibration

# Compare before/after accuracy
python test_user_calibration.py --validation-test
```

### 2. Real-World Testing
- **Perform typical speech patterns**
- **Test in target environment**
- **Verify mood detection accuracy**
- **Check transition smoothness**

### 3. Fine-Tuning
If accuracy is still suboptimal:
- **Adjust confidence thresholds**
- **Modify smoothing parameters**
- **Re-calibrate specific emotional ranges**
- **Update environmental noise profile**

## Calibration Data Management

### Storage Location
```
calibration_data/
├── {username}_calibration.json     # Main calibration data
├── {username}_samples/             # Raw audio samples
│   ├── baseline_*.wav
│   ├── calm_*.wav
│   ├── energetic_*.wav
│   └── excited_*.wav
└── {username}_validation.json      # Validation results
```

### Backup and Restore
```bash
# Backup calibration
cp calibration_data/{username}_calibration.json backup/

# Restore calibration
cp backup/{username}_calibration.json calibration_data/

# Reset to defaults
python user_calibration.py --reset-user {username}
```

### Multiple User Support
```bash
# Switch between users
python user_calibration.py --load-user alice
python user_calibration.py --load-user bob

# List available calibrations
python user_calibration.py --list-users
```

## Troubleshooting Calibration Issues

### Common Problems

**Problem**: Calibration fails to start
**Solution**: 
- Check microphone permissions
- Verify audio device availability
- Ensure sufficient disk space

**Problem**: Poor calibration quality
**Solution**:
- Reduce background noise
- Speak closer to microphone
- Use more exaggerated emotional differences

**Problem**: System doesn't use calibration
**Solution**:
- Verify calibration file exists
- Check file permissions
- Restart system after calibration

**Problem**: Calibration makes detection worse
**Solution**:
- Reset to default thresholds
- Re-calibrate with better samples
- Adjust confidence thresholds

### Advanced Calibration Options

**Custom Emotional Categories:**
```python
# Add custom mood categories
calibrator.add_custom_mood('whisper', whisper_samples)
calibrator.add_custom_mood('shouting', shouting_samples)
```

**Environmental Profiles:**
```python
# Create environment-specific calibrations
calibrator.create_environment_profile('indoor_quiet')
calibrator.create_environment_profile('outdoor_windy')
```

**Batch Calibration:**
```bash
# Calibrate multiple users from recorded samples
python user_calibration.py --batch-calibrate samples_directory/
```

## Calibration Maintenance

### Regular Updates
- **Monthly**: Quick validation test
- **Quarterly**: Environmental noise update
- **Annually**: Full recalibration

### Performance Monitoring
```bash
# Check calibration effectiveness
python performance_monitor.py --calibration-stats

# Generate calibration report
python user_calibration.py --generate-report {username}
```

The calibration system continuously learns and adapts, but periodic maintenance ensures optimal performance.