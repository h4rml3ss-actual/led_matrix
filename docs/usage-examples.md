# Usage Examples and Integration Guide

## Overview

This document provides practical examples of how to use the enhanced mood detection system in various scenarios, from basic setup to advanced integration.

## Basic Usage Examples

### 1. Quick Start with Default Settings

```bash
# Start with default configuration
python led.py

# The system will:
# - Load default mood_config.json
# - Initialize audio input
# - Begin mood detection
# - Display appropriate LED frames
```

### 2. First-Time Setup with Calibration

```bash
# Step 1: Create user calibration
python demo_user_calibration.py --interactive --user "your_name"

# Step 2: Test calibrated system
python led.py --user "your_name"

# Step 3: Fine-tune if needed
python mood_config.py --wizard
```

### 3. Testing and Validation

```bash
# Test mood detection accuracy
python test_enhanced_integration.py --validation

# Monitor real-time performance
python demo_performance_monitoring.py --real-time

# Debug mood detection decisions
python demo_debug_diagnostic.py --mood-analysis
```

## Configuration Examples

### Scenario 1: Quiet Library Performance

**Use Case**: Performing in a quiet environment with soft speech

```bash
# Copy optimized configuration
cp mood_config_examples/quiet_speaker.json mood_config.json

# Run calibration for quiet speech
python demo_user_calibration.py --quiet-mode

# Start system
python led.py
```

**Expected Behavior**:
- Detects subtle voice changes
- Higher sensitivity to low energy
- Longer smoothing to avoid noise triggers

### Scenario 2: Convention/Crowded Event

**Use Case**: Performing at a noisy convention with background chatter

```bash
# Use noisy environment configuration
cp mood_config_examples/noisy_environment.json mood_config.json

# Calibrate with background noise
python demo_user_calibration.py --noisy-environment

# Start with enhanced noise filtering
python led.py --noise-filter-aggressive
```

**Expected Behavior**:
- Filters out background conversations
- Higher thresholds to avoid false triggers
- Focus on primary voice source

### Scenario 3: High-Energy Performance

**Use Case**: Energetic cosplay performance with dramatic voice changes

```bash
# Use loud speaker configuration
cp mood_config_examples/loud_speaker.json mood_config.json

# Calibrate for high energy range
python demo_user_calibration.py --high-energy

# Start with fast transitions
python led.py --fast-response
```

**Expected Behavior**:
- Quick response to mood changes
- Higher energy thresholds
- Shorter smoothing for dynamic performance

## Advanced Integration Examples

### 1. Custom Mood Categories

```python
# custom_moods.py
from advanced_mood_detector import AdvancedMoodDetector
from enhanced_audio_features import AudioFeatures

class CustomMoodDetector(AdvancedMoodDetector):
    def __init__(self):
        super().__init__()
        self.add_custom_mood('whisper', self.detect_whisper)
        self.add_custom_mood('shouting', self.detect_shouting)
    
    def detect_whisper(self, features: AudioFeatures) -> float:
        # Custom logic for whisper detection
        if features.rms < 0.005 and features.spectral_centroid < 1500:
            return min(1.0, (0.005 - features.rms) * 200)
        return 0.0
    
    def detect_shouting(self, features: AudioFeatures) -> float:
        # Custom logic for shouting detection
        if features.rms > 0.3 and features.zero_crossing_rate > 0.2:
            return min(1.0, features.rms * 3)
        return 0.0

# Usage
detector = CustomMoodDetector()
mood_result = detector.detect_mood(audio_features)
```

### 2. External API Integration

```python
# api_integration.py
import requests
from mood_transition_smoother import MoodTransitionSmoother

class APIIntegratedMoodSystem:
    def __init__(self, api_endpoint):
        self.api_endpoint = api_endpoint
        self.smoother = MoodTransitionSmoother()
    
    def send_mood_update(self, mood, confidence):
        """Send mood updates to external API"""
        payload = {
            'mood': mood,
            'confidence': confidence,
            'timestamp': time.time()
        }
        try:
            response = requests.post(self.api_endpoint, json=payload, timeout=1)
            return response.status_code == 200
        except requests.RequestException:
            return False
    
    def process_mood_change(self, new_mood, confidence):
        """Process mood change with API notification"""
        if self.smoother.should_transition(self.current_mood, new_mood, confidence):
            self.current_mood = new_mood
            self.send_mood_update(new_mood, confidence)
            return True
        return False

# Usage
api_system = APIIntegratedMoodSystem('https://api.example.com/mood')
api_system.process_mood_change('excited', 0.85)
```

### 3. Multi-User System

```python
# multi_user_system.py
from user_calibration import UserCalibration
import json

class MultiUserMoodSystem:
    def __init__(self):
        self.users = {}
        self.current_user = None
        self.calibration = UserCalibration()
    
    def add_user(self, username, calibration_file=None):
        """Add a new user to the system"""
        if calibration_file:
            with open(calibration_file, 'r') as f:
                user_config = json.load(f)
        else:
            user_config = self.calibration.create_default_profile()
        
        self.users[username] = user_config
    
    def switch_user(self, username):
        """Switch to a different user profile"""
        if username in self.users:
            self.current_user = username
            self.load_user_configuration(self.users[username])
            return True
        return False
    
    def calibrate_current_user(self):
        """Run calibration for current user"""
        if self.current_user:
            calibration_data = self.calibration.run_calibration(self.current_user)
            self.users[self.current_user] = calibration_data
            return True
        return False

# Usage
multi_user = MultiUserMoodSystem()
multi_user.add_user('alice', 'calibration_data/alice_calibration.json')
multi_user.add_user('bob', 'calibration_data/bob_calibration.json')
multi_user.switch_user('alice')
```

## Performance Monitoring Examples

### 1. Real-Time Performance Dashboard

```python
# performance_dashboard.py
import time
import psutil
from performance_monitor import PerformanceMonitor

class RealTimeDashboard:
    def __init__(self):
        self.monitor = PerformanceMonitor()
        self.metrics_history = []
    
    def update_metrics(self):
        """Collect current performance metrics"""
        metrics = {
            'timestamp': time.time(),
            'cpu_usage': psutil.cpu_percent(),
            'memory_usage': psutil.virtual_memory().percent,
            'processing_time': self.monitor.get_avg_processing_time(),
            'mood_accuracy': self.monitor.get_accuracy_score(),
            'audio_quality': self.monitor.get_audio_quality()
        }
        self.metrics_history.append(metrics)
        
        # Keep only last 100 measurements
        if len(self.metrics_history) > 100:
            self.metrics_history.pop(0)
        
        return metrics
    
    def print_dashboard(self):
        """Print real-time dashboard"""
        metrics = self.update_metrics()
        print(f"\n{'='*50}")
        print(f"Mood Detection Performance Dashboard")
        print(f"{'='*50}")
        print(f"CPU Usage:        {metrics['cpu_usage']:.1f}%")
        print(f"Memory Usage:     {metrics['memory_usage']:.1f}%")
        print(f"Processing Time:  {metrics['processing_time']:.2f}ms")
        print(f"Mood Accuracy:    {metrics['mood_accuracy']:.1f}%")
        print(f"Audio Quality:    {metrics['audio_quality']:.1f}%")
        print(f"{'='*50}")

# Usage
dashboard = RealTimeDashboard()
while True:
    dashboard.print_dashboard()
    time.sleep(1)
```

### 2. Automated Performance Optimization

```python
# auto_optimizer.py
from mood_config import MoodConfig

class AutoPerformanceOptimizer:
    def __init__(self):
        self.config = MoodConfig()
        self.performance_history = []
    
    def optimize_for_performance(self, target_cpu_usage=0.7):
        """Automatically optimize configuration for target performance"""
        current_cpu = psutil.cpu_percent()
        
        if current_cpu > target_cpu_usage:
            # Reduce computational complexity
            self.config.update({
                'performance': {
                    'feature_complexity': 'simple',
                    'buffer_size': 512
                },
                'smoothing': {
                    'buffer_size': 3
                },
                'noise_filtering': {
                    'adaptive_gain': False
                }
            })
        elif current_cpu < target_cpu_usage * 0.6:
            # Increase quality if resources available
            self.config.update({
                'performance': {
                    'feature_complexity': 'standard',
                    'buffer_size': 1024
                },
                'noise_filtering': {
                    'adaptive_gain': True
                }
            })
        
        return self.config.save()

# Usage
optimizer = AutoPerformanceOptimizer()
optimizer.optimize_for_performance(target_cpu_usage=0.6)
```

## Debugging and Troubleshooting Examples

### 1. Comprehensive Debug Session

```bash
# Full system diagnostic
python demo_debug_diagnostic.py --full-check

# Audio system analysis
python demo_debug_diagnostic.py --audio-analysis

# Mood detection debugging
python demo_debug_diagnostic.py --mood-debug --duration 60

# Performance profiling
python demo_debug_diagnostic.py --performance-profile
```

### 2. Custom Debug Tools

```python
# debug_tools.py
import matplotlib.pyplot as plt
import numpy as np

class MoodDebugger:
    def __init__(self):
        self.feature_history = []
        self.mood_history = []
    
    def log_detection(self, features, mood_result):
        """Log detection data for analysis"""
        self.feature_history.append({
            'timestamp': time.time(),
            'rms': features.rms,
            'spectral_centroid': features.spectral_centroid,
            'zero_crossing_rate': features.zero_crossing_rate
        })
        
        self.mood_history.append({
            'timestamp': time.time(),
            'mood': mood_result.mood,
            'confidence': mood_result.confidence
        })
    
    def plot_analysis(self):
        """Generate analysis plots"""
        if not self.feature_history:
            return
        
        timestamps = [f['timestamp'] for f in self.feature_history]
        rms_values = [f['rms'] for f in self.feature_history]
        centroid_values = [f['spectral_centroid'] for f in self.feature_history]
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(3, 1, 1)
        plt.plot(timestamps, rms_values)
        plt.title('RMS Energy Over Time')
        plt.ylabel('RMS')
        
        plt.subplot(3, 1, 2)
        plt.plot(timestamps, centroid_values)
        plt.title('Spectral Centroid Over Time')
        plt.ylabel('Frequency (Hz)')
        
        plt.subplot(3, 1, 3)
        mood_timestamps = [m['timestamp'] for m in self.mood_history]
        moods = [m['mood'] for m in self.mood_history]
        confidences = [m['confidence'] for m in self.mood_history]
        
        plt.scatter(mood_timestamps, moods, c=confidences, cmap='viridis')
        plt.title('Detected Moods Over Time')
        plt.ylabel('Mood')
        plt.colorbar(label='Confidence')
        
        plt.tight_layout()
        plt.show()

# Usage
debugger = MoodDebugger()
# ... during mood detection loop ...
debugger.log_detection(audio_features, mood_result)
# ... after session ...
debugger.plot_analysis()
```

## Integration with External Systems

### 1. Home Automation Integration

```python
# home_automation.py
import paho.mqtt.client as mqtt

class HomeAutomationBridge:
    def __init__(self, mqtt_broker='localhost'):
        self.client = mqtt.Client()
        self.client.connect(mqtt_broker, 1883, 60)
    
    def publish_mood(self, mood, confidence):
        """Publish mood to home automation system"""
        payload = {
            'mood': mood,
            'confidence': confidence,
            'timestamp': time.time()
        }
        self.client.publish('cosplay/mood', json.dumps(payload))
    
    def control_smart_lights(self, mood):
        """Control smart lights based on mood"""
        light_configs = {
            'calm': {'color': 'blue', 'brightness': 30},
            'neutral': {'color': 'white', 'brightness': 50},
            'energetic': {'color': 'orange', 'brightness': 80},
            'excited': {'color': 'red', 'brightness': 100}
        }
        
        if mood in light_configs:
            config = light_configs[mood]
            self.client.publish('lights/color', config['color'])
            self.client.publish('lights/brightness', str(config['brightness']))

# Usage
home_bridge = HomeAutomationBridge()
home_bridge.publish_mood('excited', 0.92)
home_bridge.control_smart_lights('excited')
```

### 2. Streaming Integration

```python
# streaming_integration.py
import websocket
import json

class StreamingOverlay:
    def __init__(self, websocket_url):
        self.ws = websocket.WebSocket()
        self.ws.connect(websocket_url)
    
    def update_overlay(self, mood, confidence):
        """Update streaming overlay with current mood"""
        overlay_data = {
            'type': 'mood_update',
            'mood': mood,
            'confidence': confidence,
            'color': self.get_mood_color(mood),
            'animation': self.get_mood_animation(mood)
        }
        self.ws.send(json.dumps(overlay_data))
    
    def get_mood_color(self, mood):
        """Get color for mood visualization"""
        colors = {
            'calm': '#4A90E2',
            'neutral': '#7ED321',
            'energetic': '#F5A623',
            'excited': '#D0021B'
        }
        return colors.get(mood, '#FFFFFF')
    
    def get_mood_animation(self, mood):
        """Get animation type for mood"""
        animations = {
            'calm': 'fade',
            'neutral': 'steady',
            'energetic': 'pulse',
            'excited': 'flash'
        }
        return animations.get(mood, 'steady')

# Usage
overlay = StreamingOverlay('ws://localhost:8080/overlay')
overlay.update_overlay('excited', 0.88)
```

## Testing and Validation Examples

### 1. Automated Testing Suite

```bash
# Run comprehensive test suite
python test_comprehensive_suite.py

# Test specific components
python test_enhanced_audio_features.py
python test_advanced_mood_detector.py
python test_mood_transition_smoother.py

# Performance testing
python test_performance_integration.py --pi-zero-test
python test_pi_zero_performance.py
```

### 2. User Acceptance Testing

```python
# user_acceptance_test.py
class UserAcceptanceTest:
    def __init__(self):
        self.test_scenarios = [
            {'name': 'Calm Speech', 'expected_mood': 'calm'},
            {'name': 'Excited Speech', 'expected_mood': 'excited'},
            {'name': 'Normal Conversation', 'expected_mood': 'neutral'},
            {'name': 'Energetic Performance', 'expected_mood': 'energetic'}
        ]
    
    def run_acceptance_test(self, duration=30):
        """Run user acceptance test"""
        print("User Acceptance Test")
        print("===================")
        
        for scenario in self.test_scenarios:
            print(f"\nScenario: {scenario['name']}")
            print(f"Expected mood: {scenario['expected_mood']}")
            print("Please speak in the indicated style for 30 seconds...")
            
            # Record and analyze
            detected_moods = self.record_and_analyze(duration)
            accuracy = self.calculate_accuracy(detected_moods, scenario['expected_mood'])
            
            print(f"Accuracy: {accuracy:.1f}%")
            print(f"Detected moods: {detected_moods}")
    
    def calculate_accuracy(self, detected_moods, expected_mood):
        """Calculate detection accuracy"""
        correct_detections = sum(1 for mood in detected_moods if mood == expected_mood)
        return (correct_detections / len(detected_moods)) * 100 if detected_moods else 0

# Usage
test = UserAcceptanceTest()
test.run_acceptance_test()
```

These examples demonstrate the flexibility and power of the enhanced mood detection system across various use cases and integration scenarios.