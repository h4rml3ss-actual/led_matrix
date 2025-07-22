#!/usr/bin/env python3
"""
Simple integration patch for led.py - adds enhanced mood detection
while maintaining full backward compatibility.
"""

# Import original led.py functions
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import everything from original led.py
from led import *

# Enhanced components - graceful fallback
try:
    from enhanced_audio_features import EnhancedFeatureExtractor
    from advanced_mood_detector import AdvancedMoodDetector
    from mood_transition_smoother import MoodTransitionSmoother
    from user_calibration import get_calibrated_detector
    ENHANCED_AVAILABLE = True
    print("Enhanced mood detection components available")
except ImportError as e:
    print(f"Enhanced components not available: {e}")
    ENHANCED_AVAILABLE = False

# Enhanced processing globals
enhanced_processor = None
enhanced_enabled = False

def init_enhanced_system(user_id=None):
    """Initialize enhanced system if available."""
    global enhanced_processor, enhanced_enabled
    
    if not ENHANCED_AVAILABLE:
        return False
    
    try:
        from enhanced_audio_features import EnhancedFeatureExtractor
        from advanced_mood_detector import AdvancedMoodDetector
        from mood_transition_smoother import MoodTransitionSmoother
        from user_calibration import get_calibrated_detector
        
        # Create enhanced processor
        feature_extractor = EnhancedFeatureExtractor(
            samplerate=44100, frame_size=1024, enable_noise_filtering=True
        )
        
        if user_id:
            try:
                mood_detector = get_calibrated_detector(user_id)
                print(f"Using calibrated detector for user: {user_id}")
            except:
                mood_detector = AdvancedMoodDetector()
        else:
            mood_detector = AdvancedMoodDetector()
        
        transition_smoother = MoodTransitionSmoother()
        
        enhanced_processor = {
            'feature_extractor': feature_extractor,
            'mood_detector': mood_detector,
            'transition_smoother': transition_smoother,
            'latest_mood': 'neutral',
            'latest_confidence': 0.5
        }
        
        enhanced_enabled = True
        print("Enhanced mood detection initialized")
        return True
        
    except Exception as e:
        print(f"Enhanced initialization failed: {e}")
        return False

def enhanced_audio_callback(indata, frames, time_info, status):
    """Enhanced audio callback that extends the original."""
    global enhanced_processor
    
    # Call original callback first
    audio_callback(indata, frames, time_info, status)
    
    # Add enhanced processing
    if enhanced_enabled and enhanced_processor:
        try:
            # Extract features
            features = enhanced_processor['feature_extractor'].extract_features(
                indata.flatten(), timestamp=time.time()
            )
            
            # Detect mood
            mood_result = enhanced_processor['mood_detector'].detect_mood(features)
            
            # Apply smoothing
            smoothed_mood = enhanced_processor['transition_smoother'].smooth_transition(
                mood_result.mood, mood_result.confidence
            )
            
            # Store results
            enhanced_processor['latest_mood'] = smoothed_mood
            enhanced_processor['latest_confidence'] = mood_result.confidence
            
        except Exception as e:
            print(f"Enhanced processing error: {e}")

def get_enhanced_mood():
    """Get current mood from enhanced system."""
    if enhanced_enabled and enhanced_processor:
        return enhanced_processor['latest_mood'], enhanced_processor['latest_confidence']
    else:
        # Fallback to original detection
        pcm_block = latest_audio_block[0]
        if pcm_block is not None:
            rms_v, zcr_v, cent_v = extract_features(pcm_block.flatten(), 44100)
            mood = detect_mood(rms_v, zcr_v, cent_v)
            return mood, 0.7
        return "neutral", 0.5

def enhanced_main():
    """Enhanced main function."""
    global enhanced_enabled
    
    # Check for user calibration
    user_id = None
    if "--user" in sys.argv:
        try:
            user_idx = sys.argv.index("--user") + 1
            if user_idx < len(sys.argv):
                user_id = sys.argv[user_idx]
        except:
            pass
    
    # Initialize enhanced system
    enhanced_enabled = init_enhanced_system(user_id)
    
    print("=" * 50)
    print("LED Mood Detection System")
    print(f"Enhanced Mode: {'Enabled' if enhanced_enabled else 'Disabled'}")
    if user_id:
        print(f"User Calibration: {user_id}")
    print("=" * 50)
    
    # Patch the audio callback
    import led
    if enhanced_enabled:
        led.audio_callback = enhanced_audio_callback
    
    # Run original main with enhancements
    main()

if __name__ == "__main__":
    enhanced_main()
