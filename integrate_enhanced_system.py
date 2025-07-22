#!/usr/bin/env python3
"""
Integration script to update the original led.py with enhanced mood detection.
This script modifies the existing led.py to use enhanced components while
maintaining full backward compatibility.
"""

import os
import shutil
from datetime import datetime


def backup_original_led():
    """Create a backup of the original led.py file."""
    if os.path.exists('led.py'):
        backup_name = f'led_original_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.py'
        shutil.copy2('led.py', backup_name)
        print(f"Original led.py backed up as: {backup_name}")
        return True
    else:
        print("Warning: led.py not found")
        return False


def create_enhanced_led_patch():
    """Create the enhanced integration patch for led.py."""
    
    # Read the original led.py
    if not os.path.exists('led.py'):
        print("Error: led.py not found")
        return False
    
    with open('led.py', 'r') as f:
        original_content = f.read()
    
    # Enhanced imports to add at the top
    enhanced_imports = '''
# Enhanced mood detection components - graceful fallback if not available
try:
    from enhanced_audio_features import EnhancedFeatureExtractor
    from advanced_mood_detector import AdvancedMoodDetector
    from mood_transition_smoother import MoodTransitionSmoother
    from noise_filter import NoiseFilter
    from user_calibration import get_calibrated_detector
    from mood_config import ConfigManager
    ENHANCED_AVAILABLE = True
    print("Enhanced mood detection components loaded successfully")
except ImportError as e:
    print(f"Enhanced components not available: {e}")
    print("Falling back to original mood detection")
    ENHANCED_AVAILABLE = False

'''
    
    # Enhanced audio callback replacement
    enhanced_callback = '''
# Enhanced audio processing variables
enhanced_feature_extractor = None
enhanced_mood_detector = None
enhanced_transition_smoother = None
enhanced_enabled = False
latest_enhanced_mood = "neutral"
latest_enhanced_confidence = 0.5

def init_enhanced_components(user_id=None):
    """Initialize enhanced components if available."""
    global enhanced_feature_extractor, enhanced_mood_detector, enhanced_transition_smoother, enhanced_enabled
    
    if not ENHANCED_AVAILABLE:
        return False
    
    try:
        # Feature extractor with noise filtering
        enhanced_feature_extractor = EnhancedFeatureExtractor(
            samplerate=44100,
            frame_size=1024,
            enable_noise_filtering=True
        )
        
        # Mood detector (calibrated if user_id provided)
        if user_id:
            try:
                enhanced_mood_detector = get_calibrated_detector(user_id)
                print(f"Using calibrated mood detector for user: {user_id}")
            except Exception as e:
                print(f"Failed to load calibrated detector: {e}")
                enhanced_mood_detector = AdvancedMoodDetector()
        else:
            enhanced_mood_detector = AdvancedMoodDetector()
        
        # Transition smoother
        enhanced_transition_smoother = MoodTransitionSmoother()
        
        enhanced_enabled = True
        print("Enhanced audio processing initialized")
        return True
        
    except Exception as e:
        print(f"Failed to initialize enhanced components: {e}")
        return False

def enhanced_audio_callback(indata, frames, time_info, status):
    """Enhanced audio callback with fallback to original processing."""
    global latest_enhanced_mood, latest_enhanced_confidence
    
    if status:
        print(f"Sounddevice status: {status}", file=sys.stderr)
    
    # Always compute basic volume for compatibility
    rms = np.sqrt(np.mean(indata**2))
    latest_volume[0] = rms
    latest_audio_block[0] = indata.copy()
    
    if enhanced_enabled:
        try:
            # Extract enhanced features
            features = enhanced_feature_extractor.extract_features(indata.flatten(), timestamp=time.time())
            
            # Detect mood
            mood_result = enhanced_mood_detector.detect_mood(features)
            
            # Apply transition smoothing
            smoothed_mood = enhanced_transition_smoother.smooth_transition(
                mood_result.mood, 
                mood_result.confidence
            )
            
            # Update results
            latest_enhanced_mood = smoothed_mood
            latest_enhanced_confidence = mood_result.confidence
            
        except Exception as e:
            print(f"Enhanced processing failed, using fallback: {e}")
            # Fall back to original processing
            audio_callback(indata, frames, time_info, status)
    else:
        # Use original callback
        audio_callback(indata, frames, time_info, status)

def get_current_mood_enhanced():
    """Get current mood using enhanced detection if available."""
    if enhanced_enabled:
        return latest_enhanced_mood, latest_enhanced_confidence
    else:
        # Fallback to original mood detection
        pcm_block = latest_audio_block[0]
        if pcm_block is not None:
            rms_v, zcr_v, cent_v = extract_features(pcm_block.flatten(), 44100)
            mood = detect_mood(rms_v, zcr_v, cent_v)
            return mood, 0.7  # Default confidence for original method
        return "neutral", 0.5

'''
    
    # Find the position to insert enhanced imports (after existing imports)
    import_pos = original_content.find('from rgbmatrix import RGBMatrix, RGBMatrixOptions')
    if import_pos == -1:
        print("Error: Could not find import section in led.py")
        return False
    
    # Find end of import line
    import_end = original_content.find('\n', import_pos) + 1
    
    # Insert enhanced imports
    modified_content = (
        original_content[:import_end] + 
        enhanced_imports + 
        original_content[import_end:]
    )
    
    # Find the main function and add enhanced initialization
    main_func_pos = modified_content.find('def main():')
    if main_func_pos == -1:
        print("Error: Could not find main function in led.py")
        return False
    
    # Find the start of main function body
    main_body_start = modified_content.find('global silence_checker_thread', main_func_pos)
    if main_body_start == -1:
        main_body_start = modified_content.find('matrix = init_matrix()', main_func_pos)
    
    if main_body_start == -1:
        print("Error: Could not find main function body")
        return False
    
    # Insert enhanced initialization
    enhanced_init = '''
    # Enhanced system initialization
    user_id = None
    if "--user" in sys.argv:
        try:
            user_idx = sys.argv.index("--user") + 1
            if user_idx < len(sys.argv):
                user_id = sys.argv[user_idx]
        except (ValueError, IndexError):
            pass
    
    enhanced_mode = init_enhanced_components(user_id)
    print(f"Enhanced Mode: {'Enabled' if enhanced_mode else 'Disabled'}")
    if user_id:
        print(f"User Calibration: {user_id}")
    
    '''
    
    # Insert before matrix initialization
    matrix_init_pos = modified_content.find('matrix = init_matrix()', main_body_start)
    modified_content = (
        modified_content[:matrix_init_pos] + 
        enhanced_init + 
        modified_content[matrix_init_pos:]
    )
    
    # Replace audio callback usage
    callback_usage = 'callback=audio_callback'
    enhanced_callback_usage = 'callback=enhanced_audio_callback if enhanced_enabled else audio_callback'
    modified_content = modified_content.replace(callback_usage, enhanced_callback_usage)
    
    # Add enhanced mood detection in main loop
    mood_detection_pos = modified_content.find('current_mood = detect_mood(rms_v, zcr_v, cent_v)')
    if mood_detection_pos != -1:
        # Replace with enhanced mood detection
        enhanced_mood_detection = '''current_mood, confidence = get_current_mood_enhanced()
                # Original fallback (commented out): current_mood = detect_mood(rms_v, zcr_v, cent_v)'''
        
        # Find the end of the line
        line_end = modified_content.find('\n', mood_detection_pos)
        modified_content = (
            modified_content[:mood_detection_pos] + 
            enhanced_mood_detection + 
            modified_content[line_end:]
        )
    
    # Add enhanced functions before main()
    main_def_pos = modified_content.find('def main():')
    modified_content = (
        modified_content[:main_def_pos] + 
        enhanced_callback + 
        '\n' +
        modified_content[main_def_pos:]
    )
    
    # Write the enhanced version
    with open('led_enhanced.py', 'w') as f:
        f.write(modified_content)
    
    print("Enhanced led.py created as led_enhanced.py")
    return True


def create_simple_integration():
    """Create a simple integration that replaces key functions."""
    
    integration_code = '''#!/usr/bin/env python3
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
'''
    
    with open('led_with_enhancements.py', 'w') as f:
        f.write(integration_code)
    
    print("Simple integration created as led_with_enhancements.py")
    return True


def main():
    """Main integration function."""
    print("Enhanced LED System Integration")
    print("=" * 40)
    
    # Create backup
    backup_created = backup_original_led()
    
    # Create enhanced integration
    print("\nCreating enhanced integration...")
    
    # Create the simple integration approach
    if create_simple_integration():
        print("✓ Simple integration created successfully")
    else:
        print("✗ Simple integration failed")
    
    # Create the full enhanced version
    if os.path.exists('led_enhanced_integration.py'):
        print("✓ Full enhanced integration available")
    else:
        print("✗ Full enhanced integration not found")
    
    print("\nIntegration Options Created:")
    print("1. led_with_enhancements.py - Simple integration with original led.py")
    print("2. led_enhanced_integration.py - Full backward-compatible integration")
    print("3. enhanced_led.py - Complete rewrite with enhanced features")
    
    print("\nUsage:")
    print("  python3 led_with_enhancements.py                    # Enhanced mode")
    print("  python3 led_with_enhancements.py --user john        # With user calibration")
    print("  python3 led_enhanced_integration.py --enhanced      # Full enhanced mode")
    print("  python3 enhanced_led.py --calibrated --user john    # Complete enhanced system")
    
    print(f"\nOriginal led.py {'backed up' if backup_created else 'not found'}")
    print("Integration complete!")


if __name__ == "__main__":
    main()