#!/usr/bin/env python3
"""
Demonstration of the user calibration system for personalized mood detection.
Shows how to calibrate users, store calibration data, and apply personalized settings.
"""

import numpy as np
import time
from user_calibration import UserCalibrator, calibrate_user, get_calibrated_detector
from enhanced_audio_features import EnhancedFeatureExtractor, AudioFeatures
from advanced_mood_detector import AdvancedMoodDetector
from mood_config import ConfigManager


def create_user_voice_samples(user_profile: str, num_samples: int = 80) -> list:
    """
    Create simulated voice samples for different user profiles.
    
    Args:
        user_profile: Type of user ('quiet', 'normal', 'loud')
        num_samples: Number of samples to generate
        
    Returns:
        List of audio samples
    """
    samplerate = 44100
    frame_size = 1024
    duration = frame_size / samplerate
    t = np.linspace(0, duration, frame_size)
    
    samples = []
    
    # Define user characteristics
    if user_profile == 'quiet':
        base_amplitude = 0.03
        freq_range = (120, 180)
        noise_level = 0.005
    elif user_profile == 'normal':
        base_amplitude = 0.07
        freq_range = (150, 250)
        noise_level = 0.01
    elif user_profile == 'loud':
        base_amplitude = 0.15
        freq_range = (180, 300)
        noise_level = 0.02
    else:
        raise ValueError("Unknown user profile")
    
    np.random.seed(42)  # For reproducible results
    
    for i in range(num_samples):
        # Vary amplitude and frequency to simulate natural speech variation
        amplitude = base_amplitude * (0.7 + 0.6 * np.random.random())
        frequency = freq_range[0] + (freq_range[1] - freq_range[0]) * np.random.random()
        
        # Create base signal
        signal = amplitude * np.sin(2 * np.pi * frequency * t)
        
        # Add some harmonics for more realistic voice
        signal += 0.3 * amplitude * np.sin(2 * np.pi * frequency * 2 * t)
        signal += 0.1 * amplitude * np.sin(2 * np.pi * frequency * 3 * t)
        
        # Add noise
        noise = noise_level * np.random.randn(frame_size)
        signal += noise
        
        samples.append(signal.astype(np.float32))
    
    return samples


def demonstrate_calibration_process():
    """Demonstrate the complete calibration process."""
    print("=== User Calibration Process Demonstration ===\n")
    
    # Create calibrator
    calibrator = UserCalibrator()
    
    print("1. Starting Calibration Session")
    print("-" * 40)
    
    user_id = "demo_user"
    session = calibrator.start_calibration_session(user_id, duration=45.0)
    
    print(f"Started calibration for user: {session.user_id}")
    print(f"Target duration: {session.target_duration} seconds")
    print(f"Minimum samples required: {calibrator.min_samples_required}")
    
    print("\n2. Collecting Audio Samples")
    print("-" * 40)
    
    # Simulate collecting audio samples over time
    voice_samples = create_user_voice_samples('normal', 70)
    
    for i, sample in enumerate(voice_samples):
        success = calibrator.add_audio_sample(sample)
        
        if i % 15 == 0:  # Show progress every 15 samples
            progress = calibrator.get_calibration_progress()
            print(f"Sample {i+1:2d}: Added={success}, "
                  f"Progress={progress['progress']:.1%}, "
                  f"Samples={progress['samples']}, "
                  f"Voice ratio={progress['voice_ratio']:.1%}")
    
    # Simulate time passage for minimum duration
    session.start_time = time.time() - 50.0  # 50 seconds ago
    
    print(f"\nFinal progress check:")
    progress = calibrator.get_calibration_progress()
    print(f"Time progress: {progress['time_progress']:.1%}")
    print(f"Sample progress: {progress['sample_progress']:.1%}")
    print(f"Voice activity ratio: {progress['voice_ratio']:.1%}")
    print(f"Calibration complete: {calibrator.is_calibration_complete()}")
    
    print("\n3. Finishing Calibration")
    print("-" * 40)
    
    calibration_data = calibrator.finish_calibration_session()
    
    print(f"Calibration completed for user: {calibration_data.user_id}")
    print(f"Samples collected: {calibration_data.sample_count}")
    print(f"Confidence score: {calibration_data.confidence_score:.3f}")
    print(f"Duration: {calibration_data.calibration_duration:.1f} seconds")
    
    print(f"\nBaseline characteristics:")
    print(f"  RMS energy: {calibration_data.baseline_rms:.4f} ± {calibration_data.baseline_rms_std:.4f}")
    print(f"  Spectral centroid: {calibration_data.baseline_spectral_centroid:.0f} ± {calibration_data.baseline_spectral_centroid_std:.0f} Hz")
    print(f"  Zero-crossing rate: {calibration_data.baseline_zero_crossing_rate:.3f} ± {calibration_data.baseline_zero_crossing_rate_std:.3f}")
    print(f"  Fundamental freq: {calibration_data.baseline_fundamental_freq:.0f} ± {calibration_data.baseline_fundamental_freq_std:.0f} Hz")
    
    print(f"\nMood calibration factors:")
    print(f"  Calm: {calibration_data.calm_factor:.3f}")
    print(f"  Neutral: {calibration_data.neutral_factor:.3f}")
    print(f"  Energetic: {calibration_data.energetic_factor:.3f}")
    print(f"  Excited: {calibration_data.excited_factor:.3f}")
    
    return calibration_data


def demonstrate_calibration_comparison():
    """Demonstrate calibration for different user types."""
    print("\n\n=== Calibration Comparison for Different Users ===\n")
    
    user_profiles = ['quiet', 'normal', 'loud']
    calibration_results = {}
    
    for profile in user_profiles:
        print(f"Calibrating {profile} user...")
        
        # Generate samples for this user type
        samples = create_user_voice_samples(profile, 60)
        
        # Calibrate user
        calibration_data = calibrate_user(f"{profile}_user", samples)
        calibration_results[profile] = calibration_data
        
        print(f"  Baseline RMS: {calibration_data.baseline_rms:.4f}")
        print(f"  Baseline centroid: {calibration_data.baseline_spectral_centroid:.0f} Hz")
        print(f"  Calm factor: {calibration_data.calm_factor:.3f}")
        print(f"  Energetic factor: {calibration_data.energetic_factor:.3f}")
        print()
    
    print("Comparison Summary:")
    print("-" * 60)
    print(f"{'User':<8} | {'RMS':<6} | {'Centroid':<8} | {'Calm':<5} | {'Energetic':<9}")
    print("-" * 60)
    
    for profile in user_profiles:
        data = calibration_results[profile]
        print(f"{profile.capitalize():<8} | {data.baseline_rms:<6.3f} | "
              f"{data.baseline_spectral_centroid:<8.0f} | {data.calm_factor:<5.2f} | "
              f"{data.energetic_factor:<9.2f}")


def demonstrate_personalized_detection():
    """Demonstrate personalized mood detection."""
    print("\n\n=== Personalized Mood Detection ===\n")
    
    # Create different user types
    user_types = ['quiet', 'normal', 'loud']
    
    # Create test signals with different characteristics
    test_signals = {
        'low_energy': create_user_voice_samples('quiet', 1)[0],
        'medium_energy': create_user_voice_samples('normal', 1)[0],
        'high_energy': create_user_voice_samples('loud', 1)[0]
    }
    
    print("Comparing mood detection with and without calibration:")
    print("-" * 70)
    print(f"{'User':<8} | {'Signal':<12} | {'Default Mood':<12} | {'Calibrated Mood':<15} | {'Conf Diff':<9}")
    print("-" * 70)
    
    # Create default detector
    default_detector = AdvancedMoodDetector()
    feature_extractor = EnhancedFeatureExtractor()
    
    for user_type in user_types:
        # Calibrate user
        samples = create_user_voice_samples(user_type, 60)
        calibrate_user(f"{user_type}_user", samples)
        
        # Get calibrated detector
        calibrated_detector = get_calibrated_detector(f"{user_type}_user")
        
        for signal_name, signal in test_signals.items():
            # Extract features
            features = feature_extractor.extract_features(signal)
            
            # Detect mood with default detector
            default_result = default_detector.detect_mood(features)
            
            # Detect mood with calibrated detector
            calibrated_result = calibrated_detector.detect_mood(features)
            
            # Calculate confidence difference
            conf_diff = calibrated_result.confidence - default_result.confidence
            
            print(f"{user_type.capitalize():<8} | {signal_name:<12} | "
                  f"{default_result.mood:<12} | {calibrated_result.mood:<15} | "
                  f"{conf_diff:+.3f}")


def demonstrate_calibration_management():
    """Demonstrate calibration data management."""
    print("\n\n=== Calibration Data Management ===\n")
    
    calibrator = UserCalibrator()
    
    print("1. Creating calibration data for multiple users")
    print("-" * 50)
    
    # Create calibration for multiple users
    users = ['alice', 'bob', 'charlie']
    for user in users:
        samples = create_user_voice_samples('normal', 60)
        calibrate_user(user, samples)
        print(f"Calibrated user: {user}")
    
    print(f"\n2. Listing calibrated users")
    print("-" * 50)
    
    calibrated_users = calibrator.list_calibrated_users()
    print(f"Found {len(calibrated_users)} calibrated users: {calibrated_users}")
    
    print(f"\n3. Getting calibration summaries")
    print("-" * 50)
    
    for user in calibrated_users:
        summary = calibrator.get_calibration_summary(user)
        if summary:
            print(f"\nUser: {summary['user_id']}")
            print(f"  Calibration date: {summary['calibration_date']}")
            print(f"  Sample count: {summary['sample_count']}")
            print(f"  Confidence: {summary['confidence_score']:.3f}")
            print(f"  Baseline RMS: {summary['baseline_rms']:.4f}")
    
    print(f"\n4. Deleting calibration data")
    print("-" * 50)
    
    # Delete one user's data
    if 'charlie' in calibrated_users:
        success = calibrator.delete_calibration_data('charlie')
        print(f"Deleted charlie's calibration: {success}")
        
        updated_users = calibrator.list_calibrated_users()
        print(f"Remaining users: {updated_users}")


def demonstrate_calibration_quality():
    """Demonstrate calibration quality assessment."""
    print("\n\n=== Calibration Quality Assessment ===\n")
    
    print("Testing calibration with different sample qualities:")
    print("-" * 55)
    
    # Test with different numbers of samples
    sample_counts = [20, 40, 60, 100]
    
    for count in sample_counts:
        samples = create_user_voice_samples('normal', count)
        calibration_data = calibrate_user(f"user_{count}_samples", samples)
        
        print(f"Samples: {count:3d} | Confidence: {calibration_data.confidence_score:.3f} | "
              f"Duration: {calibration_data.calibration_duration:.1f}s")
    
    print("\nCalibration quality factors:")
    print("- More samples generally improve confidence")
    print("- Longer duration ensures voice activity diversity")
    print("- Higher confidence indicates more reliable calibration")


if __name__ == '__main__':
    print("User Calibration System Demonstration")
    print("=" * 50)
    
    try:
        demonstrate_calibration_process()
        demonstrate_calibration_comparison()
        demonstrate_personalized_detection()
        demonstrate_calibration_management()
        demonstrate_calibration_quality()
        
        print("\n" + "=" * 50)
        print("Calibration demonstration completed successfully!")
        print("\nKey benefits of user calibration:")
        print("- Personalized mood detection thresholds")
        print("- Improved accuracy for individual voice characteristics")
        print("- Adaptive system that learns user patterns")
        print("- Persistent calibration data storage")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()