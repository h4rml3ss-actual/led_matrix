#!/usr/bin/env python3
"""
Demonstration of the noise filtering and voice activity detection system.
Shows how to use the NoiseFilter and EnhancedFeatureExtractor with noise filtering.
"""

import numpy as np
import time
from noise_filter import NoiseFilter, filter_audio_simple
from enhanced_audio_features import EnhancedFeatureExtractor


def create_test_signals(samplerate=44100, duration=0.5):
    """Create various test signals for demonstration."""
    samples = int(samplerate * duration)
    t = np.linspace(0, duration, samples)
    
    # Voice signal (440 Hz tone)
    voice_signal = 0.1 * np.sin(2 * np.pi * 440 * t)
    
    # Background noise
    np.random.seed(42)
    noise_signal = 0.03 * np.random.randn(samples)
    
    # Mixed signal (voice + noise)
    mixed_signal = voice_signal + noise_signal
    
    # Quiet signal
    quiet_signal = 0.005 * np.sin(2 * np.pi * 200 * t)
    
    return {
        'voice': voice_signal.astype(np.float32),
        'noise': noise_signal.astype(np.float32),
        'mixed': mixed_signal.astype(np.float32),
        'quiet': quiet_signal.astype(np.float32)
    }


def demonstrate_noise_filter():
    """Demonstrate the NoiseFilter functionality."""
    print("=== Noise Filter Demonstration ===\n")
    
    # Create test signals
    signals = create_test_signals()
    
    # Initialize noise filter
    noise_filter = NoiseFilter(samplerate=44100, frame_size=len(signals['voice']))
    
    print("1. Testing Voice Activity Detection:")
    print("-" * 40)
    
    for signal_name, signal in signals.items():
        vad_result = noise_filter.detect_voice_activity(signal)
        print(f"{signal_name.capitalize():8} | Voice: {vad_result.is_voice:5} | "
              f"Confidence: {vad_result.confidence:.3f} | "
              f"Energy Ratio: {vad_result.energy_ratio:.2f}")
    
    print("\n2. Testing Complete Filtering Pipeline:")
    print("-" * 40)
    
    # First establish noise profile with noise signal
    print("Establishing noise profile...")
    noise_filter._update_noise_profile(signals['noise'])
    
    for signal_name, signal in signals.items():
        filtered_signal, vad_result = noise_filter.filter_audio(signal, update_noise_profile=False)
        
        original_rms = np.sqrt(np.mean(signal**2))
        filtered_rms = np.sqrt(np.mean(filtered_signal**2))
        
        print(f"{signal_name.capitalize():8} | Original RMS: {original_rms:.4f} | "
              f"Filtered RMS: {filtered_rms:.4f} | "
              f"Voice: {vad_result.is_voice} | "
              f"Confidence: {vad_result.confidence:.3f}")
    
    print("\n3. Noise Filter Configuration:")
    print("-" * 40)
    
    info = noise_filter.get_noise_profile_info()
    print(f"Noise profile initialized: {info['initialized']}")
    print(f"Update count: {info['update_count']}")
    print(f"Current adaptive gain: {noise_filter.get_current_gain():.3f}")
    print(f"VAD energy threshold: {noise_filter.vad_energy_threshold:.4f}")
    print(f"Target RMS level: {noise_filter.target_rms:.4f}")


def demonstrate_enhanced_feature_extractor():
    """Demonstrate the EnhancedFeatureExtractor with noise filtering."""
    print("\n\n=== Enhanced Feature Extractor with Noise Filtering ===\n")
    
    # Create test signals
    signals = create_test_signals()
    
    # Create extractors with and without noise filtering
    extractor_with_filter = EnhancedFeatureExtractor(
        samplerate=44100, 
        frame_size=len(signals['voice']), 
        enable_noise_filtering=True
    )
    
    extractor_without_filter = EnhancedFeatureExtractor(
        samplerate=44100, 
        frame_size=len(signals['voice']), 
        enable_noise_filtering=False
    )
    
    print("Comparing feature extraction with and without noise filtering:")
    print("-" * 70)
    print(f"{'Signal':<8} | {'Filter':<8} | {'RMS':<6} | {'Voice':<5} | {'Confidence':<10} | {'Centroid':<8}")
    print("-" * 70)
    
    for signal_name, signal in signals.items():
        # Extract features without filtering
        features_no_filter = extractor_without_filter.extract_features(signal)
        
        # Extract features with filtering
        features_with_filter = extractor_with_filter.extract_features(signal)
        
        print(f"{signal_name.capitalize():<8} | {'No':<8} | {features_no_filter.rms:<6.3f} | "
              f"{str(features_no_filter.voice_activity):<5} | {features_no_filter.confidence:<10.3f} | "
              f"{features_no_filter.spectral_centroid:<8.0f}")
        
        print(f"{'':<8} | {'Yes':<8} | {features_with_filter.rms:<6.3f} | "
              f"{str(features_with_filter.voice_activity):<5} | {features_with_filter.confidence:<10.3f} | "
              f"{features_with_filter.spectral_centroid:<8.0f}")
        
        print("-" * 70)
    
    print("\nNoise filter information:")
    info = extractor_with_filter.get_noise_filter_info()
    print(f"Noise filtering enabled: {info['noise_filtering_enabled']}")
    print(f"Current gain: {info.get('current_gain', 'N/A')}")


def demonstrate_convenience_functions():
    """Demonstrate the convenience functions."""
    print("\n\n=== Convenience Functions ===\n")
    
    # Create test signal
    signals = create_test_signals()
    test_signal = signals['mixed']
    
    print("Using filter_audio_simple convenience function:")
    print("-" * 50)
    
    filtered_audio, is_voice = filter_audio_simple(test_signal)
    
    original_rms = np.sqrt(np.mean(test_signal**2))
    filtered_rms = np.sqrt(np.mean(filtered_audio**2))
    
    print(f"Original RMS: {original_rms:.4f}")
    print(f"Filtered RMS: {filtered_rms:.4f}")
    print(f"Voice detected: {is_voice}")
    print(f"RMS change: {((filtered_rms - original_rms) / original_rms * 100):+.1f}%")


def demonstrate_adaptive_features():
    """Demonstrate adaptive features like gain control and noise learning."""
    print("\n\n=== Adaptive Features ===\n")
    
    signals = create_test_signals()
    noise_filter = NoiseFilter(samplerate=44100, frame_size=len(signals['voice']))
    
    print("1. Adaptive Gain Control:")
    print("-" * 30)
    
    # Process signals with different levels
    loud_signal = signals['voice'] * 5.0  # Very loud
    quiet_signal = signals['voice'] * 0.1  # Very quiet
    
    for signal_name, signal in [('Normal', signals['voice']), ('Loud', loud_signal), ('Quiet', quiet_signal)]:
        initial_gain = noise_filter.get_current_gain()
        filtered_signal, vad_result = noise_filter.filter_audio(signal, update_noise_profile=False)
        final_gain = noise_filter.get_current_gain()
        
        original_rms = np.sqrt(np.mean(signal**2))
        filtered_rms = np.sqrt(np.mean(filtered_signal**2))
        
        print(f"{signal_name:<6} | Original RMS: {original_rms:.4f} | "
              f"Filtered RMS: {filtered_rms:.4f} | "
              f"Gain: {initial_gain:.3f} → {final_gain:.3f}")
    
    print("\n2. Noise Profile Learning:")
    print("-" * 30)
    
    # Reset and learn noise profile
    noise_filter.reset_noise_profile()
    print("Initial state:", noise_filter.get_noise_profile_info()['initialized'])
    
    # Update with noise samples
    for i in range(3):
        noise_sample = 0.02 * np.random.randn(len(signals['noise']))
        noise_filter._update_noise_profile(noise_sample)
        time.sleep(0.6)  # Wait for update threshold
        
        info = noise_filter.get_noise_profile_info()
        print(f"After update {i+1}: Count={info['update_count']}, Energy={info['noise_energy']:.6f}")


if __name__ == '__main__':
    print("Noise Filtering and Voice Activity Detection Demo")
    print("=" * 50)
    
    try:
        demonstrate_noise_filter()
        demonstrate_enhanced_feature_extractor()
        demonstrate_convenience_functions()
        demonstrate_adaptive_features()
        
        print("\n" + "=" * 50)
        print("Demo completed successfully!")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()