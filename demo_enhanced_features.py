#!/usr/bin/env python3
"""
Demonstration script showing how to use the enhanced audio feature extraction system.
This shows how it extends the current extract_features function in led.py.
"""

import numpy as np
import time
from enhanced_audio_features import EnhancedFeatureExtractor

def demo_basic_usage():
    """Demonstrate basic usage of the enhanced feature extractor."""
    print("=== Enhanced Audio Feature Extraction Demo ===\n")
    
    # Initialize the extractor
    extractor = EnhancedFeatureExtractor(samplerate=44100, frame_size=1024)
    
    # Generate some test audio (simulating microphone input)
    samplerate = 44100
    duration = 0.1  # 100ms like the current system
    t = np.linspace(0, duration, int(samplerate * duration), False)
    
    print("1. Testing with different audio signals:\n")
    
    # Test 1: Calm speech simulation (low energy, stable pitch)
    print("   Calm speech simulation:")
    calm_audio = 0.3 * np.sin(2 * np.pi * 150 * t) + 0.1 * np.sin(2 * np.pi * 300 * t)
    calm_audio += 0.02 * np.random.randn(len(calm_audio))  # Light noise
    
    calm_features = extractor.extract_features(calm_audio, timestamp=time.time())
    print(f"     RMS Energy: {calm_features.rms:.4f}")
    print(f"     Spectral Centroid: {calm_features.spectral_centroid:.1f} Hz")
    print(f"     Zero Crossing Rate: {calm_features.zero_crossing_rate:.4f}")
    print(f"     Fundamental Freq: {calm_features.fundamental_freq:.1f} Hz")
    print(f"     Confidence: {calm_features.confidence:.3f}")
    print(f"     Voice Activity: {calm_features.voice_activity}")
    print()
    
    # Test 2: Excited speech simulation (high energy, variable pitch)
    print("   Excited speech simulation:")
    f0_variation = 20 * np.sin(2 * np.pi * 8 * t)  # Pitch variation
    excited_audio = 0.7 * np.sin(2 * np.pi * (200 + f0_variation) * t)
    excited_audio += 0.4 * np.sin(2 * np.pi * 2 * (200 + f0_variation) * t)
    excited_audio += 0.05 * np.random.randn(len(excited_audio))  # More noise
    
    excited_features = extractor.extract_features(excited_audio, timestamp=time.time())
    print(f"     RMS Energy: {excited_features.rms:.4f}")
    print(f"     Spectral Centroid: {excited_features.spectral_centroid:.1f} Hz")
    print(f"     Zero Crossing Rate: {excited_features.zero_crossing_rate:.4f}")
    print(f"     Fundamental Freq: {excited_features.fundamental_freq:.1f} Hz")
    print(f"     Confidence: {excited_features.confidence:.3f}")
    print(f"     Voice Activity: {excited_features.voice_activity}")
    print()
    
    # Test 3: Silence
    print("   Silence:")
    silence = np.zeros(len(t))
    silence_features = extractor.extract_features(silence, timestamp=time.time())
    print(f"     RMS Energy: {silence_features.rms:.4f}")
    print(f"     Voice Activity: {silence_features.voice_activity}")
    print(f"     Confidence: {silence_features.confidence:.3f}")
    print()

def demo_comparison_with_original():
    """Compare enhanced features with original extract_features function."""
    print("2. Comparison with original extract_features function:\n")
    
    # Simulate the original extract_features function from led.py
    def original_extract_features(pcm_block, samplerate):
        """Original feature extraction from led.py"""
        x = pcm_block.astype(np.float32) / 32768.0 if pcm_block.dtype == np.int16 else pcm_block
        # RMS energy
        rms = np.sqrt(np.mean(x**2))
        # Zero-crossing rate
        zcr = np.sum(np.abs(np.diff(np.sign(x)))) / (2 * len(x))
        # Spectral centroid
        freqs = np.fft.rfftfreq(len(x), d=1/samplerate)
        mags = np.abs(np.fft.rfft(x))
        centroid = np.sum(freqs * mags) / (np.sum(mags) + 1e-6)
        return rms, zcr, centroid
    
    # Generate test audio
    samplerate = 44100
    duration = 0.1
    t = np.linspace(0, duration, int(samplerate * duration), False)
    test_audio = 0.5 * np.sin(2 * np.pi * 1000 * t)  # 1kHz sine wave
    
    # Original features
    orig_rms, orig_zcr, orig_centroid = original_extract_features(test_audio, samplerate)
    
    # Enhanced features
    extractor = EnhancedFeatureExtractor(samplerate=samplerate)
    enhanced_features = extractor.extract_features(test_audio)
    
    print("   Original features:")
    print(f"     RMS: {orig_rms:.4f}")
    print(f"     ZCR: {orig_zcr:.4f}")
    print(f"     Spectral Centroid: {orig_centroid:.1f} Hz")
    print()
    
    print("   Enhanced features (includes all original + more):")
    print(f"     RMS: {enhanced_features.rms:.4f}")
    print(f"     ZCR: {enhanced_features.zero_crossing_rate:.4f}")
    print(f"     Spectral Centroid: {enhanced_features.spectral_centroid:.1f} Hz")
    print(f"     + Peak Energy: {enhanced_features.peak_energy:.4f}")
    print(f"     + Energy Variance: {enhanced_features.energy_variance:.6f}")
    print(f"     + Spectral Rolloff: {enhanced_features.spectral_rolloff:.1f} Hz")
    print(f"     + Spectral Flux: {enhanced_features.spectral_flux:.4f}")
    print(f"     + MFCCs: {[f'{mfcc:.2f}' for mfcc in enhanced_features.mfccs]}")
    print(f"     + Fundamental Freq: {enhanced_features.fundamental_freq:.1f} Hz")
    print(f"     + Pitch Stability: {enhanced_features.pitch_stability:.3f}")
    print(f"     + Tempo: {enhanced_features.tempo:.1f} BPM")
    print(f"     + Voice Activity: {enhanced_features.voice_activity}")
    print(f"     + Confidence: {enhanced_features.confidence:.3f}")
    print()

def demo_mood_detection_features():
    """Demonstrate features most relevant for mood detection."""
    print("3. Features most relevant for mood detection:\n")
    
    extractor = EnhancedFeatureExtractor()
    
    # Create different mood-like signals
    samplerate = 44100
    duration = 0.2
    t = np.linspace(0, duration, int(samplerate * duration), False)
    
    moods = {
        'calm': {
            'signal': 0.2 * np.sin(2 * np.pi * 120 * t) + 0.01 * np.random.randn(len(t)),
            'description': 'Low energy, stable pitch, minimal noise'
        },
        'neutral': {
            'signal': 0.4 * np.sin(2 * np.pi * 150 * t) + 0.2 * np.sin(2 * np.pi * 300 * t) + 0.02 * np.random.randn(len(t)),
            'description': 'Moderate energy, normal speech patterns'
        },
        'energetic': {
            'signal': 0.6 * np.sin(2 * np.pi * 180 * t) + 0.3 * np.sin(2 * np.pi * 360 * t) + 0.03 * np.random.randn(len(t)),
            'description': 'Higher energy, brighter spectrum'
        },
        'excited': {
            'signal': None,  # Will be generated with pitch variation
            'description': 'High energy, variable pitch, rapid changes'
        }
    }
    
    # Generate excited signal with pitch variation
    pitch_variation = 30 * np.sin(2 * np.pi * 10 * t)
    excited_signal = 0.8 * np.sin(2 * np.pi * (200 + pitch_variation) * t)
    excited_signal += 0.4 * np.sin(2 * np.pi * 2 * (200 + pitch_variation) * t)
    excited_signal += 0.05 * np.random.randn(len(excited_signal))
    moods['excited']['signal'] = excited_signal
    
    print("   Mood-relevant feature comparison:")
    print("   " + "="*80)
    print(f"   {'Mood':<10} {'RMS':<6} {'Centroid':<8} {'ZCR':<6} {'F0':<6} {'Stability':<9} {'Confidence':<10}")
    print("   " + "-"*80)
    
    for mood_name, mood_data in moods.items():
        features = extractor.extract_features(mood_data['signal'])
        print(f"   {mood_name:<10} {features.rms:<6.3f} {features.spectral_centroid:<8.0f} "
              f"{features.zero_crossing_rate:<6.3f} {features.fundamental_freq:<6.0f} "
              f"{features.pitch_stability:<9.3f} {features.confidence:<10.3f}")
    
    print("   " + "="*80)
    print()
    
    print("   Key observations for mood detection:")
    print("   - RMS energy increases from calm → excited")
    print("   - Spectral centroid generally increases with energy/brightness")
    print("   - Zero crossing rate varies with speech characteristics")
    print("   - Pitch stability decreases with emotional intensity")
    print("   - Confidence reflects signal quality and consistency")
    print()

def demo_integration_example():
    """Show how this integrates with the existing led.py system."""
    print("4. Integration with existing led.py system:\n")
    
    print("   Current led.py usage:")
    print("   ```python")
    print("   # In audio_callback:")
    print("   rms, zcr, centroid = extract_features(pcm_block.flatten(), samplerate)")
    print("   current_mood = detect_mood(rms, zcr, centroid)")
    print("   ```")
    print()
    
    print("   Enhanced system usage:")
    print("   ```python")
    print("   # Initialize once at startup:")
    print("   feature_extractor = EnhancedFeatureExtractor(samplerate=44100)")
    print()
    print("   # In audio_callback:")
    print("   features = feature_extractor.extract_features(pcm_block.flatten())")
    print("   current_mood = advanced_detect_mood(features)  # Uses all features")
    print("   ```")
    print()
    
    print("   Benefits of enhanced system:")
    print("   - More accurate mood detection with additional features")
    print("   - Confidence scoring for reliability assessment")
    print("   - Voice activity detection to ignore non-speech")
    print("   - Pitch analysis for emotional state detection")
    print("   - MFCC features for speech characteristics")
    print("   - Backward compatibility (includes original RMS, ZCR, centroid)")
    print()

if __name__ == "__main__":
    demo_basic_usage()
    demo_comparison_with_original()
    demo_mood_detection_features()
    demo_integration_example()
    
    print("=== Demo Complete ===")
    print("\nThe enhanced feature extraction system is ready for integration!")
    print("Next steps: Implement advanced mood detection algorithm using these features.")