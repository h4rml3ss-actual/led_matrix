#!/usr/bin/env python3
"""
Comprehensive tests for the noise filtering and voice activity detection system.
Tests noise reduction, voice activity detection, adaptive gain control, and spectral subtraction.
"""

import unittest
import numpy as np
import time
from noise_filter import NoiseFilter, NoiseProfile, VoiceActivityResult, filter_audio_simple
from enhanced_audio_features import EnhancedFeatureExtractor


class TestNoiseFilter(unittest.TestCase):
    """Test cases for the NoiseFilter class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.samplerate = 44100
        self.frame_size = 1024
        self.noise_filter = NoiseFilter(self.samplerate, self.frame_size)
        
        # Create test signals
        self.duration = self.frame_size / self.samplerate  # Duration in seconds
        self.t = np.linspace(0, self.duration, self.frame_size)
        
        # Pure tone signal (simulates voice)
        self.voice_signal = 0.1 * np.sin(2 * np.pi * 440 * self.t)  # 440 Hz tone
        
        # White noise signal
        np.random.seed(42)  # For reproducible tests
        self.noise_signal = 0.02 * np.random.randn(self.frame_size)
        
        # Mixed signal (voice + noise)
        self.mixed_signal = self.voice_signal + self.noise_signal
        
        # Very quiet signal (below VAD threshold)
        self.quiet_signal = 0.001 * np.sin(2 * np.pi * 200 * self.t)
    
    def test_noise_filter_initialization(self):
        """Test NoiseFilter initialization."""
        self.assertEqual(self.noise_filter.samplerate, self.samplerate)
        self.assertEqual(self.noise_filter.frame_size, self.frame_size)
        self.assertIsNone(self.noise_filter.noise_profile)
        self.assertEqual(self.noise_filter.current_gain, 1.0)
        self.assertIsInstance(self.noise_filter.voice_bins, np.ndarray)
        self.assertTrue(len(self.noise_filter.voice_bins) > 0)
    
    def test_voice_activity_detection_voice_signal(self):
        """Test VAD with clear voice signal."""
        vad_result = self.noise_filter.detect_voice_activity(self.voice_signal)
        
        self.assertIsInstance(vad_result, VoiceActivityResult)
        self.assertTrue(vad_result.is_voice)
        self.assertGreater(vad_result.confidence, 0.5)
        self.assertGreater(vad_result.energy_ratio, 1.0)
        self.assertIsInstance(vad_result.spectral_ratio, float)
        self.assertIsInstance(vad_result.zero_crossing_ratio, float)
    
    def test_voice_activity_detection_noise_signal(self):
        """Test VAD with noise-only signal."""
        vad_result = self.noise_filter.detect_voice_activity(self.noise_signal)
        
        self.assertIsInstance(vad_result, VoiceActivityResult)
        # Noise might be detected as voice depending on characteristics
        # but confidence should be lower
        self.assertIsInstance(vad_result.is_voice, bool)
        self.assertIsInstance(vad_result.confidence, float)
        self.assertGreaterEqual(vad_result.confidence, 0.0)
        self.assertLessEqual(vad_result.confidence, 1.0)
    
    def test_voice_activity_detection_quiet_signal(self):
        """Test VAD with very quiet signal."""
        vad_result = self.noise_filter.detect_voice_activity(self.quiet_signal)
        
        self.assertIsInstance(vad_result, VoiceActivityResult)
        self.assertFalse(vad_result.is_voice)  # Should be below threshold
        self.assertLess(vad_result.confidence, 0.5)
        self.assertLess(vad_result.energy_ratio, 1.0)
    
    def test_noise_profile_update(self):
        """Test noise profile learning."""
        # Initially no noise profile
        self.assertIsNone(self.noise_filter.noise_profile)
        
        # Update with noise signal
        self.noise_filter._update_noise_profile(self.noise_signal)
        
        # Should have noise profile now
        self.assertIsNotNone(self.noise_filter.noise_profile)
        self.assertIsInstance(self.noise_filter.noise_profile, NoiseProfile)
        self.assertEqual(self.noise_filter.noise_profile.update_count, 1)
        self.assertGreater(self.noise_filter.noise_profile.noise_energy, 0)
        self.assertEqual(len(self.noise_filter.noise_profile.noise_spectrum), self.frame_size // 2 + 1)
        
        # Update again after some time
        time.sleep(0.6)  # Wait longer than update threshold
        initial_count = self.noise_filter.noise_profile.update_count
        self.noise_filter._update_noise_profile(self.noise_signal)
        
        # Should have updated
        self.assertEqual(self.noise_filter.noise_profile.update_count, initial_count + 1)
    
    def test_spectral_subtraction(self):
        """Test spectral subtraction noise reduction."""
        # First, establish noise profile
        self.noise_filter._update_noise_profile(self.noise_signal)
        
        # Apply spectral subtraction to mixed signal
        filtered_signal = self.noise_filter._apply_spectral_subtraction(self.mixed_signal)
        
        self.assertEqual(len(filtered_signal), len(self.mixed_signal))
        self.assertFalse(np.any(np.isnan(filtered_signal)))
        self.assertFalse(np.any(np.isinf(filtered_signal)))
        
        # Filtered signal should have different characteristics than original
        original_energy = np.mean(self.mixed_signal**2)
        filtered_energy = np.mean(filtered_signal**2)
        self.assertIsInstance(filtered_energy, float)
        self.assertGreater(filtered_energy, 0)
    
    def test_voice_band_filter(self):
        """Test voice frequency band filtering."""
        # Create signal with frequencies outside voice range
        high_freq_signal = 0.1 * np.sin(2 * np.pi * 10000 * self.t)  # 10 kHz
        low_freq_signal = 0.1 * np.sin(2 * np.pi * 50 * self.t)      # 50 Hz
        
        filtered_high = self.noise_filter._apply_voice_band_filter(high_freq_signal)
        filtered_low = self.noise_filter._apply_voice_band_filter(low_freq_signal)
        filtered_voice = self.noise_filter._apply_voice_band_filter(self.voice_signal)
        
        # Voice signal should pass through relatively unchanged
        voice_attenuation = np.mean(filtered_voice**2) / np.mean(self.voice_signal**2)
        self.assertGreater(voice_attenuation, 0.5)  # Should retain most energy
        
        # High frequency should be attenuated more
        high_attenuation = np.mean(filtered_high**2) / np.mean(high_freq_signal**2)
        self.assertLess(high_attenuation, voice_attenuation)
        
        # Low frequency should be attenuated more
        low_attenuation = np.mean(filtered_low**2) / np.mean(low_freq_signal**2)
        self.assertLess(low_attenuation, voice_attenuation)
    
    def test_adaptive_gain_control(self):
        """Test adaptive gain control."""
        # Test with voice signal
        initial_gain = self.noise_filter.current_gain
        gained_signal = self.noise_filter._apply_adaptive_gain(self.voice_signal, is_voice=True)
        
        self.assertEqual(len(gained_signal), len(self.voice_signal))
        self.assertFalse(np.any(np.isnan(gained_signal)))
        
        # Gain should have been adjusted
        final_gain = self.noise_filter.current_gain
        self.assertIsInstance(final_gain, float)
        self.assertGreaterEqual(final_gain, self.noise_filter.min_gain)
        self.assertLessEqual(final_gain, self.noise_filter.max_gain)
        
        # Test with non-voice signal (gain should not change much)
        pre_gain = self.noise_filter.current_gain
        self.noise_filter._apply_adaptive_gain(self.noise_signal, is_voice=False)
        post_gain = self.noise_filter.current_gain
        
        # Gain should be similar (smoothed)
        gain_change = abs(post_gain - pre_gain)
        self.assertLess(gain_change, 0.5)  # Should not change drastically
    
    def test_temporal_smoothing(self):
        """Test temporal smoothing."""
        # Create signal with sharp transitions
        sharp_signal = np.concatenate([
            np.zeros(self.frame_size // 2),
            np.ones(self.frame_size // 2)
        ])
        
        smoothed_signal = self.noise_filter._apply_temporal_smoothing(sharp_signal)
        
        self.assertEqual(len(smoothed_signal), len(sharp_signal))
        
        # Smoothed signal should have less abrupt changes
        original_diff = np.max(np.abs(np.diff(sharp_signal)))
        smoothed_diff = np.max(np.abs(np.diff(smoothed_signal)))
        self.assertLessEqual(smoothed_diff, original_diff)
    
    def test_complete_filtering_pipeline(self):
        """Test the complete filtering pipeline."""
        # First, manually update noise profile with noise signal to ensure it's established
        self.noise_filter._update_noise_profile(self.noise_signal)
        
        # Test with mixed signal
        filtered_audio, vad_result = self.noise_filter.filter_audio(self.mixed_signal)
        
        self.assertEqual(len(filtered_audio), self.frame_size)
        self.assertIsInstance(vad_result, VoiceActivityResult)
        self.assertFalse(np.any(np.isnan(filtered_audio)))
        self.assertFalse(np.any(np.isinf(filtered_audio)))
        
        # Should have established noise profile
        self.assertIsNotNone(self.noise_filter.noise_profile)
        
        # Test with voice signal
        filtered_voice, vad_voice = self.noise_filter.filter_audio(self.voice_signal, update_noise_profile=False)
        
        self.assertTrue(vad_voice.is_voice)
        self.assertGreater(vad_voice.confidence, 0.5)
        
        # Test with quiet signal
        filtered_quiet, vad_quiet = self.noise_filter.filter_audio(self.quiet_signal, update_noise_profile=False)
        
        # For very quiet signals, VAD might still detect some activity due to spectral characteristics
        # but confidence should be lower than for clear voice
        self.assertLess(vad_quiet.confidence, vad_voice.confidence)
    
    def test_noise_gate_threshold_setting(self):
        """Test setting noise gate threshold."""
        original_threshold = self.noise_filter.vad_energy_threshold
        
        new_threshold = 0.05
        self.noise_filter.set_noise_gate_threshold(new_threshold)
        self.assertEqual(self.noise_filter.vad_energy_threshold, new_threshold)
        
        # Test minimum threshold enforcement
        self.noise_filter.set_noise_gate_threshold(0.0001)
        self.assertGreaterEqual(self.noise_filter.vad_energy_threshold, 0.001)
    
    def test_adaptive_gain_target_setting(self):
        """Test setting adaptive gain target."""
        original_target = self.noise_filter.target_rms
        
        new_target = 0.1
        self.noise_filter.set_adaptive_gain_target(new_target)
        self.assertEqual(self.noise_filter.target_rms, new_target)
        
        # Test bounds enforcement
        self.noise_filter.set_adaptive_gain_target(0.005)  # Too low
        self.assertGreaterEqual(self.noise_filter.target_rms, 0.01)
        
        self.noise_filter.set_adaptive_gain_target(1.0)    # Too high
        self.assertLessEqual(self.noise_filter.target_rms, 0.5)
    
    def test_noise_profile_reset(self):
        """Test noise profile reset."""
        # Establish noise profile
        self.noise_filter._update_noise_profile(self.noise_signal)
        self.assertIsNotNone(self.noise_filter.noise_profile)
        
        # Reset
        self.noise_filter.reset_noise_profile()
        self.assertIsNone(self.noise_filter.noise_profile)
    
    def test_noise_profile_info(self):
        """Test getting noise profile information."""
        # Initially no profile
        info = self.noise_filter.get_noise_profile_info()
        self.assertFalse(info['initialized'])
        self.assertEqual(info['update_count'], 0)
        
        # After establishing profile
        self.noise_filter._update_noise_profile(self.noise_signal)
        info = self.noise_filter.get_noise_profile_info()
        
        self.assertTrue(info['initialized'])
        self.assertEqual(info['update_count'], 1)
        self.assertGreater(info['noise_energy'], 0)
        self.assertIsInstance(info['last_update'], float)
        self.assertEqual(info['spectrum_length'], self.frame_size // 2 + 1)
    
    def test_current_gain_retrieval(self):
        """Test getting current gain value."""
        gain = self.noise_filter.get_current_gain()
        self.assertIsInstance(gain, float)
        self.assertEqual(gain, self.noise_filter.current_gain)
        self.assertGreaterEqual(gain, self.noise_filter.min_gain)
        self.assertLessEqual(gain, self.noise_filter.max_gain)


class TestNoiseFilterIntegration(unittest.TestCase):
    """Test integration of NoiseFilter with EnhancedFeatureExtractor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.samplerate = 44100
        self.frame_size = 1024
        
        # Create extractors with and without noise filtering
        self.extractor_with_filter = EnhancedFeatureExtractor(
            self.samplerate, self.frame_size, enable_noise_filtering=True
        )
        self.extractor_without_filter = EnhancedFeatureExtractor(
            self.samplerate, self.frame_size, enable_noise_filtering=False
        )
        
        # Create test signal
        duration = self.frame_size / self.samplerate
        t = np.linspace(0, duration, self.frame_size)
        np.random.seed(42)
        
        # Voice signal with noise
        voice = 0.1 * np.sin(2 * np.pi * 440 * t)
        noise = 0.03 * np.random.randn(self.frame_size)
        self.test_signal = (voice + noise).astype(np.float32)
    
    def test_feature_extraction_with_noise_filtering(self):
        """Test feature extraction with noise filtering enabled."""
        features = self.extractor_with_filter.extract_features(self.test_signal)
        
        # Should have all required features
        self.assertIsInstance(features.rms, float)
        self.assertIsInstance(features.voice_activity, bool)
        self.assertIsInstance(features.confidence, float)
        
        # Confidence should be reasonable
        self.assertGreaterEqual(features.confidence, 0.0)
        self.assertLessEqual(features.confidence, 1.0)
        
        # Should detect voice activity for this signal
        self.assertTrue(features.voice_activity)
    
    def test_feature_extraction_without_noise_filtering(self):
        """Test feature extraction without noise filtering."""
        features = self.extractor_without_filter.extract_features(self.test_signal)
        
        # Should still have all required features
        self.assertIsInstance(features.rms, float)
        self.assertIsInstance(features.voice_activity, bool)
        self.assertIsInstance(features.confidence, float)
        
        # Should use fallback VAD
        self.assertIsInstance(features.voice_activity, bool)
    
    def test_noise_filter_configuration(self):
        """Test noise filter configuration through feature extractor."""
        # Test threshold setting
        self.extractor_with_filter.set_noise_gate_threshold(0.02)
        if self.extractor_with_filter.noise_filter:
            self.assertEqual(self.extractor_with_filter.noise_filter.vad_energy_threshold, 0.02)
        
        # Test gain target setting
        self.extractor_with_filter.set_adaptive_gain_target(0.08)
        if self.extractor_with_filter.noise_filter:
            self.assertEqual(self.extractor_with_filter.noise_filter.target_rms, 0.08)
        
        # Test noise profile reset
        self.extractor_with_filter.reset_noise_profile()
        if self.extractor_with_filter.noise_filter:
            self.assertIsNone(self.extractor_with_filter.noise_filter.noise_profile)
    
    def test_noise_filter_info_retrieval(self):
        """Test getting noise filter information."""
        # With noise filtering enabled
        info_with = self.extractor_with_filter.get_noise_filter_info()
        self.assertTrue(info_with['noise_filtering_enabled'])
        self.assertIn('current_gain', info_with)
        
        # Without noise filtering enabled
        info_without = self.extractor_without_filter.get_noise_filter_info()
        self.assertFalse(info_without['noise_filtering_enabled'])
        self.assertEqual(info_without['current_gain'], 1.0)
    
    def test_noise_profile_update_integration(self):
        """Test noise profile update through feature extractor."""
        # Create noise signal
        np.random.seed(123)
        noise_signal = (0.02 * np.random.randn(self.frame_size)).astype(np.float32)
        
        # Update noise profile
        self.extractor_with_filter.update_noise_profile(noise_signal)
        
        # Check that noise profile was updated
        info = self.extractor_with_filter.get_noise_filter_info()
        if info['noise_filtering_enabled']:
            self.assertGreater(info['update_count'], 0)
            self.assertGreater(info['noise_energy'], 0)


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions for noise filtering."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.frame_size = 1024
        duration = self.frame_size / 44100
        t = np.linspace(0, duration, self.frame_size)
        
        # Create test signal
        self.test_signal = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    
    def test_filter_audio_simple(self):
        """Test the simple audio filtering convenience function."""
        filtered_audio, is_voice = filter_audio_simple(self.test_signal)
        
        self.assertEqual(len(filtered_audio), len(self.test_signal))
        self.assertIsInstance(is_voice, bool)
        self.assertFalse(np.any(np.isnan(filtered_audio)))
        self.assertFalse(np.any(np.isinf(filtered_audio)))
        
        # Should detect voice for this signal
        self.assertTrue(is_voice)
    
    def test_filter_audio_simple_with_custom_filter(self):
        """Test simple filtering with custom NoiseFilter instance."""
        custom_filter = NoiseFilter(44100, 1024)
        custom_filter.set_noise_gate_threshold(0.02)
        
        filtered_audio, is_voice = filter_audio_simple(self.test_signal, custom_filter)
        
        self.assertEqual(len(filtered_audio), len(self.test_signal))
        self.assertIsInstance(is_voice, bool)


class TestNoiseFilterEdgeCases(unittest.TestCase):
    """Test edge cases and error handling for noise filtering."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.noise_filter = NoiseFilter(44100, 1024)
    
    def test_empty_audio_signal(self):
        """Test handling of empty audio signal."""
        empty_signal = np.array([])
        
        # Should handle gracefully without crashing
        try:
            vad_result = self.noise_filter.detect_voice_activity(empty_signal)
            # If it doesn't crash, check the result
            self.assertIsInstance(vad_result, VoiceActivityResult)
        except (ValueError, IndexError):
            # It's acceptable to raise an error for empty input
            pass
    
    def test_very_short_audio_signal(self):
        """Test handling of very short audio signals."""
        short_signal = np.array([0.1, -0.1, 0.05])
        
        try:
            vad_result = self.noise_filter.detect_voice_activity(short_signal)
            self.assertIsInstance(vad_result, VoiceActivityResult)
        except (ValueError, IndexError):
            # Acceptable to fail on very short signals
            pass
    
    def test_all_zero_signal(self):
        """Test handling of all-zero signal."""
        zero_signal = np.zeros(1024)
        
        vad_result = self.noise_filter.detect_voice_activity(zero_signal)
        self.assertIsInstance(vad_result, VoiceActivityResult)
        self.assertFalse(vad_result.is_voice)
        self.assertEqual(vad_result.confidence, 0.0)
    
    def test_very_loud_signal(self):
        """Test handling of very loud signal (clipping prevention)."""
        loud_signal = 10.0 * np.ones(1024)  # Very loud signal
        
        # Should handle without crashing
        filtered_audio, vad_result = self.noise_filter.filter_audio(loud_signal)
        
        # Should prevent clipping
        self.assertLessEqual(np.max(np.abs(filtered_audio)), 1.0)
        self.assertIsInstance(vad_result, VoiceActivityResult)
    
    def test_nan_and_inf_handling(self):
        """Test handling of NaN and infinite values."""
        # Signal with NaN
        nan_signal = np.full(1024, np.nan)
        
        try:
            filtered_audio, vad_result = self.noise_filter.filter_audio(nan_signal)
            # Should not contain NaN or inf in output
            self.assertFalse(np.any(np.isnan(filtered_audio)))
            self.assertFalse(np.any(np.isinf(filtered_audio)))
        except (ValueError, RuntimeError):
            # Acceptable to fail on invalid input
            pass
        
        # Signal with infinity
        inf_signal = np.full(1024, np.inf)
        
        try:
            filtered_audio, vad_result = self.noise_filter.filter_audio(inf_signal)
            # Should not contain NaN or inf in output
            self.assertFalse(np.any(np.isnan(filtered_audio)))
            self.assertFalse(np.any(np.isinf(filtered_audio)))
        except (ValueError, RuntimeError):
            # Acceptable to fail on invalid input
            pass


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)