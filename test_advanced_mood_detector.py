#!/usr/bin/env python3
"""
Unit tests for the advanced mood detection algorithm.
Tests the AdvancedMoodDetector class with synthetic audio feature data.
"""

import unittest
import numpy as np
import time
from unittest.mock import Mock, patch
from advanced_mood_detector import AdvancedMoodDetector, MoodResult, detect_mood_advanced, detect_mood_simple
from enhanced_audio_features import AudioFeatures
from mood_config import MoodConfig, EnergyThresholds, SpectralThresholds, TemporalThresholds, SmoothingConfig, NoiseFilteringConfig


class TestAdvancedMoodDetector(unittest.TestCase):
    """Test cases for the AdvancedMoodDetector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = AdvancedMoodDetector()
        self.timestamp = time.time()
    
    def create_test_features(self, rms=0.05, peak_energy=0.1, energy_variance=0.001,
                           spectral_centroid=2000, spectral_rolloff=3000, spectral_flux=1000,
                           mfccs=None, zero_crossing_rate=0.1, tempo=120, voice_activity=True,
                           fundamental_freq=150, pitch_stability=0.8, pitch_range=50,
                           confidence=0.8):
        """Create synthetic AudioFeatures for testing."""
        if mfccs is None:
            mfccs = [1.0, 0.5, -0.2, 0.1]
        
        return AudioFeatures(
            rms=rms,
            peak_energy=peak_energy,
            energy_variance=energy_variance,
            spectral_centroid=spectral_centroid,
            spectral_rolloff=spectral_rolloff,
            spectral_flux=spectral_flux,
            mfccs=mfccs,
            zero_crossing_rate=zero_crossing_rate,
            tempo=tempo,
            voice_activity=voice_activity,
            fundamental_freq=fundamental_freq,
            pitch_stability=pitch_stability,
            pitch_range=pitch_range,
            timestamp=self.timestamp,
            confidence=confidence
        )
    
    def test_calm_mood_detection(self):
        """Test detection of calm mood."""
        # Create features typical of calm speech
        features = self.create_test_features(
            rms=0.01,  # Low energy
            energy_variance=0.0005,  # Very low variance
            spectral_centroid=1500,  # Low centroid
            spectral_flux=500,  # Low flux
            zero_crossing_rate=0.03,  # Low ZCR
            tempo=80,  # Low tempo
            pitch_stability=0.9,  # High stability
            pitch_range=30  # Low range
        )
        
        result = self.detector.detect_mood(features)
        
        self.assertEqual(result.mood, 'calm')
        self.assertGreater(result.confidence, 0.5)
        self.assertIsInstance(result.debug_scores, dict)
        self.assertIn('calm_total', result.debug_scores)
    
    def test_energetic_mood_detection(self):
        """Test detection of energetic mood."""
        # Create features typical of energetic speech
        features = self.create_test_features(
            rms=0.12,  # High energy
            energy_variance=0.008,  # High variance
            spectral_centroid=3500,  # High centroid
            spectral_flux=8000,  # High flux
            zero_crossing_rate=0.18,  # High ZCR
            tempo=160,  # High tempo
            pitch_stability=0.4,  # Low stability
            pitch_range=120  # High range
        )
        
        result = self.detector.detect_mood(features)
        
        self.assertEqual(result.mood, 'energetic')
        self.assertGreater(result.confidence, 0.5)
    
    def test_excited_mood_detection(self):
        """Test detection of excited mood."""
        # Create features typical of excited speech
        features = self.create_test_features(
            rms=0.20,  # Very high energy
            peak_energy=0.4,  # High peak
            energy_variance=0.015,  # Very high variance
            spectral_centroid=4500,  # Very high centroid
            spectral_flux=15000,  # Very high flux
            zero_crossing_rate=0.25,  # Very high ZCR
            tempo=180,  # Very high tempo
            pitch_stability=0.2,  # Very low stability
            pitch_range=200,  # Very high range
            mfccs=[2.0, 1.5, -1.0, 0.8]  # High MFCC variance
        )
        
        result = self.detector.detect_mood(features)
        
        self.assertEqual(result.mood, 'excited')
        self.assertGreater(result.confidence, 0.5)
    
    def test_neutral_mood_detection(self):
        """Test detection of neutral mood."""
        # Create features typical of neutral speech
        features = self.create_test_features(
            rms=0.05,  # Medium energy
            spectral_centroid=2500,  # Medium centroid
            zero_crossing_rate=0.10,  # Medium ZCR
            tempo=120,  # Medium tempo
            pitch_stability=0.6,  # Medium stability
            pitch_range=80  # Medium range
        )
        
        result = self.detector.detect_mood(features)
        
        self.assertEqual(result.mood, 'neutral')
        self.assertGreater(result.confidence, 0.5)
    
    def test_confidence_calculation(self):
        """Test confidence calculation with various feature qualities."""
        # High quality features should give high confidence
        high_quality_features = self.create_test_features(
            confidence=0.9,
            voice_activity=True
        )
        
        result = self.detector.detect_mood(high_quality_features)
        high_confidence = result.confidence
        
        # Low quality features should give lower confidence
        low_quality_features = self.create_test_features(
            confidence=0.3,
            voice_activity=False
        )
        
        result = self.detector.detect_mood(low_quality_features)
        low_confidence = result.confidence
        
        self.assertGreater(high_confidence, low_confidence)
    
    def test_feature_history_tracking(self):
        """Test that feature history is properly maintained."""
        initial_history_length = len(self.detector.feature_history)
        
        features = self.create_test_features()
        self.detector.detect_mood(features)
        
        self.assertEqual(len(self.detector.feature_history), initial_history_length + 1)
        self.assertEqual(self.detector.feature_history[-1], features)
    
    def test_mood_history_tracking(self):
        """Test that mood history is properly maintained."""
        initial_history_length = len(self.detector.mood_history)
        
        features = self.create_test_features()
        result = self.detector.detect_mood(features)
        
        self.assertEqual(len(self.detector.mood_history), initial_history_length + 1)
        self.assertEqual(self.detector.mood_history[-1], result.mood)
    
    def test_history_buffer_limits(self):
        """Test that history buffers don't exceed maximum length."""
        # Fill history beyond maximum
        for i in range(15):  # More than max_history_length (10)
            features = self.create_test_features()
            self.detector.detect_mood(features)
        
        self.assertLessEqual(len(self.detector.feature_history), self.detector.max_history_length)
        self.assertLessEqual(len(self.detector.mood_history), self.detector.max_history_length)
    
    def test_transition_recommendation(self):
        """Test transition recommendation logic."""
        # First detection should recommend transition
        features = self.create_test_features()
        result = self.detector.detect_mood(features)
        
        # With high confidence, should recommend transition
        if result.confidence >= self.detector.config.smoothing.confidence_threshold:
            self.assertTrue(result.transition_recommended)
    
    def test_calibration(self):
        """Test user calibration functionality."""
        # Create calibration data with specific characteristics
        calibration_features = []
        for i in range(5):
            features = self.create_test_features(
                rms=0.08,  # Higher than default
                spectral_centroid=2500,  # Different from default
                zero_crossing_rate=0.12  # Different from default
            )
            calibration_features.append(features)
        
        # Store original thresholds
        original_calm_max = self.detector.config.energy.calm_max
        
        # Calibrate
        self.detector.calibrate_for_user(calibration_features)
        
        # Thresholds should be adjusted
        self.assertNotEqual(self.detector.config.energy.calm_max, original_calm_max)
    
    def test_threshold_updates(self):
        """Test threshold update functionality."""
        new_config = MoodConfig()
        new_config.energy.calm_max = 0.03  # Different from default
        
        self.detector.update_thresholds(new_config)
        
        self.assertEqual(self.detector.config.energy.calm_max, 0.03)
    
    def test_debug_scores(self):
        """Test that debug scores are properly calculated."""
        features = self.create_test_features()
        result = self.detector.detect_mood(features)
        
        # Check that all expected debug scores are present
        expected_keys = [
            'calm_energy', 'calm_spectral', 'calm_temporal', 'calm_pitch', 'calm_total',
            'neutral_energy', 'neutral_spectral', 'neutral_temporal', 'neutral_pitch', 'neutral_total',
            'energetic_energy', 'energetic_spectral', 'energetic_temporal', 'energetic_pitch', 'energetic_total',
            'excited_energy', 'excited_spectral', 'excited_temporal', 'excited_pitch', 'excited_total'
        ]
        
        for key in expected_keys:
            self.assertIn(key, result.debug_scores)
            self.assertIsInstance(result.debug_scores[key], float)
    
    def test_no_pitch_handling(self):
        """Test handling of features with no pitch information."""
        features = self.create_test_features(
            fundamental_freq=0,  # No pitch detected
            pitch_stability=0,
            pitch_range=0
        )
        
        result = self.detector.detect_mood(features)
        
        # Should still work and return a valid mood
        self.assertIn(result.mood, ['calm', 'neutral', 'energetic', 'excited'])
        self.assertGreater(result.confidence, 0)
    
    def test_edge_case_values(self):
        """Test handling of edge case feature values."""
        # Test with extreme values
        features = self.create_test_features(
            rms=0.0,  # Minimum energy
            spectral_centroid=0,  # Minimum centroid
            zero_crossing_rate=1.0,  # Maximum ZCR
            confidence=0.0  # Minimum confidence
        )
        
        result = self.detector.detect_mood(features)
        
        # Should handle gracefully
        self.assertIn(result.mood, ['calm', 'neutral', 'energetic', 'excited'])
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)


class TestConvenienceFunctions(unittest.TestCase):
    """Test cases for convenience functions."""
    
    def test_detect_mood_advanced(self):
        """Test the detect_mood_advanced convenience function."""
        features = AudioFeatures(
            rms=0.05, peak_energy=0.1, energy_variance=0.001,
            spectral_centroid=2000, spectral_rolloff=3000, spectral_flux=1000,
            mfccs=[1.0, 0.5, -0.2, 0.1], zero_crossing_rate=0.1, tempo=120,
            voice_activity=True, fundamental_freq=150, pitch_stability=0.8,
            pitch_range=50, timestamp=time.time(), confidence=0.8
        )
        
        result = detect_mood_advanced(features)
        
        self.assertIsInstance(result, MoodResult)
        self.assertIn(result.mood, ['calm', 'neutral', 'energetic', 'excited'])
    
    def test_detect_mood_advanced_with_detector(self):
        """Test detect_mood_advanced with provided detector."""
        detector = AdvancedMoodDetector()
        features = AudioFeatures(
            rms=0.05, peak_energy=0.1, energy_variance=0.001,
            spectral_centroid=2000, spectral_rolloff=3000, spectral_flux=1000,
            mfccs=[1.0, 0.5, -0.2, 0.1], zero_crossing_rate=0.1, tempo=120,
            voice_activity=True, fundamental_freq=150, pitch_stability=0.8,
            pitch_range=50, timestamp=time.time(), confidence=0.8
        )
        
        result = detect_mood_advanced(features, detector)
        
        self.assertIsInstance(result, MoodResult)
        # Should use the same detector instance
        self.assertEqual(len(detector.feature_history), 1)
    
    def test_detect_mood_simple_compatibility(self):
        """Test backward compatibility function."""
        # Test calm detection
        mood = detect_mood_simple(rms=0.01, zcr=0.03, centroid=2000)
        self.assertEqual(mood, 'calm')
        
        # Test energetic detection
        mood = detect_mood_simple(rms=0.10, zcr=0.18, centroid=2500)
        self.assertEqual(mood, 'energetic')
        
        # Test bright detection
        mood = detect_mood_simple(rms=0.05, zcr=0.10, centroid=3500)
        self.assertEqual(mood, 'bright')
        
        # Test neutral detection
        mood = detect_mood_simple(rms=0.05, zcr=0.10, centroid=2500)
        self.assertEqual(mood, 'neutral')


class TestMoodScoring(unittest.TestCase):
    """Test cases for individual mood scoring functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = AdvancedMoodDetector()
    
    def test_energy_scoring(self):
        """Test energy-based scoring for different moods."""
        # Test calm energy scoring
        features = AudioFeatures(
            rms=0.01, peak_energy=0.02, energy_variance=0.0005,
            spectral_centroid=2000, spectral_rolloff=3000, spectral_flux=1000,
            mfccs=[1.0, 0.5, -0.2, 0.1], zero_crossing_rate=0.1, tempo=120,
            voice_activity=True, fundamental_freq=150, pitch_stability=0.8,
            pitch_range=50, timestamp=time.time(), confidence=0.8
        )
        
        calm_score = self.detector._calculate_energy_score(features, 'calm')
        energetic_score = self.detector._calculate_energy_score(features, 'energetic')
        
        # Low energy should score higher for calm than energetic
        self.assertGreater(calm_score, energetic_score)
    
    def test_spectral_scoring(self):
        """Test spectral-based scoring for different moods."""
        # Test bright spectral characteristics
        features = AudioFeatures(
            rms=0.05, peak_energy=0.1, energy_variance=0.001,
            spectral_centroid=4000, spectral_rolloff=5000, spectral_flux=2000,
            mfccs=[1.0, 0.5, -0.2, 0.1], zero_crossing_rate=0.1, tempo=120,
            voice_activity=True, fundamental_freq=150, pitch_stability=0.8,
            pitch_range=50, timestamp=time.time(), confidence=0.8
        )
        
        calm_score = self.detector._calculate_spectral_score(features, 'calm')
        excited_score = self.detector._calculate_spectral_score(features, 'excited')
        
        # High centroid should score higher for excited than calm
        self.assertGreater(excited_score, calm_score)
    
    def test_temporal_scoring(self):
        """Test temporal-based scoring for different moods."""
        # Test high ZCR characteristics
        features = AudioFeatures(
            rms=0.05, peak_energy=0.1, energy_variance=0.001,
            spectral_centroid=2000, spectral_rolloff=3000, spectral_flux=1000,
            mfccs=[1.0, 0.5, -0.2, 0.1], zero_crossing_rate=0.20, tempo=160,
            voice_activity=True, fundamental_freq=150, pitch_stability=0.8,
            pitch_range=50, timestamp=time.time(), confidence=0.8
        )
        
        calm_score = self.detector._calculate_temporal_score(features, 'calm')
        energetic_score = self.detector._calculate_temporal_score(features, 'energetic')
        
        # High ZCR should score higher for energetic than calm
        self.assertGreater(energetic_score, calm_score)
    
    def test_pitch_scoring(self):
        """Test pitch-based scoring for different moods."""
        # Test unstable pitch characteristics
        features = AudioFeatures(
            rms=0.05, peak_energy=0.1, energy_variance=0.001,
            spectral_centroid=2000, spectral_rolloff=3000, spectral_flux=1000,
            mfccs=[1.0, 0.5, -0.2, 0.1], zero_crossing_rate=0.1, tempo=120,
            voice_activity=True, fundamental_freq=150, pitch_stability=0.2,
            pitch_range=150, timestamp=time.time(), confidence=0.8
        )
        
        calm_score = self.detector._calculate_pitch_score(features, 'calm')
        excited_score = self.detector._calculate_pitch_score(features, 'excited')
        
        # Unstable pitch should score higher for excited than calm
        self.assertGreater(excited_score, calm_score)


class TestConfigurationIntegration(unittest.TestCase):
    """Test integration with configuration system."""
    
    def test_custom_config_usage(self):
        """Test detector with custom configuration."""
        custom_config = MoodConfig()
        custom_config.energy.calm_max = 0.03  # Higher than default
        
        detector = AdvancedMoodDetector(custom_config)
        
        # Should use the custom configuration
        self.assertEqual(detector.config.energy.calm_max, 0.03)
    
    def test_config_validation_integration(self):
        """Test that detector works with validated configuration."""
        from mood_config import ConfigManager
        
        config_manager = ConfigManager()
        config = config_manager.get_default_config()
        
        detector = AdvancedMoodDetector(config)
        
        # Should work without issues
        features = AudioFeatures(
            rms=0.05, peak_energy=0.1, energy_variance=0.001,
            spectral_centroid=2000, spectral_rolloff=3000, spectral_flux=1000,
            mfccs=[1.0, 0.5, -0.2, 0.1], zero_crossing_rate=0.1, tempo=120,
            voice_activity=True, fundamental_freq=150, pitch_stability=0.8,
            pitch_range=50, timestamp=time.time(), confidence=0.8
        )
        
        result = detector.detect_mood(features)
        self.assertIsInstance(result, MoodResult)


if __name__ == '__main__':
    # Set up logging for tests
    import logging
    logging.basicConfig(level=logging.WARNING)
    
    # Run the tests
    unittest.main(verbosity=2)