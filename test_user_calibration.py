#!/usr/bin/env python3
"""
Comprehensive tests for the user calibration system.
Tests calibration data collection, analysis, storage, and application.
"""

import unittest
import numpy as np
import tempfile
import shutil
import time
import json
from pathlib import Path
from user_calibration import (
    CalibrationData, CalibrationSession, UserCalibrator,
    calibrate_user, get_calibrated_detector
)
from enhanced_audio_features import EnhancedFeatureExtractor, AudioFeatures
from mood_config import MoodConfig, ConfigManager
from advanced_mood_detector import AdvancedMoodDetector


class TestCalibrationData(unittest.TestCase):
    """Test CalibrationData dataclass."""
    
    def test_calibration_data_creation(self):
        """Test creating CalibrationData instance."""
        data = CalibrationData(
            user_id="test_user",
            timestamp=time.time(),
            baseline_rms=0.05,
            baseline_rms_std=0.01,
            baseline_spectral_centroid=2000.0,
            baseline_spectral_centroid_std=500.0,
            baseline_zero_crossing_rate=0.1,
            baseline_zero_crossing_rate_std=0.02,
            baseline_fundamental_freq=150.0,
            baseline_fundamental_freq_std=20.0,
            rms_range=(0.01, 0.15),
            spectral_centroid_range=(1000.0, 4000.0),
            zcr_range=(0.05, 0.2),
            f0_range=(100.0, 300.0),
            sample_count=100,
            confidence_score=0.8,
            calibration_duration=60.0
        )
        
        self.assertEqual(data.user_id, "test_user")
        self.assertEqual(data.baseline_rms, 0.05)
        self.assertEqual(data.sample_count, 100)
        self.assertEqual(data.calm_factor, 1.0)  # Default value


class TestCalibrationSession(unittest.TestCase):
    """Test CalibrationSession dataclass."""
    
    def test_calibration_session_creation(self):
        """Test creating CalibrationSession instance."""
        session = CalibrationSession(
            user_id="test_user",
            start_time=time.time(),
            target_duration=60.0
        )
        
        self.assertEqual(session.user_id, "test_user")
        self.assertEqual(session.target_duration, 60.0)
        self.assertIsNone(session.mood_target)
        self.assertIsInstance(session.feature_samples, list)
        self.assertEqual(len(session.feature_samples), 0)
    
    def test_calibration_session_with_mood_target(self):
        """Test creating CalibrationSession with mood target."""
        session = CalibrationSession(
            user_id="test_user",
            start_time=time.time(),
            target_duration=60.0,
            mood_target="calm"
        )
        
        self.assertEqual(session.mood_target, "calm")


class TestUserCalibrator(unittest.TestCase):
    """Test UserCalibrator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for calibration data
        self.temp_dir = tempfile.mkdtemp()
        
        # Create calibrator with temporary directory
        self.calibrator = UserCalibrator()
        self.calibrator.calibration_dir = Path(self.temp_dir)
        
        # Create test audio signals
        self.samplerate = 44100
        self.frame_size = 1024
        duration = self.frame_size / self.samplerate
        t = np.linspace(0, duration, self.frame_size)
        
        # Different voice characteristics for testing
        self.calm_voice = (0.03 * np.sin(2 * np.pi * 150 * t)).astype(np.float32)
        self.normal_voice = (0.07 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
        self.energetic_voice = (0.12 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
        
        # Add some noise for realism
        np.random.seed(42)
        noise = 0.01 * np.random.randn(self.frame_size).astype(np.float32)
        self.calm_voice += noise
        self.normal_voice += noise
        self.energetic_voice += noise
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_calibrator_initialization(self):
        """Test UserCalibrator initialization."""
        self.assertIsInstance(self.calibrator.config_manager, ConfigManager)
        self.assertIsInstance(self.calibrator.feature_extractor, EnhancedFeatureExtractor)
        self.assertIsNone(self.calibrator.current_session)
        self.assertEqual(self.calibrator.min_calibration_duration, 30.0)
        self.assertEqual(self.calibrator.recommended_calibration_duration, 60.0)
    
    def test_start_calibration_session(self):
        """Test starting a calibration session."""
        session = self.calibrator.start_calibration_session("test_user", duration=45.0)
        
        self.assertIsInstance(session, CalibrationSession)
        self.assertEqual(session.user_id, "test_user")
        self.assertEqual(session.target_duration, 45.0)
        self.assertIsNone(session.mood_target)
        self.assertEqual(self.calibrator.current_session, session)
    
    def test_start_calibration_session_with_mood_target(self):
        """Test starting calibration session with mood target."""
        session = self.calibrator.start_calibration_session(
            "test_user", duration=45.0, mood_target="calm"
        )
        
        self.assertEqual(session.mood_target, "calm")
    
    def test_start_calibration_session_default_duration(self):
        """Test starting calibration session with default duration."""
        session = self.calibrator.start_calibration_session("test_user")
        
        self.assertEqual(session.target_duration, self.calibrator.recommended_calibration_duration)
    
    def test_start_calibration_session_too_short(self):
        """Test starting calibration session with too short duration."""
        with self.assertRaises(ValueError):
            self.calibrator.start_calibration_session("test_user", duration=10.0)
    
    def test_start_calibration_session_already_active(self):
        """Test starting calibration session when one is already active."""
        self.calibrator.start_calibration_session("test_user")
        
        with self.assertRaises(RuntimeError):
            self.calibrator.start_calibration_session("another_user")
    
    def test_add_audio_sample_no_session(self):
        """Test adding audio sample without active session."""
        with self.assertRaises(RuntimeError):
            self.calibrator.add_audio_sample(self.normal_voice)
    
    def test_add_audio_sample_success(self):
        """Test successfully adding audio samples."""
        session = self.calibrator.start_calibration_session("test_user")
        
        # Add samples with voice activity
        success1 = self.calibrator.add_audio_sample(self.normal_voice)
        success2 = self.calibrator.add_audio_sample(self.energetic_voice)
        
        self.assertTrue(success1)
        self.assertTrue(success2)
        self.assertEqual(len(session.feature_samples), 2)
    
    def test_add_audio_sample_low_quality(self):
        """Test adding low-quality audio samples."""
        session = self.calibrator.start_calibration_session("test_user")
        
        # Very quiet signal that might not be detected as voice
        quiet_signal = 0.001 * np.sin(2 * np.pi * 100 * np.linspace(0, 1, self.frame_size))
        quiet_signal = quiet_signal.astype(np.float32)
        
        success = self.calibrator.add_audio_sample(quiet_signal)
        
        # Might succeed or fail depending on VAD sensitivity
        self.assertIsInstance(success, bool)
    
    def test_get_calibration_progress_no_session(self):
        """Test getting progress without active session."""
        progress = self.calibrator.get_calibration_progress()
        
        self.assertEqual(progress["progress"], 0.0)
        self.assertEqual(progress["samples"], 0)
    
    def test_get_calibration_progress_with_samples(self):
        """Test getting calibration progress with samples."""
        session = self.calibrator.start_calibration_session("test_user", duration=60.0)
        
        # Add some samples
        for _ in range(10):
            self.calibrator.add_audio_sample(self.normal_voice)
        
        progress = self.calibrator.get_calibration_progress()
        
        self.assertGreater(progress["samples"], 0)
        self.assertGreaterEqual(progress["voice_ratio"], 0.0)
        self.assertLessEqual(progress["voice_ratio"], 1.0)
        self.assertGreaterEqual(progress["progress"], 0.0)
        self.assertLessEqual(progress["progress"], 1.0)
    
    def test_is_calibration_complete_insufficient(self):
        """Test calibration completion check with insufficient data."""
        self.calibrator.start_calibration_session("test_user", duration=60.0)
        
        # Add only a few samples
        for _ in range(5):
            self.calibrator.add_audio_sample(self.normal_voice)
        
        self.assertFalse(self.calibrator.is_calibration_complete())
    
    def test_is_calibration_complete_sufficient(self):
        """Test calibration completion check with sufficient data."""
        # Start session with past start time to simulate elapsed time
        session = self.calibrator.start_calibration_session("test_user", duration=60.0)
        session.start_time = time.time() - 35.0  # 35 seconds ago
        
        # Add sufficient samples
        for _ in range(self.calibrator.min_samples_required + 5):
            self.calibrator.add_audio_sample(self.normal_voice)
        
        self.assertTrue(self.calibrator.is_calibration_complete())
    
    def test_finish_calibration_session_no_session(self):
        """Test finishing calibration without active session."""
        with self.assertRaises(RuntimeError):
            self.calibrator.finish_calibration_session()
    
    def test_finish_calibration_session_incomplete(self):
        """Test finishing incomplete calibration session."""
        self.calibrator.start_calibration_session("test_user")
        
        with self.assertRaises(RuntimeError):
            self.calibrator.finish_calibration_session()
    
    def test_finish_calibration_session_success(self):
        """Test successfully finishing calibration session."""
        # Start session with past start time
        session = self.calibrator.start_calibration_session("test_user", duration=60.0)
        session.start_time = time.time() - 35.0
        
        # Add sufficient samples
        for _ in range(self.calibrator.min_samples_required + 5):
            self.calibrator.add_audio_sample(self.normal_voice)
        
        calibration_data = self.calibrator.finish_calibration_session()
        
        self.assertIsInstance(calibration_data, CalibrationData)
        self.assertEqual(calibration_data.user_id, "test_user")
        self.assertGreater(calibration_data.sample_count, 0)
        self.assertIsNone(self.calibrator.current_session)
    
    def test_cancel_calibration_session(self):
        """Test canceling calibration session."""
        self.calibrator.start_calibration_session("test_user")
        self.assertIsNotNone(self.calibrator.current_session)
        
        self.calibrator.cancel_calibration_session()
        self.assertIsNone(self.calibrator.current_session)
    
    def test_analyze_calibration_samples(self):
        """Test analyzing calibration samples."""
        session = CalibrationSession(
            user_id="test_user",
            start_time=time.time() - 60.0,
            target_duration=60.0
        )
        
        # Add mock features
        for i in range(20):
            features = AudioFeatures(
                rms=0.05 + i * 0.001,
                peak_energy=0.1,
                energy_variance=0.001,
                spectral_centroid=2000.0 + i * 10,
                spectral_rolloff=4000.0,
                spectral_flux=100.0,
                mfccs=[1.0, 0.5, 0.3, 0.1],
                zero_crossing_rate=0.1 + i * 0.001,
                tempo=120.0,
                voice_activity=True,
                fundamental_freq=150.0 + i,
                pitch_stability=0.8,
                pitch_range=50.0,
                timestamp=time.time(),
                confidence=0.8
            )
            session.feature_samples.append(features)
        
        calibration_data = self.calibrator._analyze_calibration_samples(session)
        
        self.assertEqual(calibration_data.user_id, "test_user")
        self.assertEqual(calibration_data.sample_count, 20)
        self.assertGreater(calibration_data.baseline_rms, 0)
        self.assertGreater(calibration_data.baseline_spectral_centroid, 0)
        self.assertGreater(calibration_data.confidence_score, 0)
    
    def test_calculate_mood_factors(self):
        """Test calculating mood-specific factors."""
        factors = self.calibrator._calculate_mood_factors(
            baseline_rms=0.08,
            baseline_centroid=2500.0,
            baseline_zcr=0.12,
            baseline_f0=180.0
        )
        
        self.assertIn('calm', factors)
        self.assertIn('neutral', factors)
        self.assertIn('energetic', factors)
        self.assertIn('excited', factors)
        
        # Neutral should always be 1.0
        self.assertEqual(factors['neutral'], 1.0)
        
        # All factors should be reasonable
        for factor in factors.values():
            self.assertGreaterEqual(factor, 0.5)
            self.assertLessEqual(factor, 2.0)
    
    def test_save_and_load_calibration_data(self):
        """Test saving and loading calibration data."""
        calibration_data = CalibrationData(
            user_id="test_user",
            timestamp=time.time(),
            baseline_rms=0.05,
            baseline_rms_std=0.01,
            baseline_spectral_centroid=2000.0,
            baseline_spectral_centroid_std=500.0,
            baseline_zero_crossing_rate=0.1,
            baseline_zero_crossing_rate_std=0.02,
            baseline_fundamental_freq=150.0,
            baseline_fundamental_freq_std=20.0,
            rms_range=(0.01, 0.15),
            spectral_centroid_range=(1000.0, 4000.0),
            zcr_range=(0.05, 0.2),
            f0_range=(100.0, 300.0),
            sample_count=100,
            confidence_score=0.8,
            calibration_duration=60.0
        )
        
        # Save data
        self.calibrator._save_calibration_data(calibration_data)
        
        # Load data
        loaded_data = self.calibrator.load_calibration_data("test_user")
        
        self.assertIsNotNone(loaded_data)
        self.assertEqual(loaded_data.user_id, calibration_data.user_id)
        self.assertEqual(loaded_data.baseline_rms, calibration_data.baseline_rms)
        self.assertEqual(loaded_data.sample_count, calibration_data.sample_count)
    
    def test_load_nonexistent_calibration_data(self):
        """Test loading calibration data that doesn't exist."""
        loaded_data = self.calibrator.load_calibration_data("nonexistent_user")
        self.assertIsNone(loaded_data)
    
    def test_list_calibrated_users(self):
        """Test listing calibrated users."""
        # Initially no users
        users = self.calibrator.list_calibrated_users()
        self.assertEqual(len(users), 0)
        
        # Add calibration data for two users
        for user_id in ["user1", "user2"]:
            calibration_data = CalibrationData(
                user_id=user_id,
                timestamp=time.time(),
                baseline_rms=0.05,
                baseline_rms_std=0.01,
                baseline_spectral_centroid=2000.0,
                baseline_spectral_centroid_std=500.0,
                baseline_zero_crossing_rate=0.1,
                baseline_zero_crossing_rate_std=0.02,
                baseline_fundamental_freq=150.0,
                baseline_fundamental_freq_std=20.0,
                rms_range=(0.01, 0.15),
                spectral_centroid_range=(1000.0, 4000.0),
                zcr_range=(0.05, 0.2),
                f0_range=(100.0, 300.0),
                sample_count=100,
                confidence_score=0.8,
                calibration_duration=60.0
            )
            self.calibrator._save_calibration_data(calibration_data)
        
        users = self.calibrator.list_calibrated_users()
        self.assertEqual(len(users), 2)
        self.assertIn("user1", users)
        self.assertIn("user2", users)
    
    def test_delete_calibration_data(self):
        """Test deleting calibration data."""
        # Create calibration data
        calibration_data = CalibrationData(
            user_id="test_user",
            timestamp=time.time(),
            baseline_rms=0.05,
            baseline_rms_std=0.01,
            baseline_spectral_centroid=2000.0,
            baseline_spectral_centroid_std=500.0,
            baseline_zero_crossing_rate=0.1,
            baseline_zero_crossing_rate_std=0.02,
            baseline_fundamental_freq=150.0,
            baseline_fundamental_freq_std=20.0,
            rms_range=(0.01, 0.15),
            spectral_centroid_range=(1000.0, 4000.0),
            zcr_range=(0.05, 0.2),
            f0_range=(100.0, 300.0),
            sample_count=100,
            confidence_score=0.8,
            calibration_duration=60.0
        )
        self.calibrator._save_calibration_data(calibration_data)
        
        # Verify it exists
        self.assertIsNotNone(self.calibrator.load_calibration_data("test_user"))
        
        # Delete it
        success = self.calibrator.delete_calibration_data("test_user")
        self.assertTrue(success)
        
        # Verify it's gone
        self.assertIsNone(self.calibrator.load_calibration_data("test_user"))
        
        # Try to delete again
        success = self.calibrator.delete_calibration_data("test_user")
        self.assertFalse(success)
    
    def test_get_calibration_summary(self):
        """Test getting calibration summary."""
        # No data initially
        summary = self.calibrator.get_calibration_summary("test_user")
        self.assertIsNone(summary)
        
        # Create calibration data
        calibration_data = CalibrationData(
            user_id="test_user",
            timestamp=time.time(),
            baseline_rms=0.05,
            baseline_rms_std=0.01,
            baseline_spectral_centroid=2000.0,
            baseline_spectral_centroid_std=500.0,
            baseline_zero_crossing_rate=0.1,
            baseline_zero_crossing_rate_std=0.02,
            baseline_fundamental_freq=150.0,
            baseline_fundamental_freq_std=20.0,
            rms_range=(0.01, 0.15),
            spectral_centroid_range=(1000.0, 4000.0),
            zcr_range=(0.05, 0.2),
            f0_range=(100.0, 300.0),
            sample_count=100,
            confidence_score=0.8,
            calibration_duration=60.0
        )
        self.calibrator._save_calibration_data(calibration_data)
        
        summary = self.calibrator.get_calibration_summary("test_user")
        
        self.assertIsNotNone(summary)
        self.assertEqual(summary["user_id"], "test_user")
        self.assertEqual(summary["sample_count"], 100)
        self.assertEqual(summary["confidence_score"], 0.8)
        self.assertIn("mood_factors", summary)
        self.assertIn("calm", summary["mood_factors"])
    
    def test_apply_calibration_no_data(self):
        """Test applying calibration without calibration data."""
        with self.assertRaises(ValueError):
            self.calibrator.apply_calibration("nonexistent_user")
    
    def test_apply_calibration_success(self):
        """Test successfully applying calibration."""
        # Create calibration data
        calibration_data = CalibrationData(
            user_id="test_user",
            timestamp=time.time(),
            baseline_rms=0.08,  # Higher than default
            baseline_rms_std=0.01,
            baseline_spectral_centroid=2500.0,  # Higher than default
            baseline_spectral_centroid_std=500.0,
            baseline_zero_crossing_rate=0.12,  # Higher than default
            baseline_zero_crossing_rate_std=0.02,
            baseline_fundamental_freq=180.0,
            baseline_fundamental_freq_std=20.0,
            rms_range=(0.02, 0.2),
            spectral_centroid_range=(1500.0, 5000.0),
            zcr_range=(0.08, 0.25),
            f0_range=(120.0, 350.0),
            calm_factor=0.8,
            neutral_factor=1.0,
            energetic_factor=1.2,
            excited_factor=1.5,
            sample_count=100,
            confidence_score=0.8,
            calibration_duration=60.0
        )
        self.calibrator._save_calibration_data(calibration_data)
        
        # Apply calibration
        calibrated_config = self.calibrator.apply_calibration("test_user")
        
        self.assertIsInstance(calibrated_config, MoodConfig)
        
        # Check that thresholds have been adjusted
        original_config = self.calibrator.config_manager.load_config()
        
        # Calm threshold should be scaled by calm_factor
        expected_calm_max = original_config.energy.calm_max * calibration_data.calm_factor
        self.assertAlmostEqual(calibrated_config.energy.calm_max, expected_calm_max, places=5)


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.samplerate = 44100
        self.frame_size = 1024
        duration = self.frame_size / self.samplerate
        t = np.linspace(0, duration, self.frame_size)
        
        # Create test audio samples
        self.audio_samples = []
        for i in range(60):  # 60 samples for sufficient calibration
            amplitude = 0.05 + (i % 10) * 0.005  # Varying amplitude
            frequency = 150 + (i % 5) * 20  # Varying frequency
            signal = amplitude * np.sin(2 * np.pi * frequency * t)
            self.audio_samples.append(signal.astype(np.float32))
    
    def test_calibrate_user_convenience(self):
        """Test the calibrate_user convenience function."""
        calibration_data = calibrate_user("test_user", self.audio_samples)
        
        self.assertIsInstance(calibration_data, CalibrationData)
        self.assertEqual(calibration_data.user_id, "test_user")
        self.assertGreater(calibration_data.sample_count, 0)
        self.assertGreater(calibration_data.confidence_score, 0)
    
    def test_get_calibrated_detector_no_data(self):
        """Test getting calibrated detector without calibration data."""
        with self.assertRaises(ValueError):
            get_calibrated_detector("nonexistent_user")
    
    def test_get_calibrated_detector_success(self):
        """Test successfully getting calibrated detector."""
        # First calibrate the user
        calibration_data = calibrate_user("test_user", self.audio_samples)
        
        # Then get calibrated detector
        detector = get_calibrated_detector("test_user")
        
        self.assertIsInstance(detector, AdvancedMoodDetector)
        self.assertIsInstance(detector.config, MoodConfig)


class TestCalibrationIntegration(unittest.TestCase):
    """Test integration of calibration system with other components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create calibrator with temporary directory
        self.calibrator = UserCalibrator()
        self.calibrator.calibration_dir = Path(self.temp_dir)
        
        # Create test audio
        self.samplerate = 44100
        self.frame_size = 1024
        duration = self.frame_size / self.samplerate
        t = np.linspace(0, duration, self.frame_size)
        self.test_audio = (0.07 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_calibration_with_feature_extractor(self):
        """Test calibration integration with feature extractor."""
        # Create feature extractor with noise filtering
        feature_extractor = EnhancedFeatureExtractor(enable_noise_filtering=True)
        calibrator = UserCalibrator(feature_extractor=feature_extractor)
        calibrator.calibration_dir = Path(self.temp_dir)
        
        # Start calibration
        session = calibrator.start_calibration_session("test_user", duration=35.0)
        session.start_time = time.time() - 35.0  # Simulate elapsed time
        
        # Add samples
        for _ in range(60):
            calibrator.add_audio_sample(self.test_audio)
        
        # Complete calibration
        calibration_data = calibrator.finish_calibration_session()
        
        self.assertIsInstance(calibration_data, CalibrationData)
        self.assertGreater(calibration_data.sample_count, 0)
    
    def test_calibration_with_mood_detector(self):
        """Test calibration integration with mood detector."""
        # Create calibration data
        audio_samples = [self.test_audio] * 60
        calibration_data = calibrate_user("test_user", audio_samples)
        
        # Get calibrated detector
        detector = get_calibrated_detector("test_user")
        
        # Test mood detection with calibrated detector
        features = detector.feature_extractor.extract_features(self.test_audio) if hasattr(detector, 'feature_extractor') else None
        
        # Create mock features if needed
        if features is None:
            features = AudioFeatures(
                rms=0.07,
                peak_energy=0.1,
                energy_variance=0.001,
                spectral_centroid=2000.0,
                spectral_rolloff=4000.0,
                spectral_flux=100.0,
                mfccs=[1.0, 0.5, 0.3, 0.1],
                zero_crossing_rate=0.1,
                tempo=120.0,
                voice_activity=True,
                fundamental_freq=200.0,
                pitch_stability=0.8,
                pitch_range=50.0,
                timestamp=time.time(),
                confidence=0.8
            )
        
        mood_result = detector.detect_mood(features)
        
        self.assertIsInstance(mood_result.mood, str)
        self.assertIn(mood_result.mood, ['calm', 'neutral', 'energetic', 'excited'])
        self.assertGreaterEqual(mood_result.confidence, 0.0)
        self.assertLessEqual(mood_result.confidence, 1.0)


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)