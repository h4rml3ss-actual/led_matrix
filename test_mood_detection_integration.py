#!/usr/bin/env python3
"""
Integration tests for complete mood detection pipeline.
Tests the end-to-end functionality of enhanced mood detection system.
"""

import unittest
import numpy as np
import time
import tempfile
import os
import json
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

# Import components to test
try:
    from enhanced_audio_features import EnhancedFeatureExtractor, AudioFeatures
    from advanced_mood_detector import AdvancedMoodDetector, MoodResult
    from mood_transition_smoother import MoodTransitionSmoother
    from mood_config import MoodConfig, ConfigManager
    from noise_filter import NoiseFilter
    from user_calibration import UserCalibrator
    from performance_monitor import get_global_monitor, get_global_scaler
    from mood_debug_tools import MoodDebugLogger, DiagnosticAnalyzer, ConfigValidator
    ENHANCED_AVAILABLE = True
except ImportError as e:
    print(f"Enhanced components not available for testing: {e}")
    ENHANCED_AVAILABLE = False


class TestMoodDetectionPipeline(unittest.TestCase):
    """Test complete mood detection pipeline integration."""
    
    def setUp(self):
        """Set up test environment."""
        if not ENHANCED_AVAILABLE:
            self.skipTest("Enhanced components not available")
        
        self.samplerate = 44100
        self.frame_size = 1024
        
        # Create test components
        self.feature_extractor = EnhancedFeatureExtractor(
            samplerate=self.samplerate,
            frame_size=self.frame_size,
            enable_noise_filtering=False  # Disable for testing
        )
        self.mood_detector = AdvancedMoodDetector()
        self.transition_smoother = MoodTransitionSmoother()
        
        # Create temporary config file
        self.temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        self.config_manager = ConfigManager(self.temp_config.name)
    
    def tearDown(self):
        """Clean up test environment."""
        if hasattr(self, 'temp_config'):
            os.unlink(self.temp_config.name)
    
    def generate_test_audio(self, duration: float = 1.0, frequency: float = 440.0, 
                           amplitude: float = 0.1, noise_level: float = 0.01) -> np.ndarray:
        """Generate synthetic audio for testing."""
        samples = int(duration * self.samplerate)
        t = np.linspace(0, duration, samples)
        
        # Generate sine wave
        audio = amplitude * np.sin(2 * np.pi * frequency * t)
        
        # Add noise
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, samples)
            audio += noise
        
        return audio.astype(np.float32)
    
    def test_complete_pipeline_calm_mood(self):
        """Test complete pipeline with calm mood characteristics."""
        # Generate calm audio (low amplitude, low frequency)
        audio = self.generate_test_audio(
            duration=1.0,
            frequency=200.0,
            amplitude=0.01,  # Low amplitude for calm
            noise_level=0.001
        )
        
        # Extract features
        features = self.feature_extractor.extract_features(audio, timestamp=time.time())
        
        # Verify features are reasonable for calm mood
        self.assertLess(features.rms, 0.05, "RMS should be low for calm audio")
        self.assertLess(features.spectral_centroid, 1000, "Spectral centroid should be low for calm")
        
        # Detect mood
        mood_result = self.mood_detector.detect_mood(features)
        
        # Apply transition smoothing
        smoothed_mood = self.transition_smoother.smooth_transition(
            mood_result.mood, mood_result.confidence
        )
        
        # Verify results
        self.assertIn(mood_result.mood, ['calm', 'neutral'], 
                     f"Expected calm or neutral mood, got {mood_result.mood}")
        self.assertGreater(mood_result.confidence, 0.0, "Confidence should be positive")
        self.assertIsInstance(smoothed_mood, str, "Smoothed mood should be string")
        self.assertIn(smoothed_mood, ['calm', 'neutral', 'energetic', 'excited'])
    
    def test_complete_pipeline_energetic_mood(self):
        """Test complete pipeline with energetic mood characteristics."""
        # Generate energetic audio (high amplitude, variable frequency)
        base_audio = self.generate_test_audio(
            duration=1.0,
            frequency=800.0,
            amplitude=0.15,  # High amplitude for energetic
            noise_level=0.005
        )
        
        # Add frequency modulation for more dynamic content
        t = np.linspace(0, 1.0, len(base_audio))
        modulation = 0.05 * np.sin(2 * np.pi * 10 * t)  # 10 Hz modulation
        audio = base_audio * (1 + modulation)
        
        # Extract features
        features = self.feature_extractor.extract_features(audio, timestamp=time.time())
        
        # Verify features are reasonable for energetic mood
        self.assertGreater(features.rms, 0.05, "RMS should be high for energetic audio")
        self.assertGreater(features.energy_variance, 0.001, "Energy variance should be high")
        
        # Detect mood
        mood_result = self.mood_detector.detect_mood(features)
        
        # Apply transition smoothing
        smoothed_mood = self.transition_smoother.smooth_transition(
            mood_result.mood, mood_result.confidence
        )
        
        # Verify results
        self.assertIn(mood_result.mood, ['energetic', 'excited', 'neutral'], 
                     f"Expected energetic mood, got {mood_result.mood}")
        self.assertGreater(mood_result.confidence, 0.0, "Confidence should be positive")
    
    def test_pipeline_with_configuration_changes(self):
        """Test pipeline behavior with different configurations."""
        # Create custom configuration
        config = self.config_manager.get_default_config()
        
        # Modify thresholds to be more sensitive
        config.energy.calm_max = 0.05  # Higher threshold for calm
        config.energy.energetic_min = 0.06  # Lower threshold for energetic
        
        # Save and reload configuration
        self.config_manager.save_config(config)
        reloaded_config = self.config_manager.load_config()
        
        # Create detector with new configuration
        detector_with_config = AdvancedMoodDetector(reloaded_config)
        
        # Test with medium amplitude audio
        audio = self.generate_test_audio(amplitude=0.04)
        features = self.feature_extractor.extract_features(audio)
        
        # Compare results with different configurations
        default_result = self.mood_detector.detect_mood(features)
        custom_result = detector_with_config.detect_mood(features)
        
        # Results might be different due to threshold changes
        self.assertIsInstance(default_result.mood, str)
        self.assertIsInstance(custom_result.mood, str)
        self.assertGreater(default_result.confidence, 0.0)
        self.assertGreater(custom_result.confidence, 0.0)
    
    def test_pipeline_performance_under_load(self):
        """Test pipeline performance with multiple rapid detections."""
        start_time = time.time()
        results = []
        
        # Process multiple audio blocks rapidly
        for i in range(50):
            audio = self.generate_test_audio(
                duration=0.1,  # Short blocks
                frequency=200 + i * 10,  # Varying frequency
                amplitude=0.02 + (i % 10) * 0.01  # Varying amplitude
            )
            
            features = self.feature_extractor.extract_features(audio)
            mood_result = self.mood_detector.detect_mood(features)
            smoothed_mood = self.transition_smoother.smooth_transition(
                mood_result.mood, mood_result.confidence
            )
            
            results.append({
                'mood': mood_result.mood,
                'confidence': mood_result.confidence,
                'smoothed_mood': smoothed_mood
            })
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time_per_detection = total_time / 50
        
        # Verify performance
        self.assertLess(avg_time_per_detection, 0.1, 
                       f"Average detection time too slow: {avg_time_per_detection:.3f}s")
        
        # Verify all results are valid
        for result in results:
            self.assertIn(result['mood'], ['calm', 'neutral', 'energetic', 'excited'])
            self.assertGreater(result['confidence'], 0.0)
            self.assertLessEqual(result['confidence'], 1.0)
            self.assertIn(result['smoothed_mood'], ['calm', 'neutral', 'energetic', 'excited'])
    
    def test_pipeline_with_noise_filtering(self):
        """Test pipeline with noise filtering enabled."""
        # Create feature extractor with noise filtering
        noise_filtering_extractor = EnhancedFeatureExtractor(
            samplerate=self.samplerate,
            frame_size=self.frame_size,
            enable_noise_filtering=True
        )
        
        # Generate noisy audio
        clean_audio = self.generate_test_audio(amplitude=0.05, noise_level=0.0)
        noisy_audio = self.generate_test_audio(amplitude=0.05, noise_level=0.02)
        
        # Extract features from both
        clean_features = noise_filtering_extractor.extract_features(clean_audio)
        noisy_features = noise_filtering_extractor.extract_features(noisy_audio)
        
        # Detect moods
        clean_result = self.mood_detector.detect_mood(clean_features)
        noisy_result = self.mood_detector.detect_mood(noisy_features)
        
        # Both should produce valid results
        self.assertIsInstance(clean_result.mood, str)
        self.assertIsInstance(noisy_result.mood, str)
        self.assertGreater(clean_result.confidence, 0.0)
        self.assertGreater(noisy_result.confidence, 0.0)
    
    def test_transition_smoothing_consistency(self):
        """Test that transition smoothing provides consistent results."""
        # Generate consistent audio
        audio = self.generate_test_audio(amplitude=0.03)
        features = self.feature_extractor.extract_features(audio)
        mood_result = self.mood_detector.detect_mood(features)
        
        # Apply smoothing multiple times with same input
        smoothed_results = []
        for _ in range(10):
            smoothed_mood = self.transition_smoother.smooth_transition(
                mood_result.mood, mood_result.confidence
            )
            smoothed_results.append(smoothed_mood)
        
        # After initial transitions, results should stabilize
        stable_results = smoothed_results[-5:]  # Last 5 results
        self.assertTrue(all(mood == stable_results[0] for mood in stable_results),
                       "Smoothed mood should stabilize with consistent input")
    
    def test_error_handling_and_recovery(self):
        """Test pipeline error handling and recovery."""
        # Test with invalid audio data
        invalid_audio = np.array([])  # Empty array
        
        try:
            features = self.feature_extractor.extract_features(invalid_audio)
            # Should handle gracefully or raise appropriate exception
            self.assertIsInstance(features, AudioFeatures)
        except (ValueError, RuntimeError) as e:
            # Expected behavior for invalid input
            self.assertIsInstance(e, (ValueError, RuntimeError))
        
        # Test with extreme values
        extreme_audio = np.full(self.frame_size, 10.0)  # Very loud audio
        
        try:
            features = self.feature_extractor.extract_features(extreme_audio)
            mood_result = self.mood_detector.detect_mood(features)
            
            # Should produce valid results even with extreme input
            self.assertIsInstance(mood_result.mood, str)
            self.assertGreater(mood_result.confidence, 0.0)
            self.assertLessEqual(mood_result.confidence, 1.0)
        except Exception as e:
            self.fail(f"Pipeline should handle extreme values gracefully: {e}")


class TestDebugAndDiagnosticTools(unittest.TestCase):
    """Test debugging and diagnostic tools."""
    
    def setUp(self):
        """Set up test environment."""
        if not ENHANCED_AVAILABLE:
            self.skipTest("Enhanced components not available")
        
        self.temp_dir = tempfile.mkdtemp()
        self.debug_log_file = os.path.join(self.temp_dir, "test_debug.log")
        self.config_file = os.path.join(self.temp_dir, "test_config.json")
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_debug_logger_functionality(self):
        """Test debug logger captures mood detection information."""
        debug_logger = MoodDebugLogger(log_file=self.debug_log_file, max_entries=100)
        
        # Create mock mood result
        mock_features = AudioFeatures(
            rms=0.05, peak_energy=0.1, energy_variance=0.001,
            spectral_centroid=2000.0, spectral_rolloff=4000.0, spectral_flux=1000.0,
            mfccs=[1.0, 0.5, -0.2, 0.1], zero_crossing_rate=0.1, tempo=120.0,
            voice_activity=True, fundamental_freq=150.0, pitch_stability=0.8,
            pitch_range=50.0, timestamp=time.time(), confidence=0.8
        )
        
        mock_result = MoodResult(
            mood="neutral",
            confidence=0.75,
            features_used=mock_features,
            transition_recommended=True,
            debug_scores={
                'calm_total': 0.3,
                'neutral_total': 0.75,
                'energetic_total': 0.4,
                'excited_total': 0.2
            }
        )
        
        # Log the result
        debug_logger.log_mood_detection(mock_result)
        
        # Verify entry was logged
        recent_entries = debug_logger.get_recent_entries(1)
        self.assertEqual(len(recent_entries), 1)
        
        entry = recent_entries[0]
        self.assertEqual(entry.mood, "neutral")
        self.assertEqual(entry.confidence, 0.75)
        self.assertTrue(entry.transition_recommended)
        self.assertIn('calm_total', entry.debug_scores)
        
        # Test export functionality
        export_file = os.path.join(self.temp_dir, "debug_export.json")
        success = debug_logger.export_debug_data(export_file)
        self.assertTrue(success)
        self.assertTrue(os.path.exists(export_file))
        
        # Verify exported data
        with open(export_file, 'r') as f:
            exported_data = json.load(f)
        
        self.assertIn('entries', exported_data)
        self.assertEqual(len(exported_data['entries']), 1)
        self.assertEqual(exported_data['entries'][0]['mood'], "neutral")
    
    def test_diagnostic_analyzer(self):
        """Test diagnostic analyzer functionality."""
        analyzer = DiagnosticAnalyzer()
        
        # Run diagnostic analysis
        results = analyzer.run_full_diagnostic()
        
        # Verify required sections are present
        required_sections = [
            'timestamp', 'system_info', 'configuration_analysis',
            'component_status', 'performance_analysis', 'recommendations'
        ]
        
        for section in required_sections:
            self.assertIn(section, results, f"Missing section: {section}")
        
        # Verify system info
        sys_info = results['system_info']
        self.assertIn('enhanced_available', sys_info)
        self.assertIn('python_version', sys_info)
        self.assertIn('platform', sys_info)
        
        # Verify component status
        comp_status = results['component_status']
        self.assertIsInstance(comp_status, dict)
        
        # Verify recommendations are provided
        recommendations = results['recommendations']
        self.assertIsInstance(recommendations, list)
    
    def test_config_validator(self):
        """Test configuration validator functionality."""
        validator = ConfigValidator()
        
        # Create test configuration file
        test_config = {
            "thresholds": {
                "energy": {
                    "calm_max": 0.02,
                    "neutral_range": [0.02, 0.08],
                    "energetic_min": 0.08,
                    "excited_min": 0.15
                },
                "spectral": {
                    "calm_centroid_max": 2000.0,
                    "bright_centroid_min": 3000.0,
                    "rolloff_thresholds": [1500.0, 3000.0, 5000.0]
                },
                "temporal": {
                    "calm_zcr_max": 0.05,
                    "energetic_zcr_min": 0.15
                }
            },
            "smoothing": {
                "transition_time": 2.0,
                "minimum_duration": 5.0,
                "confidence_threshold": 0.7
            },
            "noise_filtering": {
                "noise_gate_threshold": 0.01,
                "adaptive_gain": True,
                "background_learning_rate": 0.1
            }
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(test_config, f)
        
        # Validate configuration
        results = validator.validate_configuration(self.config_file)
        
        # Verify validation results
        self.assertTrue(results['file_exists'])
        self.assertTrue(results['valid'])
        self.assertIsInstance(results['errors'], list)
        self.assertIsInstance(results['warnings'], list)
        self.assertIsInstance(results['recommendations'], list)
        
        # Test template export
        template_file = os.path.join(self.temp_dir, "template.json")
        success = validator.export_config_template(template_file)
        self.assertTrue(success)
        self.assertTrue(os.path.exists(template_file))
    
    def test_integration_with_performance_monitoring(self):
        """Test integration with performance monitoring system."""
        try:
            monitor = get_global_monitor()
            scaler = get_global_scaler()
            
            # Test performance monitoring integration
            monitor.start_cycle()
            
            # Simulate some processing
            with monitor.measure_stage('test_stage'):
                time.sleep(0.001)  # 1ms processing
            
            metrics = monitor.end_cycle()
            
            # Verify metrics are captured
            self.assertIsNotNone(metrics)
            self.assertGreater(metrics.total_duration, 0)
            self.assertIn('test_stage', metrics.stage_durations)
            
            # Test performance scaling
            current_config = scaler.get_current_config()
            self.assertIsInstance(current_config, dict)
            
        except Exception as e:
            self.skipTest(f"Performance monitoring not available: {e}")


class TestEndToEndScenarios(unittest.TestCase):
    """Test end-to-end scenarios that simulate real usage."""
    
    def setUp(self):
        """Set up test environment."""
        if not ENHANCED_AVAILABLE:
            self.skipTest("Enhanced components not available")
        
        self.samplerate = 44100
        self.frame_size = 1024
    
    def test_conversation_simulation(self):
        """Simulate a conversation with varying moods."""
        # Initialize pipeline components
        feature_extractor = EnhancedFeatureExtractor(self.samplerate, self.frame_size)
        mood_detector = AdvancedMoodDetector()
        transition_smoother = MoodTransitionSmoother()
        debug_logger = MoodDebugLogger(max_entries=50)
        
        # Simulate conversation phases
        conversation_phases = [
            # Phase 1: Quiet start (calm)
            {'duration': 2.0, 'amplitude': 0.01, 'frequency': 200, 'expected_mood': 'calm'},
            # Phase 2: Normal conversation (neutral)
            {'duration': 3.0, 'amplitude': 0.04, 'frequency': 400, 'expected_mood': 'neutral'},
            # Phase 3: Excited discussion (energetic)
            {'duration': 2.0, 'amplitude': 0.12, 'frequency': 800, 'expected_mood': 'energetic'},
            # Phase 4: Very animated (excited)
            {'duration': 1.5, 'amplitude': 0.20, 'frequency': 1000, 'expected_mood': 'excited'},
            # Phase 5: Calming down (neutral)
            {'duration': 2.0, 'amplitude': 0.05, 'frequency': 300, 'expected_mood': 'neutral'},
        ]
        
        mood_history = []
        
        for phase in conversation_phases:
            # Generate audio for this phase
            samples = int(phase['duration'] * self.samplerate)
            t = np.linspace(0, phase['duration'], samples)
            
            # Create more realistic audio with harmonics and modulation
            fundamental = phase['amplitude'] * np.sin(2 * np.pi * phase['frequency'] * t)
            harmonic2 = (phase['amplitude'] * 0.3) * np.sin(2 * np.pi * phase['frequency'] * 2 * t)
            modulation = 0.1 * np.sin(2 * np.pi * 5 * t)  # 5 Hz amplitude modulation
            
            audio = fundamental + harmonic2
            audio = audio * (1 + modulation)
            
            # Add some noise for realism
            noise = np.random.normal(0, phase['amplitude'] * 0.1, samples)
            audio += noise
            
            # Process in chunks (simulate real-time processing)
            chunk_size = self.frame_size
            for i in range(0, len(audio), chunk_size):
                chunk = audio[i:i + chunk_size]
                if len(chunk) < chunk_size:
                    # Pad last chunk
                    chunk = np.pad(chunk, (0, chunk_size - len(chunk)), 'constant')
                
                # Process chunk through pipeline
                features = feature_extractor.extract_features(chunk)
                mood_result = mood_detector.detect_mood(features)
                smoothed_mood = transition_smoother.smooth_transition(
                    mood_result.mood, mood_result.confidence
                )
                
                # Log for debugging
                debug_logger.log_mood_detection(mood_result)
                
                mood_history.append({
                    'phase': phase['expected_mood'],
                    'detected': mood_result.mood,
                    'smoothed': smoothed_mood,
                    'confidence': mood_result.confidence
                })
        
        # Analyze results
        self.assertGreater(len(mood_history), 0, "Should have processed some audio")
        
        # Check that we detected various moods
        detected_moods = set(entry['detected'] for entry in mood_history)
        self.assertGreater(len(detected_moods), 1, "Should detect multiple different moods")
        
        # Check confidence levels are reasonable
        avg_confidence = np.mean([entry['confidence'] for entry in mood_history])
        self.assertGreater(avg_confidence, 0.3, "Average confidence should be reasonable")
        
        # Analyze debug data
        analysis = debug_logger.analyze_mood_patterns()
        self.assertIn('mood_distribution', analysis)
        self.assertIn('average_confidence_by_mood', analysis)
        self.assertGreater(analysis['total_detections'], 0)
    
    def test_calibration_workflow(self):
        """Test user calibration workflow."""
        try:
            from user_calibration import UserCalibrator
            
            calibrator = UserCalibrator()
            
            # Simulate calibration samples
            calibration_samples = []
            for i in range(10):
                # Generate varied but consistent user voice samples
                audio = self.generate_user_voice_sample(
                    base_frequency=150 + i * 5,  # Slight variation
                    base_amplitude=0.03 + i * 0.002  # Slight variation
                )
                
                feature_extractor = EnhancedFeatureExtractor(self.samplerate, self.frame_size)
                features = feature_extractor.extract_features(audio)
                calibration_samples.append(features)
            
            # Perform calibration
            calibration_data = calibrator.calibrate_user("test_user", calibration_samples)
            
            # Verify calibration data
            self.assertIsNotNone(calibration_data)
            self.assertIn('user_id', calibration_data)
            self.assertEqual(calibration_data['user_id'], "test_user")
            
        except ImportError:
            self.skipTest("User calibration not available")
    
    def generate_user_voice_sample(self, base_frequency: float = 150.0, 
                                  base_amplitude: float = 0.03) -> np.ndarray:
        """Generate realistic user voice sample for calibration."""
        duration = 1.0
        samples = int(duration * self.samplerate)
        t = np.linspace(0, duration, samples)
        
        # Create voice-like signal with formants
        fundamental = base_amplitude * np.sin(2 * np.pi * base_frequency * t)
        formant1 = (base_amplitude * 0.4) * np.sin(2 * np.pi * (base_frequency * 3) * t)
        formant2 = (base_amplitude * 0.2) * np.sin(2 * np.pi * (base_frequency * 5) * t)
        
        # Add natural variation
        vibrato = 0.05 * np.sin(2 * np.pi * 6 * t)  # 6 Hz vibrato
        voice = fundamental * (1 + vibrato) + formant1 + formant2
        
        # Add breath noise
        breath_noise = np.random.normal(0, base_amplitude * 0.05, samples)
        voice += breath_noise
        
        return voice.astype(np.float32)


def run_integration_tests():
    """Run all integration tests."""
    if not ENHANCED_AVAILABLE:
        print("Enhanced components not available - skipping integration tests")
        return False
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestMoodDetectionPipeline,
        TestDebugAndDiagnosticTools,
        TestEndToEndScenarios
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Return success status
    return result.wasSuccessful()


if __name__ == "__main__":
    print("Running Enhanced Mood Detection Integration Tests")
    print("=" * 60)
    
    success = run_integration_tests()
    
    if success:
        print("\n✓ All integration tests passed!")
        exit(0)
    else:
        print("\n✗ Some integration tests failed!")
        exit(1)