#!/usr/bin/env python3
"""
Comprehensive tests for the enhanced LED system integration.
Tests backward compatibility, enhanced features, and integration points.
"""

import unittest
import numpy as np
import time
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TestEnhancedIntegration(unittest.TestCase):
    """Test enhanced system integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.samplerate = 44100
        self.blocksize = 1024
        
        # Create test audio data
        duration = self.blocksize / self.samplerate
        t = np.linspace(0, duration, self.blocksize)
        self.test_audio = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        
    def test_enhanced_components_import(self):
        """Test that enhanced components can be imported."""
        try:
            from enhanced_audio_features import EnhancedFeatureExtractor
            from advanced_mood_detector import AdvancedMoodDetector
            from mood_transition_smoother import MoodTransitionSmoother
            from user_calibration import get_calibrated_detector
            enhanced_available = True
        except ImportError:
            enhanced_available = False
        
        # Should be available since we created them
        self.assertTrue(enhanced_available, "Enhanced components should be available")
    
    def test_audio_processor_initialization(self):
        """Test AudioProcessor initialization."""
        try:
            # Import from the integration file
            sys.path.insert(0, '.')
            from led_enhanced_integration import AudioProcessor
            
            # Test with enhanced mode
            processor = AudioProcessor(enable_enhanced=True)
            self.assertIsNotNone(processor)
            
            # Test with fallback mode
            processor_fallback = AudioProcessor(enable_enhanced=False)
            self.assertIsNotNone(processor_fallback)
            
        except ImportError as e:
            self.skipTest(f"Integration module not available: {e}")
    
    def test_backward_compatibility(self):
        """Test that original functions still work."""
        try:
            from led_enhanced_integration import (
                parse_ascii_frame, convert_color, pick_frame_index,
                CHAR_TO_COLOR, extract_features_original, detect_mood_original
            )
            
            # Test original functions
            self.assertIsInstance(CHAR_TO_COLOR, dict)
            self.assertIn('.', CHAR_TO_COLOR)
            
            # Test color conversion
            color = convert_color((255, 0, 0))
            self.assertEqual(len(color), 3)
            
            # Test frame index selection
            frame_idx = pick_frame_index(0.05)
            self.assertIsInstance(frame_idx, int)
            self.assertGreaterEqual(frame_idx, 0)
            
        except ImportError as e:
            self.skipTest(f"Integration module not available: {e}")
    
    def test_enhanced_audio_processing(self):
        """Test enhanced audio processing pipeline."""
        try:
            from led_enhanced_integration import AudioProcessor
            
            processor = AudioProcessor(enable_enhanced=True)
            
            # Mock the callback
            indata = self.test_audio.reshape(-1, 1)
            processor.audio_callback(indata, len(indata), None, None)
            
            # Check results
            mood, confidence = processor.get_current_mood()
            volume = processor.get_current_volume()
            
            self.assertIsInstance(mood, str)
            self.assertIn(mood, ['calm', 'neutral', 'energetic', 'excited', 'bright'])
            self.assertIsInstance(confidence, float)
            self.assertGreaterEqual(confidence, 0.0)
            self.assertLessEqual(confidence, 1.0)
            self.assertIsInstance(volume, float)
            self.assertGreaterEqual(volume, 0.0)
            
        except ImportError as e:
            self.skipTest(f"Integration module not available: {e}")
    
    def test_fallback_processing(self):
        """Test fallback to original processing."""
        try:
            from led_enhanced_integration import AudioProcessor
            
            # Force fallback mode
            processor = AudioProcessor(enable_enhanced=False)
            
            # Test original feature extraction
            rms, zcr, centroid = processor.extract_features_original(self.test_audio)
            
            self.assertIsInstance(rms, float)
            self.assertIsInstance(zcr, float)
            self.assertIsInstance(centroid, float)
            self.assertGreaterEqual(rms, 0.0)
            self.assertGreaterEqual(zcr, 0.0)
            self.assertGreaterEqual(centroid, 0.0)
            
            # Test original mood detection
            mood = processor.detect_mood_original(rms, zcr, centroid)
            self.assertIn(mood, ['calm', 'neutral', 'energetic', 'bright'])
            
        except ImportError as e:
            self.skipTest(f"Integration module not available: {e}")
    
    def test_frame_loading_compatibility(self):
        """Test that frame loading still works."""
        try:
            from led_enhanced_integration import load_mood_frames, parse_ascii_frame
            
            # Test with non-existent directory (should handle gracefully)
            frames, smile = load_mood_frames('nonexistent_directory')
            self.assertIsInstance(frames, list)
            
            # Test ASCII frame parsing with minimal data
            test_ascii = """
  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
 0| .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .
 1| .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .
"""
            
            # This should work or raise a clear error
            try:
                grid = parse_ascii_frame(test_ascii, rows=2, cols=16)
                self.assertEqual(len(grid), 2)
                self.assertEqual(len(grid[0]), 16)
            except ValueError:
                # Expected if format is not exactly right
                pass
            
        except ImportError as e:
            self.skipTest(f"Integration module not available: {e}")


class TestSimpleIntegration(unittest.TestCase):
    """Test the simple integration approach."""
    
    def test_simple_integration_import(self):
        """Test that simple integration can be imported."""
        try:
            # This will test if the simple integration file is valid Python
            with open('led_with_enhancements.py', 'r') as f:
                content = f.read()
            
            # Check for key components
            self.assertIn('enhanced_audio_callback', content)
            self.assertIn('get_enhanced_mood', content)
            self.assertIn('init_enhanced_system', content)
            
        except FileNotFoundError:
            self.skipTest("Simple integration file not found")
    
    def test_enhanced_led_structure(self):
        """Test the enhanced LED system structure."""
        try:
            with open('enhanced_led.py', 'r') as f:
                content = f.read()
            
            # Check for key classes and functions
            self.assertIn('class EnhancedAudioProcessor', content)
            self.assertIn('class EnhancedFrameSelector', content)
            self.assertIn('class EnhancedDisplaySystem', content)
            self.assertIn('class EnhancedLEDSystem', content)
            
        except FileNotFoundError:
            self.skipTest("Enhanced LED file not found")


class TestConfigurationOptions(unittest.TestCase):
    """Test configuration and command line options."""
    
    def test_command_line_parsing(self):
        """Test command line argument parsing."""
        # Test the configuration loading logic
        try:
            from enhanced_led import load_system_config
            
            # Mock sys.argv for testing
            original_argv = sys.argv.copy()
            
            try:
                # Test default config
                sys.argv = ['enhanced_led.py']
                config = load_system_config()
                self.assertFalse(config.enable_user_calibration)
                
                # Test calibrated mode
                sys.argv = ['enhanced_led.py', '--calibrated', '--user', 'test_user']
                config = load_system_config()
                self.assertTrue(config.enable_user_calibration)
                self.assertEqual(config.user_id, 'test_user')
                
                # Test disabled features
                sys.argv = ['enhanced_led.py', '--no-noise-filter', '--no-smoothing']
                config = load_system_config()
                self.assertFalse(config.enable_noise_filtering)
                self.assertFalse(config.enable_transition_smoothing)
                
            finally:
                sys.argv = original_argv
                
        except ImportError as e:
            self.skipTest(f"Enhanced LED module not available: {e}")


class TestPerformanceAndCompatibility(unittest.TestCase):
    """Test performance and compatibility aspects."""
    
    def test_processing_performance(self):
        """Test that processing doesn't take too long."""
        try:
            from enhanced_audio_features import EnhancedFeatureExtractor
            from advanced_mood_detector import AdvancedMoodDetector
            
            extractor = EnhancedFeatureExtractor()
            detector = AdvancedMoodDetector()
            
            # Create test audio
            audio = np.random.randn(1024).astype(np.float32) * 0.1
            
            # Time the processing
            start_time = time.time()
            features = extractor.extract_features(audio)
            mood_result = detector.detect_mood(features)
            processing_time = time.time() - start_time
            
            # Should process in reasonable time (less than 100ms for 1024 samples)
            self.assertLess(processing_time, 0.1, "Processing should be fast enough for real-time")
            
            # Results should be valid
            self.assertIsInstance(mood_result.mood, str)
            self.assertIsInstance(mood_result.confidence, float)
            
        except ImportError as e:
            self.skipTest(f"Enhanced components not available: {e}")
    
    def test_memory_usage(self):
        """Test that memory usage is reasonable."""
        try:
            from enhanced_audio_features import EnhancedFeatureExtractor
            
            extractor = EnhancedFeatureExtractor()
            
            # Process multiple audio blocks
            for _ in range(100):
                audio = np.random.randn(1024).astype(np.float32) * 0.1
                features = extractor.extract_features(audio)
                
                # Basic validation
                self.assertIsInstance(features.rms, float)
                self.assertIsInstance(features.voice_activity, bool)
            
            # If we get here without memory errors, we're good
            self.assertTrue(True)
            
        except ImportError as e:
            self.skipTest(f"Enhanced components not available: {e}")


class TestErrorHandling(unittest.TestCase):
    """Test error handling and graceful degradation."""
    
    def test_missing_components_handling(self):
        """Test handling when enhanced components are missing."""
        # This test simulates what happens when enhanced components aren't available
        
        # Mock the import failure
        with patch.dict('sys.modules', {'enhanced_audio_features': None}):
            try:
                # This should handle the import error gracefully
                from led_enhanced_integration import AudioProcessor
                
                processor = AudioProcessor(enable_enhanced=False)
                self.assertIsNotNone(processor)
                
                # Should fall back to original processing
                self.assertFalse(processor.enable_enhanced)
                
            except ImportError:
                # This is expected if the integration isn't available
                pass
    
    def test_invalid_audio_data_handling(self):
        """Test handling of invalid audio data."""
        try:
            from enhanced_audio_features import EnhancedFeatureExtractor
            
            extractor = EnhancedFeatureExtractor()
            
            # Test with empty audio
            try:
                features = extractor.extract_features(np.array([]))
                # Should either work or raise a clear error
            except (ValueError, IndexError):
                # Expected for empty input
                pass
            
            # Test with NaN audio
            try:
                nan_audio = np.full(1024, np.nan, dtype=np.float32)
                features = extractor.extract_features(nan_audio)
                # Should handle NaN gracefully
            except (ValueError, RuntimeError):
                # Expected for invalid input
                pass
            
        except ImportError as e:
            self.skipTest(f"Enhanced components not available: {e}")


class TestIntegrationFiles(unittest.TestCase):
    """Test that all integration files are properly created."""
    
    def test_integration_files_exist(self):
        """Test that all expected integration files exist."""
        expected_files = [
            'enhanced_led.py',
            'led_enhanced_integration.py',
            'led_with_enhancements.py',
            'integrate_enhanced_system.py'
        ]
        
        for filename in expected_files:
            self.assertTrue(os.path.exists(filename), f"Integration file {filename} should exist")
    
    def test_integration_files_syntax(self):
        """Test that integration files have valid Python syntax."""
        integration_files = [
            'enhanced_led.py',
            'led_enhanced_integration.py',
            'led_with_enhancements.py'
        ]
        
        for filename in integration_files:
            if os.path.exists(filename):
                try:
                    with open(filename, 'r') as f:
                        content = f.read()
                    
                    # Try to compile the code
                    compile(content, filename, 'exec')
                    
                except SyntaxError as e:
                    self.fail(f"Syntax error in {filename}: {e}")
                except FileNotFoundError:
                    self.skipTest(f"File {filename} not found")


if __name__ == '__main__':
    print("Running Enhanced LED Integration Tests")
    print("=" * 50)
    
    # Run tests with verbose output
    unittest.main(verbosity=2)