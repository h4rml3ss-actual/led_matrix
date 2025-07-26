#!/usr/bin/env python3
"""
Comprehensive tests for error handling and fallback mechanisms.
Tests various error scenarios and recovery procedures.
"""

import unittest
import numpy as np
import time
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

# Import the modules to test
from error_handling import (
    ErrorRecoveryManager, ErrorSeverity, ErrorEvent, SafeAudioProcessor,
    MicrophoneErrorHandler, get_global_error_manager
)


class TestErrorRecoveryManager(unittest.TestCase):
    """Test the ErrorRecoveryManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.error_manager = ErrorRecoveryManager()
        self.test_component = "test_component"
    
    def test_error_event_creation(self):
        """Test error event creation and logging."""
        test_exception = ValueError("Test error")
        
        result = self.error_manager.handle_error(
            self.test_component,
            "test_error",
            test_exception,
            ErrorSeverity.MEDIUM
        )
        
        # Should return False since no recovery strategy is registered
        self.assertFalse(result)
        
        # Check error was logged
        self.assertEqual(len(self.error_manager.error_history), 1)
        error_event = self.error_manager.error_history[0]
        self.assertEqual(error_event.component, self.test_component)
        self.assertEqual(error_event.error_type, "test_error")
        self.assertEqual(error_event.severity, ErrorSeverity.MEDIUM)
        self.assertEqual(error_event.exception, test_exception)
    
    def test_recovery_strategy_registration(self):
        """Test recovery strategy registration and execution."""
        recovery_called = False
        
        def mock_recovery_strategy(error_event):
            nonlocal recovery_called
            recovery_called = True
            return True
        
        # Register recovery strategy
        self.error_manager.register_recovery_strategy(
            self.test_component, mock_recovery_strategy
        )
        
        # Trigger error
        result = self.error_manager.handle_error(
            self.test_component,
            "test_error",
            ValueError("Test error"),
            ErrorSeverity.MEDIUM
        )
        
        # Recovery should have been called and succeeded
        self.assertTrue(result)
        self.assertTrue(recovery_called)
    
    def test_fallback_handler_registration(self):
        """Test fallback handler registration and execution."""
        fallback_called = False
        
        def mock_fallback_handler(error_event):
            nonlocal fallback_called
            fallback_called = True
        
        # Register fallback handler
        self.error_manager.register_fallback_handler(
            self.test_component, mock_fallback_handler
        )
        
        # Trigger error (no recovery strategy, so fallback should be used)
        result = self.error_manager.handle_error(
            self.test_component,
            "test_error",
            ValueError("Test error"),
            ErrorSeverity.MEDIUM
        )
        
        # Fallback should have been called
        self.assertFalse(result)  # Recovery failed, fallback used
        self.assertTrue(fallback_called)
        self.assertIn(self.test_component, self.error_manager.degraded_components)
    
    def test_consecutive_error_handling(self):
        """Test handling of consecutive errors."""
        # Generate multiple consecutive errors
        for i in range(12):  # More than the threshold
            self.error_manager.handle_error(
                self.test_component,
                "consecutive_error",
                ValueError(f"Error {i}"),
                ErrorSeverity.MEDIUM
            )
        
        # Component should be disabled after too many consecutive errors
        self.assertIn(self.test_component, self.error_manager.disabled_components)
    
    def test_component_health_evaluation(self):
        """Test component health evaluation and degradation."""
        # Generate moderate number of errors
        for i in range(6):
            self.error_manager.handle_error(
                self.test_component,
                "moderate_error",
                ValueError(f"Error {i}"),
                ErrorSeverity.MEDIUM
            )
        
        # Component should be degraded but not disabled
        self.assertIn(self.test_component, self.error_manager.degraded_components)
        self.assertNotIn(self.test_component, self.error_manager.disabled_components)
    
    def test_emergency_mode_activation(self):
        """Test emergency mode activation."""
        # Disable multiple components to trigger emergency mode
        components = ["comp1", "comp2", "comp3", "comp4"]
        for comp in components:
            for i in range(12):  # Enough to disable each component
                self.error_manager.handle_error(
                    comp,
                    "critical_error",
                    ValueError(f"Error in {comp}"),
                    ErrorSeverity.HIGH
                )
        
        # Emergency mode should be activated
        self.assertTrue(self.error_manager.emergency_mode)
    
    def test_error_reset(self):
        """Test error counter reset functionality."""
        # Generate some errors
        for i in range(3):
            self.error_manager.handle_error(
                self.test_component,
                "test_error",
                ValueError(f"Error {i}"),
                ErrorSeverity.MEDIUM
            )
        
        # Component should be degraded
        self.assertIn(self.test_component, self.error_manager.degraded_components)
        
        # Reset errors
        self.error_manager.reset_component_errors(self.test_component)
        
        # Component should no longer be degraded
        self.assertNotIn(self.test_component, self.error_manager.degraded_components)
        self.assertEqual(self.error_manager.consecutive_errors[self.test_component], 0)
    
    def test_error_summary(self):
        """Test error summary generation."""
        # Generate some errors
        self.error_manager.handle_error(
            "comp1", "error1", ValueError("Error 1"), ErrorSeverity.LOW
        )
        self.error_manager.handle_error(
            "comp2", "error2", ValueError("Error 2"), ErrorSeverity.HIGH
        )
        
        summary = self.error_manager.get_error_summary()
        
        # Check summary structure
        self.assertIn('emergency_mode', summary)
        self.assertIn('disabled_components', summary)
        self.assertIn('degraded_components', summary)
        self.assertIn('total_errors', summary)
        self.assertIn('error_counts_by_component', summary)
        
        # Check values
        self.assertEqual(summary['total_errors'], 2)
        self.assertIn('comp1', summary['error_counts_by_component'])
        self.assertIn('comp2', summary['error_counts_by_component'])


class TestSafeAudioProcessor(unittest.TestCase):
    """Test the SafeAudioProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.error_manager = ErrorRecoveryManager()
        self.safe_processor = SafeAudioProcessor(self.error_manager)
        self.test_audio = np.random.randn(1024).astype(np.float32)
    
    def test_safe_feature_extraction_success(self):
        """Test successful feature extraction."""
        # Mock feature extractor
        mock_extractor = Mock()
        mock_features = Mock()
        mock_extractor.extract_features.return_value = mock_features
        
        result = self.safe_processor.safe_extract_features(
            self.test_audio, mock_extractor
        )
        
        self.assertEqual(result, mock_features)
        mock_extractor.extract_features.assert_called_once()
    
    def test_safe_feature_extraction_failure(self):
        """Test feature extraction with error."""
        # Mock feature extractor that raises exception
        mock_extractor = Mock()
        mock_extractor.extract_features.side_effect = ValueError("Extraction failed")
        
        result = self.safe_processor.safe_extract_features(
            self.test_audio, mock_extractor
        )
        
        # Should return fallback features
        self.assertIsNotNone(result)
        self.assertEqual(result.confidence, 0.3)  # Fallback confidence
    
    def test_safe_mood_detection_success(self):
        """Test successful mood detection."""
        # Mock mood detector
        mock_detector = Mock()
        mock_result = Mock()
        mock_detector.detect_mood.return_value = mock_result
        mock_features = Mock()
        
        result = self.safe_processor.safe_detect_mood(mock_features, mock_detector)
        
        self.assertEqual(result, mock_result)
        mock_detector.detect_mood.assert_called_once_with(mock_features)
    
    def test_safe_mood_detection_failure(self):
        """Test mood detection with error."""
        # Mock mood detector that raises exception
        mock_detector = Mock()
        mock_detector.detect_mood.side_effect = RuntimeError("Detection failed")
        mock_features = Mock()
        
        result = self.safe_processor.safe_detect_mood(mock_features, mock_detector)
        
        # Should return fallback mood result
        self.assertIsNotNone(result)
        self.assertEqual(result.mood, 'neutral')
        self.assertEqual(result.confidence, 0.5)
    
    def test_safe_audio_filtering_success(self):
        """Test successful audio filtering."""
        # Mock noise filter
        mock_filter = Mock()
        mock_vad_result = Mock()
        mock_filter.filter_audio.return_value = (self.test_audio, mock_vad_result)
        
        filtered_audio, vad_result = self.safe_processor.safe_filter_audio(
            self.test_audio, mock_filter
        )
        
        np.testing.assert_array_equal(filtered_audio, self.test_audio)
        self.assertEqual(vad_result, mock_vad_result)
    
    def test_safe_audio_filtering_failure(self):
        """Test audio filtering with error."""
        # Mock noise filter that raises exception
        mock_filter = Mock()
        mock_filter.filter_audio.side_effect = Exception("Filtering failed")
        
        filtered_audio, vad_result = self.safe_processor.safe_filter_audio(
            self.test_audio, mock_filter
        )
        
        # Should return fallback filtered audio
        self.assertIsNotNone(filtered_audio)
        self.assertIsNotNone(vad_result)
        self.assertTrue(hasattr(vad_result, 'is_voice'))


class TestMicrophoneErrorHandler(unittest.TestCase):
    """Test the MicrophoneErrorHandler class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.error_manager = ErrorRecoveryManager()
        self.mic_handler = MicrophoneErrorHandler(self.error_manager)
    
    @patch('sounddevice.query_devices')
    @patch('sounddevice.InputStream')
    def test_microphone_recovery_success(self, mock_stream, mock_query):
        """Test successful microphone recovery."""
        # Mock successful device query and stream creation
        mock_query.return_value = [{'name': 'Test Mic'}]
        mock_stream.return_value.__enter__ = Mock()
        mock_stream.return_value.__exit__ = Mock()
        
        # Create error event
        error_event = ErrorEvent(
            timestamp=time.time(),
            component='microphone_input',
            error_type='disconnection',
            severity=ErrorSeverity.HIGH,
            message='Microphone disconnected',
            exception=Exception('Microphone disconnected'),
            recovery_action=None,
            fallback_used=False
        )
        
        # Attempt recovery
        result = self.mic_handler._recover_microphone(error_event)
        
        self.assertTrue(result)
        self.assertEqual(self.mic_handler.reconnection_attempts, 0)  # Reset on success
    
    @patch('sounddevice.query_devices')
    def test_microphone_recovery_failure(self, mock_query):
        """Test failed microphone recovery."""
        # Mock failed device query
        mock_query.side_effect = Exception("No devices found")
        
        # Create error event
        error_event = ErrorEvent(
            timestamp=time.time(),
            component='microphone_input',
            error_type='disconnection',
            severity=ErrorSeverity.HIGH,
            message='Microphone disconnected',
            exception=Exception('Microphone disconnected'),
            recovery_action=None,
            fallback_used=False
        )
        
        # Attempt recovery
        result = self.mic_handler._recover_microphone(error_event)
        
        self.assertFalse(result)
        self.assertEqual(self.mic_handler.reconnection_attempts, 1)
    
    def test_max_reconnection_attempts(self):
        """Test maximum reconnection attempts limit."""
        # Set attempts to maximum
        self.mic_handler.reconnection_attempts = self.mic_handler.max_reconnection_attempts
        
        # Create error event
        error_event = ErrorEvent(
            timestamp=time.time(),
            component='microphone_input',
            error_type='disconnection',
            severity=ErrorSeverity.HIGH,
            message='Microphone disconnected',
            exception=Exception('Microphone disconnected'),
            recovery_action=None,
            fallback_used=False
        )
        
        # Attempt recovery
        result = self.mic_handler._recover_microphone(error_event)
        
        # Should fail due to max attempts reached
        self.assertFalse(result)


class TestErrorHandlingIntegration(unittest.TestCase):
    """Integration tests for error handling system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.error_manager = ErrorRecoveryManager()
    
    def test_feature_extraction_error_flow(self):
        """Test complete error flow for feature extraction."""
        # Register fallback handler
        fallback_called = False
        
        def fallback_handler(error_event):
            nonlocal fallback_called
            fallback_called = True
        
        self.error_manager.register_fallback_handler(
            'feature_extraction', fallback_handler
        )
        
        # Simulate feature extraction error
        result = self.error_manager.handle_error(
            'feature_extraction',
            'extraction_error',
            RuntimeError("Feature extraction failed"),
            ErrorSeverity.MEDIUM
        )
        
        # Fallback should have been called
        self.assertFalse(result)  # Recovery failed
        self.assertTrue(fallback_called)
        self.assertIn('feature_extraction', self.error_manager.degraded_components)
    
    def test_cascading_error_handling(self):
        """Test handling of cascading errors across components."""
        components = ['feature_extraction', 'mood_detection', 'transition_smoothing']
        
        # Simulate errors in multiple components
        for component in components:
            for i in range(3):
                self.error_manager.handle_error(
                    component,
                    'cascading_error',
                    Exception(f"Error in {component}"),
                    ErrorSeverity.MEDIUM
                )
        
        # All components should be degraded
        for component in components:
            self.assertIn(component, self.error_manager.degraded_components)
        
        # Total error count should be correct
        summary = self.error_manager.get_error_summary()
        self.assertEqual(summary['total_errors'], 9)  # 3 components × 3 errors each
    
    def test_performance_degradation_scenario(self):
        """Test system behavior under performance degradation."""
        # Simulate high-frequency errors that might occur under resource constraints
        for i in range(20):
            self.error_manager.handle_error(
                'audio_processing',
                'performance_error',
                Exception(f"Performance error {i}"),
                ErrorSeverity.LOW
            )
            time.sleep(0.01)  # Small delay to simulate real-time processing
        
        # Component should be disabled due to too many errors
        self.assertIn('audio_processing', self.error_manager.disabled_components)
    
    def test_recovery_after_errors(self):
        """Test system recovery after errors are resolved."""
        component = 'test_recovery'
        
        # Generate some errors
        for i in range(5):
            self.error_manager.handle_error(
                component,
                'temporary_error',
                Exception(f"Temporary error {i}"),
                ErrorSeverity.MEDIUM
            )
        
        # Component should be degraded
        self.assertIn(component, self.error_manager.degraded_components)
        
        # Simulate successful operations
        for i in range(3):
            self.error_manager.reset_component_errors(component)
        
        # Component should be recovered
        self.assertNotIn(component, self.error_manager.degraded_components)


class TestErrorHandlingWithMocks(unittest.TestCase):
    """Test error handling with mocked components."""
    
    def setUp(self):
        """Set up test fixtures with mocks."""
        self.error_manager = ErrorRecoveryManager()
        
        # Mock enhanced components
        self.mock_feature_extractor = Mock()
        self.mock_mood_detector = Mock()
        self.mock_noise_filter = Mock()
    
    def test_enhanced_feature_extractor_error_handling(self):
        """Test error handling in enhanced feature extractor."""
        # Mock feature extractor that fails
        self.mock_feature_extractor.extract_features.side_effect = [
            Exception("First failure"),
            Exception("Second failure"),
            Mock()  # Success on third try
        ]
        
        safe_processor = SafeAudioProcessor(self.error_manager)
        test_audio = np.random.randn(1024).astype(np.float32)
        
        # First two calls should return fallback features
        result1 = safe_processor.safe_extract_features(test_audio, self.mock_feature_extractor)
        result2 = safe_processor.safe_extract_features(test_audio, self.mock_feature_extractor)
        
        # Both should return fallback features
        self.assertEqual(result1.confidence, 0.3)
        self.assertEqual(result2.confidence, 0.3)
        
        # Third call should succeed
        result3 = safe_processor.safe_extract_features(test_audio, self.mock_feature_extractor)
        self.assertNotEqual(result3.confidence, 0.3)  # Not fallback
    
    def test_mood_detector_fallback_mode(self):
        """Test mood detector fallback mode activation."""
        # Mock mood detector that consistently fails
        self.mock_mood_detector.detect_mood.side_effect = RuntimeError("Detection failed")
        
        safe_processor = SafeAudioProcessor(self.error_manager)
        mock_features = Mock()
        
        # Multiple failures should trigger fallback
        for i in range(6):
            result = safe_processor.safe_detect_mood(mock_features, self.mock_mood_detector)
            self.assertEqual(result.mood, 'neutral')  # Fallback mood
        
        # Check that errors were logged
        summary = self.error_manager.get_error_summary()
        self.assertGreater(summary['total_errors'], 0)


if __name__ == '__main__':
    # Create a temporary directory for test logs
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        
        # Run the tests
        unittest.main(verbosity=2)