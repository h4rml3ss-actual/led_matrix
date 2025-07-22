#!/usr/bin/env python3
"""
Performance tests for the enhanced mood detection system.
Tests are designed for Pi Zero 2 W constraints and real-world scenarios.
"""

import unittest
import time
import numpy as np
import threading
import tempfile
import os
import json
from unittest.mock import Mock, patch, MagicMock

from performance_monitor import (
    PerformanceMonitor, PerformanceScaler, PerformanceConstraints,
    TimingMeasurement, PerformanceMetrics, get_global_monitor, get_global_scaler
)


class TestPerformanceMonitor(unittest.TestCase):
    """Test cases for the PerformanceMonitor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.constraints = PerformanceConstraints(
            max_cpu_percent=80.0,
            max_memory_mb=400.0,
            max_processing_time_ms=20.0,
            target_processing_time_ms=15.0
        )
        self.monitor = PerformanceMonitor(self.constraints)
    
    def test_initialization(self):
        """Test monitor initialization."""
        self.assertEqual(self.monitor.current_performance_level, 'high')
        self.assertEqual(self.monitor.performance_scale_factor, 1.0)
        self.assertEqual(self.monitor.total_cycles, 0)
        self.assertEqual(self.monitor.overrun_cycles, 0)
    
    def test_stage_measurement(self):
        """Test individual stage timing measurement."""
        # Simulate a processing stage
        with self.monitor.measure_stage('test_stage'):
            time.sleep(0.01)  # 10ms processing
        
        # Check that measurement was recorded
        self.assertIn('test_stage', self.monitor.current_measurements)
        measurement = self.monitor.current_measurements['test_stage']
        self.assertEqual(measurement.stage_name, 'test_stage')
        self.assertGreater(measurement.duration, 0.009)  # At least 9ms
        self.assertLess(measurement.duration, 0.02)      # Less than 20ms
    
    def test_cycle_measurement(self):
        """Test complete cycle measurement."""
        self.monitor.start_cycle()
        
        # Simulate processing stages
        with self.monitor.measure_stage('feature_extraction'):
            time.sleep(0.005)  # 5ms
        
        with self.monitor.measure_stage('mood_detection'):
            time.sleep(0.003)  # 3ms
        
        metrics = self.monitor.end_cycle()
        
        # Verify metrics
        self.assertIsInstance(metrics, PerformanceMetrics)
        self.assertGreater(metrics.total_duration, 0.007)  # At least 7ms
        self.assertIn('feature_extraction', metrics.stage_timings)
        self.assertIn('mood_detection', metrics.stage_timings)
        self.assertEqual(self.monitor.total_cycles, 1)
    
    def test_performance_level_detection(self):
        """Test performance level determination."""
        # Test high performance (fast processing)
        self.monitor.start_cycle()
        with self.monitor.measure_stage('fast_stage'):
            time.sleep(0.005)  # 5ms - well within limits
        metrics = self.monitor.end_cycle()
        self.assertEqual(metrics.performance_level, 'high')
        
        # Test medium performance (approaching limits)
        self.monitor.start_cycle()
        with self.monitor.measure_stage('medium_stage'):
            time.sleep(0.018)  # 18ms - approaching 20ms limit
        metrics = self.monitor.end_cycle()
        # Note: actual performance level depends on CPU/memory as well
        self.assertIn(metrics.performance_level, ['high', 'medium'])
    
    def test_overrun_detection(self):
        """Test detection of processing time overruns."""
        initial_overruns = self.monitor.overrun_cycles
        
        # Simulate an overrun (processing takes longer than audio block time)
        self.monitor.start_cycle()
        with self.monitor.measure_stage('slow_stage'):
            time.sleep(0.025)  # 25ms - exceeds 20ms limit
        metrics = self.monitor.end_cycle()
        
        # Check that overrun was detected
        self.assertEqual(self.monitor.overrun_cycles, initial_overruns + 1)
        self.assertGreater(metrics.total_duration * 1000, self.constraints.max_processing_time_ms)
    
    def test_performance_scaling(self):
        """Test automatic performance scaling."""
        callback_calls = []
        
        def test_callback(level, scale_factor):
            callback_calls.append((level, scale_factor))
        
        self.monitor.register_performance_callback(test_callback)
        
        # Simulate multiple overruns to trigger scaling
        for _ in range(4):  # Need 3+ consecutive overruns
            self.monitor.start_cycle()
            with self.monitor.measure_stage('overrun_stage'):
                time.sleep(0.025)  # 25ms overrun
            self.monitor.end_cycle()
        
        # Check that performance level was reduced
        self.assertEqual(self.monitor.current_performance_level, 'low')
        self.assertEqual(self.monitor.performance_scale_factor, 0.5)
        self.assertTrue(len(callback_calls) > 0)
    
    def test_timing_history(self):
        """Test timing history tracking."""
        stage_name = 'test_history_stage'
        
        # Add multiple measurements
        for i in range(5):
            with self.monitor.measure_stage(stage_name):
                time.sleep(0.001 * (i + 1))  # Variable timing
        
        # Check history was recorded
        self.assertIn(stage_name, self.monitor.timing_history)
        history = self.monitor.timing_history[stage_name]
        self.assertEqual(len(history), 5)
        
        # Check average calculation
        avg_timing = self.monitor.get_average_stage_timing(stage_name)
        self.assertGreater(avg_timing, 0.0)
    
    def test_performance_summary(self):
        """Test performance summary generation."""
        # Generate some test data
        for _ in range(3):
            self.monitor.start_cycle()
            with self.monitor.measure_stage('summary_test'):
                time.sleep(0.01)
            self.monitor.end_cycle()
        
        summary = self.monitor.get_performance_summary()
        
        # Verify summary structure
        self.assertEqual(summary['status'], 'active')
        self.assertIn('current_performance_level', summary)
        self.assertIn('averages', summary)
        self.assertIn('stage_timings', summary)
        self.assertIn('recommendations', summary)
        self.assertEqual(summary['total_cycles'], 3)
    
    def test_audio_underrun_reporting(self):
        """Test audio underrun event reporting."""
        initial_underruns = self.monitor.underrun_events
        
        self.monitor.report_audio_underrun()
        self.monitor.report_audio_underrun()
        
        self.assertEqual(self.monitor.underrun_events, initial_underruns + 2)
    
    def test_performance_log_saving(self):
        """Test saving performance logs to file."""
        # Generate test data
        self.monitor.start_cycle()
        with self.monitor.measure_stage('log_test'):
            time.sleep(0.005)
        self.monitor.end_cycle()
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            self.monitor.save_performance_log(temp_path)
            
            # Verify file was created and contains valid JSON
            self.assertTrue(os.path.exists(temp_path))
            with open(temp_path, 'r') as f:
                log_data = json.load(f)
            
            self.assertIn('summary', log_data)
            self.assertIn('recent_metrics', log_data)
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_thread_safety(self):
        """Test thread safety of performance monitoring."""
        results = []
        
        def worker_thread(thread_id):
            for i in range(10):
                self.monitor.start_cycle()
                with self.monitor.measure_stage(f'thread_{thread_id}_stage_{i}'):
                    time.sleep(0.001)
                metrics = self.monitor.end_cycle()
                results.append((thread_id, i, metrics.total_duration))
        
        # Start multiple threads
        threads = []
        for tid in range(3):
            thread = threading.Thread(target=worker_thread, args=(tid,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify all measurements were recorded
        self.assertEqual(len(results), 30)  # 3 threads × 10 measurements
        self.assertEqual(self.monitor.total_cycles, 30)


class TestPerformanceScaler(unittest.TestCase):
    """Test cases for the PerformanceScaler class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.monitor = PerformanceMonitor()
        self.scaler = PerformanceScaler(self.monitor)
    
    def test_initialization(self):
        """Test scaler initialization."""
        config = self.scaler.get_current_config()
        self.assertEqual(config['mfcc_coefficients'], 4)  # High performance default
        self.assertTrue(config['spectral_analysis_enabled'])
        self.assertTrue(config['pitch_analysis_enabled'])
    
    def test_performance_level_scaling(self):
        """Test configuration changes based on performance level."""
        # Simulate performance level change to 'medium'
        self.scaler._on_performance_change('medium', 0.75)
        
        config = self.scaler.get_current_config()
        self.assertEqual(config['mfcc_coefficients'], 2)
        self.assertTrue(config['spectral_analysis_enabled'])
        self.assertFalse(config['pitch_analysis_enabled'])
        
        # Simulate performance level change to 'low'
        self.scaler._on_performance_change('low', 0.5)
        
        config = self.scaler.get_current_config()
        self.assertEqual(config['mfcc_coefficients'], 0)
        self.assertFalse(config['spectral_analysis_enabled'])
        self.assertFalse(config['pitch_analysis_enabled'])
    
    def test_feature_enabling(self):
        """Test feature enabling based on performance level."""
        # High performance - all features enabled
        self.assertTrue(self.scaler.should_enable_feature('spectral_analysis_enabled'))
        self.assertTrue(self.scaler.should_enable_feature('pitch_analysis_enabled'))
        
        # Switch to low performance
        self.scaler._on_performance_change('low', 0.5)
        self.assertFalse(self.scaler.should_enable_feature('spectral_analysis_enabled'))
        self.assertFalse(self.scaler.should_enable_feature('pitch_analysis_enabled'))
    
    def test_parameter_values(self):
        """Test parameter value retrieval."""
        # Test with existing parameter
        mfcc_count = self.scaler.get_parameter_value('mfcc_coefficients')
        self.assertEqual(mfcc_count, 4)
        
        # Test with non-existing parameter and default
        unknown_param = self.scaler.get_parameter_value('unknown_param', 42)
        self.assertEqual(unknown_param, 42)
    
    def test_config_callbacks(self):
        """Test configuration change callbacks."""
        callback_calls = []
        
        def test_callback(config):
            callback_calls.append(config.copy())
        
        self.scaler.register_config_callback(test_callback)
        
        # Trigger configuration change
        self.scaler._on_performance_change('medium', 0.75)
        
        # Verify callback was called
        self.assertEqual(len(callback_calls), 1)
        self.assertEqual(callback_calls[0]['mfcc_coefficients'], 2)


class TestPiZero2WConstraints(unittest.TestCase):
    """Test cases specifically for Pi Zero 2 W performance constraints."""
    
    def setUp(self):
        """Set up Pi Zero 2 W specific test environment."""
        # Pi Zero 2 W has limited resources
        self.pi_constraints = PerformanceConstraints(
            max_cpu_percent=85.0,      # Allow slightly higher CPU usage
            max_memory_mb=350.0,       # Conservative memory limit
            max_processing_time_ms=23.0,  # Audio block time at 44.1kHz, 1024 samples
            target_processing_time_ms=18.0,  # Target for good performance
            memory_warning_threshold_mb=300.0,
            cpu_warning_threshold=75.0
        )
        self.monitor = PerformanceMonitor(self.pi_constraints)
    
    def test_audio_block_timing_constraints(self):
        """Test that processing completes within audio block time."""
        # Audio block time calculation: 1024 samples / 44100 Hz = ~23.2ms
        audio_block_time_ms = (1024 / 44100) * 1000
        
        self.assertAlmostEqual(audio_block_time_ms, 23.2, places=1)
        self.assertLessEqual(self.pi_constraints.max_processing_time_ms, audio_block_time_ms)
    
    def test_memory_constraints(self):
        """Test memory usage constraints for Pi Zero 2 W."""
        # Pi Zero 2 W has 512MB RAM, we should use less than 70%
        max_reasonable_memory = 512 * 0.7  # 358MB
        self.assertLessEqual(self.pi_constraints.max_memory_mb, max_reasonable_memory)
    
    def test_realistic_processing_simulation(self):
        """Simulate realistic audio processing workload."""
        # Simulate typical enhanced mood detection processing
        processing_stages = [
            ('audio_preprocessing', 0.002),    # 2ms - noise filtering, windowing
            ('feature_extraction', 0.008),     # 8ms - RMS, ZCR, spectral features
            ('mfcc_calculation', 0.004),       # 4ms - MFCC computation
            ('pitch_analysis', 0.003),         # 3ms - F0 estimation
            ('mood_detection', 0.002),         # 2ms - classification
            ('transition_smoothing', 0.001),   # 1ms - smoothing logic
        ]
        
        self.monitor.start_cycle()
        
        for stage_name, duration in processing_stages:
            with self.monitor.measure_stage(stage_name):
                # Simulate processing with some variability
                actual_duration = duration * (0.8 + 0.4 * np.random.random())
                time.sleep(actual_duration)
        
        metrics = self.monitor.end_cycle()
        
        # Total should be around 20ms, well within 23ms limit
        total_ms = metrics.total_duration * 1000
        self.assertLess(total_ms, self.pi_constraints.max_processing_time_ms)
        self.assertGreater(total_ms, 15.0)  # Should be at least 15ms for realistic processing
    
    def test_performance_degradation_simulation(self):
        """Simulate performance degradation and recovery."""
        # Start with good performance
        for _ in range(5):
            self.monitor.start_cycle()
            with self.monitor.measure_stage('normal_processing'):
                time.sleep(0.015)  # 15ms - good performance
            self.monitor.end_cycle()
        
        self.assertEqual(self.monitor.current_performance_level, 'high')
        
        # Simulate system load causing overruns
        for _ in range(4):
            self.monitor.start_cycle()
            with self.monitor.measure_stage('overloaded_processing'):
                time.sleep(0.025)  # 25ms - exceeds limit
            self.monitor.end_cycle()
        
        # Should have scaled down performance
        self.assertEqual(self.monitor.current_performance_level, 'low')
        
        # Simulate recovery
        for _ in range(6):
            self.monitor.start_cycle()
            with self.monitor.measure_stage('recovered_processing'):
                time.sleep(0.012)  # 12ms - good performance again
            self.monitor.end_cycle()
        
        # Should have scaled back up
        self.assertEqual(self.monitor.current_performance_level, 'high')
    
    @patch('psutil.Process')
    def test_resource_monitoring_accuracy(self, mock_process):
        """Test accuracy of resource monitoring."""
        # Mock process to return specific values
        mock_instance = Mock()
        mock_instance.cpu_percent.return_value = 65.0
        mock_instance.memory_info.return_value = Mock(rss=300 * 1024 * 1024)  # 300MB
        mock_process.return_value = mock_instance
        
        monitor = PerformanceMonitor(self.pi_constraints)
        
        monitor.start_cycle()
        with monitor.measure_stage('test_stage'):
            time.sleep(0.01)
        metrics = monitor.end_cycle()
        
        # Verify that mocked values are reflected in metrics
        self.assertEqual(metrics.cpu_percent, 65.0)
        self.assertAlmostEqual(metrics.memory_mb, 300.0, places=1)


class TestGlobalInstances(unittest.TestCase):
    """Test cases for global monitor and scaler instances."""
    
    def test_global_monitor_singleton(self):
        """Test that global monitor returns the same instance."""
        monitor1 = get_global_monitor()
        monitor2 = get_global_monitor()
        self.assertIs(monitor1, monitor2)
    
    def test_global_scaler_singleton(self):
        """Test that global scaler returns the same instance."""
        scaler1 = get_global_scaler()
        scaler2 = get_global_scaler()
        self.assertIs(scaler1, scaler2)
    
    def test_global_instances_integration(self):
        """Test integration between global monitor and scaler."""
        monitor = get_global_monitor()
        scaler = get_global_scaler()
        
        # Verify they're connected
        self.assertIs(scaler.monitor, monitor)


class TestPerformanceIntegration(unittest.TestCase):
    """Integration tests for the complete performance monitoring system."""
    
    def test_end_to_end_monitoring(self):
        """Test complete end-to-end performance monitoring workflow."""
        monitor = PerformanceMonitor()
        scaler = PerformanceScaler(monitor)
        
        config_changes = []
        
        def config_callback(config):
            config_changes.append(config.copy())
        
        scaler.register_config_callback(config_callback)
        
        # Simulate a complete audio processing cycle
        monitor.start_cycle()
        
        # Feature extraction with performance scaling
        if scaler.should_enable_feature('spectral_analysis_enabled'):
            with monitor.measure_stage('spectral_analysis'):
                time.sleep(0.005)
        
        if scaler.should_enable_feature('pitch_analysis_enabled'):
            with monitor.measure_stage('pitch_analysis'):
                time.sleep(0.003)
        
        mfcc_count = scaler.get_parameter_value('mfcc_coefficients', 0)
        if mfcc_count > 0:
            with monitor.measure_stage('mfcc_calculation'):
                time.sleep(0.002 * mfcc_count)
        
        with monitor.measure_stage('mood_detection'):
            time.sleep(0.002)
        
        metrics = monitor.end_cycle()
        
        # Verify the cycle completed successfully
        self.assertIsInstance(metrics, PerformanceMetrics)
        self.assertGreater(len(metrics.stage_timings), 0)
        
        # Generate performance summary
        summary = monitor.get_performance_summary()
        self.assertEqual(summary['status'], 'active')
        self.assertGreater(summary['total_cycles'], 0)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)