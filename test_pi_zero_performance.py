#!/usr/bin/env python3
"""
Pi Zero 2 W specific performance tests for the enhanced mood detection system.
Tests realistic constraints and performance scenarios for the target hardware.
"""

import time
import numpy as np
import sys

# Mock psutil for testing purposes
class MockProcess:
    def cpu_percent(self):
        return 65.0 + 15.0 * np.random.random()  # 65-80% CPU usage
    
    def memory_info(self):
        class MemInfo:
            rss = int((280 + 40 * np.random.random()) * 1024 * 1024)  # 280-320MB
        return MemInfo()

class MockPsutil:
    def Process(self):
        return MockProcess()
    
    def cpu_count(self):
        return 4  # Pi Zero 2 W has 4 cores

# Replace psutil import
sys.modules['psutil'] = MockPsutil()

from performance_monitor import PerformanceMonitor, PerformanceScaler, PerformanceConstraints

def test_pi_zero_audio_constraints():
    """Test Pi Zero 2 W audio processing constraints."""
    print("=== Pi Zero 2 W Audio Constraints Test ===")
    
    # Pi Zero 2 W audio parameters
    sample_rate = 44100
    block_size = 1024
    audio_block_time_ms = (block_size / sample_rate) * 1000
    
    print(f"Audio block time: {audio_block_time_ms:.2f}ms")
    
    # Create Pi Zero 2 W specific constraints
    constraints = PerformanceConstraints(
        max_cpu_percent=85.0,      # Allow higher CPU on Pi Zero 2 W
        max_memory_mb=350.0,       # Conservative for 512MB system
        max_processing_time_ms=23.0,  # Must complete within audio block
        target_processing_time_ms=18.0,  # Target for good performance
        memory_warning_threshold_mb=300.0,
        cpu_warning_threshold=75.0
    )
    
    monitor = PerformanceMonitor(constraints)
    
    # Test realistic processing pipeline
    print("\nTesting realistic audio processing pipeline:")
    
    for cycle in range(10):
        monitor.start_cycle()
        
        # Simulate typical enhanced mood detection stages
        stages = [
            ('audio_preprocessing', 0.002),    # 2ms - windowing, normalization
            ('noise_filtering', 0.003),        # 3ms - background noise removal
            ('energy_features', 0.001),        # 1ms - RMS, peak energy
            ('spectral_features', 0.006),      # 6ms - centroid, rolloff, flux
            ('mfcc_calculation', 0.004),       # 4ms - MFCC coefficients
            ('pitch_analysis', 0.003),         # 3ms - F0 estimation
            ('mood_detection', 0.002),         # 2ms - classification
            ('transition_smoothing', 0.001),   # 1ms - smoothing logic
        ]
        
        total_expected = sum(duration for _, duration in stages)
        
        for stage_name, base_duration in stages:
            # Add realistic variability (±20%)
            actual_duration = base_duration * (0.8 + 0.4 * np.random.random())
            
            with monitor.measure_stage(stage_name):
                time.sleep(actual_duration)
        
        metrics = monitor.end_cycle()
        
        # Check if processing completed within audio block time
        processing_time_ms = metrics.total_duration * 1000
        within_limit = processing_time_ms <= constraints.max_processing_time_ms
        
        print(f"Cycle {cycle+1:2d}: {processing_time_ms:5.1f}ms "
              f"({'✓' if within_limit else '✗'}) "
              f"Level: {metrics.performance_level}")
    
    # Generate summary
    summary = monitor.get_performance_summary()
    print(f"\nSummary:")
    print(f"  Average processing time: {summary['averages']['duration_ms']:.1f}ms")
    print(f"  Overrun rate: {summary['overrun_rate']:.1%}")
    print(f"  Performance level: {summary['current_performance_level']}")
    
    return summary['overrun_rate'] < 0.1  # Less than 10% overruns

def test_performance_scaling_effectiveness():
    """Test that performance scaling effectively reduces processing time."""
    print("\n=== Performance Scaling Effectiveness Test ===")
    
    monitor = PerformanceMonitor()
    scaler = PerformanceScaler(monitor)
    
    # Test different performance levels
    results = {}
    
    for level, scale_factor in [('high', 1.0), ('medium', 0.75), ('low', 0.5)]:
        print(f"\nTesting {level} performance level:")
        
        # Set performance level
        scaler._on_performance_change(level, scale_factor)
        config = scaler.get_current_config()
        
        # Simulate processing with current configuration
        processing_times = []
        
        for _ in range(5):
            monitor.start_cycle()
            
            # Simulate feature extraction based on configuration
            if config['spectral_analysis_enabled']:
                with monitor.measure_stage('spectral_analysis'):
                    time.sleep(0.005)
            
            if config['pitch_analysis_enabled']:
                with monitor.measure_stage('pitch_analysis'):
                    time.sleep(0.003)
            
            # MFCC calculation scales with coefficient count
            mfcc_count = config['mfcc_coefficients']
            if mfcc_count > 0:
                with monitor.measure_stage('mfcc_calculation'):
                    time.sleep(0.001 * mfcc_count)
            
            # Basic processing always happens
            with monitor.measure_stage('basic_processing'):
                time.sleep(0.004)
            
            metrics = monitor.end_cycle()
            processing_times.append(metrics.total_duration * 1000)
        
        avg_time = np.mean(processing_times)
        results[level] = avg_time
        
        print(f"  MFCC coefficients: {config['mfcc_coefficients']}")
        print(f"  Spectral analysis: {config['spectral_analysis_enabled']}")
        print(f"  Pitch analysis: {config['pitch_analysis_enabled']}")
        print(f"  Average processing time: {avg_time:.1f}ms")
    
    # Verify that scaling reduces processing time
    print(f"\nScaling effectiveness:")
    print(f"  High → Medium: {results['high']:.1f}ms → {results['medium']:.1f}ms "
          f"({((results['medium']/results['high']-1)*100):+.1f}%)")
    print(f"  Medium → Low: {results['medium']:.1f}ms → {results['low']:.1f}ms "
          f"({((results['low']/results['medium']-1)*100):+.1f}%)")
    
    # Performance scaling should reduce processing time
    scaling_effective = (results['medium'] < results['high'] and 
                        results['low'] < results['medium'])
    
    return scaling_effective

def test_memory_usage_monitoring():
    """Test memory usage monitoring and leak detection."""
    print("\n=== Memory Usage Monitoring Test ===")
    
    constraints = PerformanceConstraints(max_memory_mb=350.0)
    monitor = PerformanceMonitor(constraints)
    
    initial_metrics = None
    final_metrics = None
    
    # Simulate extended processing to check for memory leaks
    for cycle in range(20):
        monitor.start_cycle()
        
        # Simulate processing with some memory allocation
        with monitor.measure_stage('memory_test'):
            # Simulate temporary memory usage
            temp_data = np.random.random(1000)  # Small allocation
            time.sleep(0.005)
            del temp_data
        
        metrics = monitor.end_cycle()
        
        if cycle == 0:
            initial_metrics = metrics
        if cycle == 19:
            final_metrics = metrics
    
    # Check memory usage
    memory_increase = final_metrics.memory_mb - initial_metrics.memory_mb
    print(f"Initial memory: {initial_metrics.memory_mb:.1f}MB")
    print(f"Final memory: {final_metrics.memory_mb:.1f}MB")
    print(f"Memory change: {memory_increase:+.1f}MB")
    
    # Memory should not increase significantly (< 10MB increase is acceptable)
    memory_stable = abs(memory_increase) < 10.0
    
    # Check if memory warnings would be triggered
    memory_warning = final_metrics.memory_mb > constraints.memory_warning_threshold_mb
    memory_critical = final_metrics.memory_mb > constraints.max_memory_mb
    
    print(f"Memory warning triggered: {memory_warning}")
    print(f"Memory critical triggered: {memory_critical}")
    
    return memory_stable and not memory_critical

def test_real_time_constraints():
    """Test real-time processing constraints."""
    print("\n=== Real-Time Constraints Test ===")
    
    # Pi Zero 2 W real-time constraints
    constraints = PerformanceConstraints(
        max_processing_time_ms=20.0,  # Must complete within audio block
        target_processing_time_ms=15.0
    )
    
    monitor = PerformanceMonitor(constraints)
    
    # Test with varying processing loads
    test_scenarios = [
        ("Light load", 0.008),      # 8ms processing
        ("Medium load", 0.015),     # 15ms processing  
        ("Heavy load", 0.018),      # 18ms processing
        ("Overload", 0.025),        # 25ms processing (overrun)
    ]
    
    results = {}
    
    for scenario_name, processing_time in test_scenarios:
        overruns = 0
        
        for _ in range(10):
            monitor.start_cycle()
            
            with monitor.measure_stage('test_load'):
                time.sleep(processing_time)
            
            metrics = monitor.end_cycle()
            
            if metrics.total_duration * 1000 > constraints.max_processing_time_ms:
                overruns += 1
        
        overrun_rate = overruns / 10
        results[scenario_name] = overrun_rate
        
        print(f"{scenario_name:12s}: {processing_time*1000:4.0f}ms → "
              f"{overrun_rate:.0%} overruns")
    
    # Verify that overruns only occur with actual overload
    real_time_ok = (results["Light load"] == 0 and 
                   results["Medium load"] == 0 and
                   results["Heavy load"] == 0 and
                   results["Overload"] > 0)
    
    return real_time_ok

def main():
    """Run all Pi Zero 2 W performance tests."""
    print("Pi Zero 2 W Performance Testing")
    print("=" * 50)
    
    tests = [
        ("Audio constraints", test_pi_zero_audio_constraints),
        ("Performance scaling", test_performance_scaling_effectiveness),
        ("Memory monitoring", test_memory_usage_monitoring),
        ("Real-time constraints", test_real_time_constraints),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*20} {test_name} {'='*20}")
            if test_func():
                print(f"✓ {test_name} PASSED")
                passed += 1
            else:
                print(f"✗ {test_name} FAILED")
                failed += 1
        except Exception as e:
            print(f"✗ {test_name} ERROR: {e}")
            failed += 1
    
    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All Pi Zero 2 W performance tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1

if __name__ == '__main__':
    sys.exit(main())