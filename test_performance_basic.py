#!/usr/bin/env python3
"""
Basic test script for performance monitoring without external dependencies.
Tests core functionality that can run on Pi Zero 2 W.
"""

import time
import sys
import os

# Mock psutil for testing purposes
class MockProcess:
    def cpu_percent(self):
        return 45.0  # Mock CPU usage
    
    def memory_info(self):
        class MemInfo:
            rss = 200 * 1024 * 1024  # 200MB in bytes
        return MemInfo()

class MockPsutil:
    def Process(self):
        return MockProcess()
    
    def cpu_count(self):
        return 4

# Replace psutil import in performance_monitor
sys.modules['psutil'] = MockPsutil()

# Now import our performance monitor
from performance_monitor import PerformanceMonitor, PerformanceScaler, PerformanceConstraints

def test_basic_functionality():
    """Test basic performance monitoring functionality."""
    print("Testing basic performance monitoring...")
    
    # Create monitor with Pi Zero 2 W constraints
    constraints = PerformanceConstraints(
        max_cpu_percent=80.0,
        max_memory_mb=400.0,
        max_processing_time_ms=20.0,
        target_processing_time_ms=15.0
    )
    
    monitor = PerformanceMonitor(constraints)
    print(f"✓ Monitor initialized with performance level: {monitor.current_performance_level}")
    
    # Test stage measurement
    monitor.start_cycle()
    
    with monitor.measure_stage('feature_extraction'):
        time.sleep(0.008)  # 8ms processing
    
    with monitor.measure_stage('mood_detection'):
        time.sleep(0.005)  # 5ms processing
    
    metrics = monitor.end_cycle()
    
    print(f"✓ Cycle completed in {metrics.total_duration * 1000:.1f}ms")
    print(f"✓ Performance level: {metrics.performance_level}")
    print(f"✓ Stage timings: {[(k, f'{v*1000:.1f}ms') for k, v in metrics.stage_timings.items()]}")
    
    return True

def test_performance_scaling():
    """Test performance scaling functionality."""
    print("\nTesting performance scaling...")
    
    monitor = PerformanceMonitor()
    scaler = PerformanceScaler(monitor)
    
    print(f"✓ Initial config: MFCC={scaler.get_parameter_value('mfcc_coefficients')}")
    print(f"✓ Spectral analysis enabled: {scaler.should_enable_feature('spectral_analysis_enabled')}")
    
    # Simulate performance degradation
    scaler._on_performance_change('medium', 0.75)
    print(f"✓ Medium performance: MFCC={scaler.get_parameter_value('mfcc_coefficients')}")
    
    scaler._on_performance_change('low', 0.5)
    print(f"✓ Low performance: MFCC={scaler.get_parameter_value('mfcc_coefficients')}")
    print(f"✓ Spectral analysis enabled: {scaler.should_enable_feature('spectral_analysis_enabled')}")
    
    return True

def test_overrun_detection():
    """Test processing time overrun detection."""
    print("\nTesting overrun detection...")
    
    monitor = PerformanceMonitor()
    
    # Simulate normal processing
    monitor.start_cycle()
    with monitor.measure_stage('normal_stage'):
        time.sleep(0.015)  # 15ms - within limits
    metrics1 = monitor.end_cycle()
    
    # Simulate overrun
    monitor.start_cycle()
    with monitor.measure_stage('slow_stage'):
        time.sleep(0.025)  # 25ms - exceeds 20ms limit
    metrics2 = monitor.end_cycle()
    
    print(f"✓ Normal processing: {metrics1.total_duration * 1000:.1f}ms")
    print(f"✓ Overrun processing: {metrics2.total_duration * 1000:.1f}ms")
    print(f"✓ Overrun cycles: {monitor.overrun_cycles}")
    
    return True

def test_pi_zero_constraints():
    """Test Pi Zero 2 W specific constraints."""
    print("\nTesting Pi Zero 2 W constraints...")
    
    # Audio block time calculation: 1024 samples / 44100 Hz
    audio_block_time_ms = (1024 / 44100) * 1000
    print(f"✓ Audio block time: {audio_block_time_ms:.1f}ms")
    
    constraints = PerformanceConstraints()
    print(f"✓ Max processing time: {constraints.max_processing_time_ms}ms")
    print(f"✓ Target processing time: {constraints.target_processing_time_ms}ms")
    print(f"✓ Max memory: {constraints.max_memory_mb}MB")
    print(f"✓ Max CPU: {constraints.max_cpu_percent}%")
    
    # Verify constraints are reasonable for Pi Zero 2 W
    assert constraints.max_processing_time_ms <= audio_block_time_ms
    assert constraints.max_memory_mb <= 400  # Conservative for 512MB system
    
    return True

def test_performance_summary():
    """Test performance summary generation."""
    print("\nTesting performance summary...")
    
    monitor = PerformanceMonitor()
    
    # Generate some test data
    for i in range(3):
        monitor.start_cycle()
        with monitor.measure_stage(f'test_stage_{i}'):
            time.sleep(0.01)
        monitor.end_cycle()
    
    summary = monitor.get_performance_summary()
    
    print(f"✓ Status: {summary['status']}")
    print(f"✓ Total cycles: {summary['total_cycles']}")
    print(f"✓ Performance level: {summary['current_performance_level']}")
    print(f"✓ Average duration: {summary['averages']['duration_ms']:.1f}ms")
    print(f"✓ Recommendations: {len(summary['recommendations'])} items")
    
    return True

def main():
    """Run all basic tests."""
    print("=== Performance Monitor Basic Tests ===")
    
    tests = [
        test_basic_functionality,
        test_performance_scaling,
        test_overrun_detection,
        test_pi_zero_constraints,
        test_performance_summary
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
                print("✓ PASSED")
            else:
                failed += 1
                print("✗ FAILED")
        except Exception as e:
            failed += 1
            print(f"✗ FAILED: {e}")
    
    print(f"\n=== Results ===")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {passed + failed}")
    
    if failed == 0:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1

if __name__ == '__main__':
    sys.exit(main())