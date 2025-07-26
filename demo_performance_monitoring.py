#!/usr/bin/env python3
"""
Demo script for performance monitoring integration.
Tests the performance monitoring system with the enhanced mood detection pipeline.
"""

import time
import numpy as np
import json
import sys

# Mock psutil for testing purposes
class MockProcess:
    def cpu_percent(self):
        return 45.0 + 20.0 * np.random.random()  # Mock CPU usage with variation
    
    def memory_info(self):
        class MemInfo:
            rss = int((200 + 50 * np.random.random()) * 1024 * 1024)  # 200-250MB in bytes
        return MemInfo()

class MockPsutil:
    def Process(self):
        return MockProcess()
    
    def cpu_count(self):
        return 4

# Replace psutil import in performance_monitor
sys.modules['psutil'] = MockPsutil()

from performance_monitor import PerformanceMonitor, PerformanceScaler, PerformanceConstraints

def simulate_audio_processing_pipeline():
    """Simulate a complete audio processing pipeline with performance monitoring."""
    print("=== Performance Monitoring Demo ===\n")
    
    # Create performance monitor with Pi Zero 2 W constraints
    constraints = PerformanceConstraints(
        max_cpu_percent=80.0,
        max_memory_mb=400.0,
        max_processing_time_ms=20.0,  # Must complete within audio block time
        target_processing_time_ms=15.0
    )
    
    monitor = PerformanceMonitor(constraints)
    scaler = PerformanceScaler(monitor)
    
    print("Performance monitor initialized with Pi Zero 2 W constraints:")
    print(f"  Max processing time: {constraints.max_processing_time_ms}ms")
    print(f"  Target processing time: {constraints.target_processing_time_ms}ms")
    print(f"  Max CPU usage: {constraints.max_cpu_percent}%")
    print(f"  Max memory usage: {constraints.max_memory_mb}MB\n")
    
    # Simulate multiple audio processing cycles
    print("Simulating audio processing cycles...\n")
    
    for cycle in range(20):
        monitor.start_cycle()
        
        # Simulate audio preprocessing
        with monitor.measure_stage('audio_preprocessing'):
            time.sleep(0.001 + 0.001 * np.random.random())  # 1-2ms
        
        # Simulate feature extraction (varies based on performance level)
        if scaler.should_enable_feature('spectral_analysis_enabled'):
            with monitor.measure_stage('spectral_analysis'):
                time.sleep(0.003 + 0.002 * np.random.random())  # 3-5ms
        
        # Simulate MFCC calculation (count varies based on performance)
        mfcc_count = scaler.get_parameter_value('mfcc_coefficients', 0)
        if mfcc_count > 0:
            with monitor.measure_stage('mfcc_calculation'):
                time.sleep(0.001 * mfcc_count + 0.001 * np.random.random())
        
        # Simulate pitch analysis (may be disabled for performance)
        if scaler.should_enable_feature('pitch_analysis_enabled'):
            with monitor.measure_stage('pitch_analysis'):
                time.sleep(0.002 + 0.001 * np.random.random())  # 2-3ms
        
        # Simulate mood detection
        with monitor.measure_stage('mood_detection'):
            time.sleep(0.001 + 0.001 * np.random.random())  # 1-2ms
        
        # Simulate transition smoothing
        with monitor.measure_stage('transition_smoothing'):
            time.sleep(0.0005 + 0.0005 * np.random.random())  # 0.5-1ms
        
        # Add some cycles with intentional overruns to test scaling
        if 8 <= cycle <= 11:
            with monitor.measure_stage('simulated_overload'):
                time.sleep(0.015)  # 15ms extra - causes overrun
        
        metrics = monitor.end_cycle()
        
        # Print progress every 5 cycles
        if (cycle + 1) % 5 == 0:
            print(f"Cycle {cycle + 1:2d}: {metrics.total_duration*1000:5.1f}ms "
                  f"(level: {metrics.performance_level:6s}, "
                  f"scale: {monitor.performance_scale_factor:.2f})")
    
    print("\n=== Performance Summary ===")
    summary = monitor.get_performance_summary()
    
    print(f"Total cycles: {summary['total_cycles']}")
    print(f"Overrun rate: {summary['overrun_rate']:.1%}")
    print(f"Current performance level: {summary['current_performance_level']}")
    print(f"Performance scale factor: {summary['performance_scale_factor']:.2f}")
    print(f"Average processing time: {summary['averages']['duration_ms']:.1f}ms")
    print(f"Average CPU usage: {summary['averages']['cpu_percent']:.1f}%")
    print(f"Average memory usage: {summary['averages']['memory_mb']:.1f}MB")
    
    print("\n=== Stage Timing Analysis ===")
    for stage_name, timing_info in summary['stage_timings'].items():
        print(f"{stage_name:20s}: avg={timing_info['avg_ms']:5.1f}ms, "
              f"max={timing_info['max_ms']:5.1f}ms, "
              f"min={timing_info['min_ms']:5.1f}ms")
    
    print("\n=== Performance Recommendations ===")
    if summary['recommendations']:
        for i, rec in enumerate(summary['recommendations'], 1):
            print(f"{i}. {rec}")
    else:
        print("No performance issues detected - system running optimally!")
    
    # Save performance log
    log_filename = f"performance_log_{int(time.time())}.json"
    monitor.save_performance_log(log_filename)
    print(f"\nPerformance log saved to: {log_filename}")
    
    return monitor, scaler

def test_performance_scaling():
    """Test performance scaling functionality."""
    print("\n=== Performance Scaling Test ===\n")
    
    monitor = PerformanceMonitor()
    scaler = PerformanceScaler(monitor)
    
    # Test different performance levels
    levels = ['high', 'medium', 'low']
    scale_factors = [1.0, 0.75, 0.5]
    
    for level, scale in zip(levels, scale_factors):
        print(f"Testing {level} performance level (scale: {scale}):")
        scaler._on_performance_change(level, scale)
        
        config = scaler.get_current_config()
        print(f"  MFCC coefficients: {config['mfcc_coefficients']}")
        print(f"  Spectral analysis: {config['spectral_analysis_enabled']}")
        print(f"  Pitch analysis: {config['pitch_analysis_enabled']}")
        print(f"  Noise filtering: {config['noise_filtering_enabled']}")
        print(f"  History length: {config['history_length']}")
        print()

def test_pi_zero_constraints():
    """Test Pi Zero 2 W specific performance constraints."""
    print("=== Pi Zero 2 W Constraint Validation ===\n")
    
    # Audio block time calculation
    sample_rate = 44100
    block_size = 1024
    audio_block_time_ms = (block_size / sample_rate) * 1000
    
    print(f"Audio block time calculation:")
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Block size: {block_size} samples")
    print(f"  Block time: {audio_block_time_ms:.2f}ms")
    
    # Verify constraints are reasonable
    constraints = PerformanceConstraints()
    print(f"\nPerformance constraints:")
    print(f"  Max processing time: {constraints.max_processing_time_ms}ms "
          f"({'✓' if constraints.max_processing_time_ms <= audio_block_time_ms else '✗'})")
    print(f"  Target processing time: {constraints.target_processing_time_ms}ms")
    print(f"  Max memory (Pi Zero 2W has 512MB): {constraints.max_memory_mb}MB "
          f"({'✓' if constraints.max_memory_mb <= 400 else '✗'})")
    print(f"  Max CPU usage: {constraints.max_cpu_percent}%")
    
    # Test realistic processing times
    print(f"\nRealistic processing time breakdown:")
    stages = {
        'Audio preprocessing': 2.0,
        'Feature extraction': 6.0,
        'MFCC calculation': 3.0,
        'Pitch analysis': 2.5,
        'Mood detection': 1.5,
        'Transition smoothing': 1.0
    }
    
    total_time = 0
    for stage, time_ms in stages.items():
        print(f"  {stage:20s}: {time_ms:4.1f}ms")
        total_time += time_ms
    
    print(f"  {'Total estimated':20s}: {total_time:4.1f}ms "
          f"({'✓' if total_time <= constraints.max_processing_time_ms else '✗'})")

def main():
    """Run all performance monitoring demos."""
    try:
        # Test basic performance monitoring
        monitor, scaler = simulate_audio_processing_pipeline()
        
        # Test performance scaling
        test_performance_scaling()
        
        # Test Pi Zero 2 W constraints
        test_pi_zero_constraints()
        
        print("\n🎉 All performance monitoring tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during performance monitoring demo: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())