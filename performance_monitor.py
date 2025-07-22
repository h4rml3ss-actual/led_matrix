#!/usr/bin/env python3
"""
Real-time performance monitoring system for the enhanced mood detection pipeline.
Tracks CPU usage, timing measurements, memory usage, and provides performance scaling.
"""

import time
import psutil
import threading
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from collections import deque
import json
import os
from contextlib import contextmanager


@dataclass
class TimingMeasurement:
    """Individual timing measurement for a processing stage."""
    stage_name: str
    start_time: float
    end_time: float
    duration: float
    cpu_percent: float
    memory_mb: float


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for a processing cycle."""
    timestamp: float
    total_duration: float
    stage_timings: Dict[str, float]
    cpu_percent: float
    memory_mb: float
    memory_peak_mb: float
    audio_buffer_underruns: int
    processing_load: float  # 0.0 to 1.0
    performance_level: str  # 'high', 'medium', 'low'


@dataclass
class PerformanceConstraints:
    """Performance constraints for Pi Zero 2 W."""
    max_cpu_percent: float = 80.0
    max_memory_mb: float = 400.0  # Leave room for system
    max_processing_time_ms: float = 20.0  # Must complete within audio block time
    target_processing_time_ms: float = 15.0  # Target for good performance
    memory_warning_threshold_mb: float = 350.0
    cpu_warning_threshold: float = 70.0


class PerformanceMonitor:
    """
    Real-time performance monitoring system for audio processing pipeline.
    
    Tracks CPU usage, memory consumption, timing measurements, and provides
    performance scaling recommendations based on available system resources.
    """
    
    def __init__(self, constraints: Optional[PerformanceConstraints] = None):
        """
        Initialize the performance monitor.
        
        Args:
            constraints: Performance constraints, defaults to Pi Zero 2 W settings
        """
        self.constraints = constraints or PerformanceConstraints()
        
        # Performance history
        self.metrics_history: deque = deque(maxlen=100)  # Last 100 measurements
        self.timing_history: Dict[str, deque] = {}
        
        # Current measurement state
        self.current_measurements: Dict[str, TimingMeasurement] = {}
        self.cycle_start_time: Optional[float] = None
        self.cycle_start_memory: Optional[float] = None
        self.peak_memory_this_cycle: float = 0.0
        
        # Performance scaling state
        self.current_performance_level = 'high'
        self.performance_scale_factor = 1.0
        self.consecutive_overruns = 0
        self.consecutive_good_performance = 0
        
        # System monitoring
        self.process = psutil.Process()
        self.system_cpu_count = psutil.cpu_count()
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Performance callbacks
        self.performance_callbacks: List[Callable[[str, float], None]] = []
        
        # Statistics
        self.total_cycles = 0
        self.overrun_cycles = 0
        self.underrun_events = 0
        
    def register_performance_callback(self, callback: Callable[[str, float], None]) -> None:
        """
        Register a callback to be called when performance level changes.
        
        Args:
            callback: Function that takes (performance_level, scale_factor)
        """
        self.performance_callbacks.append(callback)
    
    @contextmanager
    def measure_stage(self, stage_name: str):
        """
        Context manager for measuring the performance of a processing stage.
        
        Args:
            stage_name: Name of the processing stage
            
        Usage:
            with monitor.measure_stage('feature_extraction'):
                # Your processing code here
                features = extract_features(audio)
        """
        start_time = time.perf_counter()
        start_cpu = self.process.cpu_percent()
        start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        try:
            yield
        finally:
            end_time = time.perf_counter()
            end_cpu = self.process.cpu_percent()
            end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            
            duration = end_time - start_time
            avg_cpu = (start_cpu + end_cpu) / 2
            
            measurement = TimingMeasurement(
                stage_name=stage_name,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                cpu_percent=avg_cpu,
                memory_mb=end_memory
            )
            
            with self._lock:
                self.current_measurements[stage_name] = measurement
                
                # Update timing history
                if stage_name not in self.timing_history:
                    self.timing_history[stage_name] = deque(maxlen=50)
                self.timing_history[stage_name].append(duration)
                
                # Track peak memory for this cycle
                self.peak_memory_this_cycle = max(self.peak_memory_this_cycle, end_memory)
    
    def start_cycle(self) -> None:
        """Start measuring a complete processing cycle."""
        with self._lock:
            self.cycle_start_time = time.perf_counter()
            self.cycle_start_memory = self.process.memory_info().rss / 1024 / 1024
            self.peak_memory_this_cycle = self.cycle_start_memory
            self.current_measurements.clear()
    
    def end_cycle(self) -> PerformanceMetrics:
        """
        End the current processing cycle and return performance metrics.
        
        Returns:
            PerformanceMetrics object with comprehensive performance data
        """
        if self.cycle_start_time is None:
            raise RuntimeError("end_cycle() called without start_cycle()")
        
        end_time = time.perf_counter()
        total_duration = end_time - self.cycle_start_time
        
        with self._lock:
            # Collect stage timings
            stage_timings = {
                name: measurement.duration 
                for name, measurement in self.current_measurements.items()
            }
            
            # Get current system metrics
            cpu_percent = self.process.cpu_percent()
            current_memory = self.process.memory_info().rss / 1024 / 1024
            
            # Calculate processing load (0.0 to 1.0)
            processing_load = min(total_duration / (self.constraints.max_processing_time_ms / 1000), 1.0)
            
            # Determine performance level
            performance_level = self._determine_performance_level(
                total_duration * 1000,  # Convert to ms
                cpu_percent,
                current_memory,
                processing_load
            )
            
            # Create metrics object
            metrics = PerformanceMetrics(
                timestamp=time.time(),
                total_duration=total_duration,
                stage_timings=stage_timings,
                cpu_percent=cpu_percent,
                memory_mb=current_memory,
                memory_peak_mb=self.peak_memory_this_cycle,
                audio_buffer_underruns=self.underrun_events,
                processing_load=processing_load,
                performance_level=performance_level
            )
            
            # Update history
            self.metrics_history.append(metrics)
            
            # Update statistics
            self.total_cycles += 1
            if total_duration * 1000 > self.constraints.max_processing_time_ms:
                self.overrun_cycles += 1
                self.consecutive_overruns += 1
                self.consecutive_good_performance = 0
            else:
                self.consecutive_overruns = 0
                self.consecutive_good_performance += 1
            
            # Update performance scaling
            self._update_performance_scaling(metrics)
            
            # Reset cycle state
            self.cycle_start_time = None
            self.cycle_start_memory = None
            
            return metrics
    
    def _determine_performance_level(self, duration_ms: float, cpu_percent: float, 
                                   memory_mb: float, processing_load: float) -> str:
        """Determine the current performance level based on metrics."""
        # Check for critical constraints
        if (duration_ms > self.constraints.max_processing_time_ms or
            cpu_percent > self.constraints.max_cpu_percent or
            memory_mb > self.constraints.max_memory_mb):
            return 'low'
        
        # Check for warning thresholds
        if (duration_ms > self.constraints.target_processing_time_ms or
            cpu_percent > self.constraints.cpu_warning_threshold or
            memory_mb > self.constraints.memory_warning_threshold_mb):
            return 'medium'
        
        return 'high'
    
    def _update_performance_scaling(self, metrics: PerformanceMetrics) -> None:
        """Update performance scaling based on recent metrics."""
        old_level = self.current_performance_level
        new_level = metrics.performance_level
        
        # Only change performance level after consistent measurements
        if new_level != old_level:
            if new_level == 'low' and self.consecutive_overruns >= 3:
                self._set_performance_level('low', 0.5)
            elif new_level == 'medium' and self.consecutive_overruns >= 2:
                self._set_performance_level('medium', 0.75)
            elif new_level == 'high' and self.consecutive_good_performance >= 5:
                self._set_performance_level('high', 1.0)
    
    def _set_performance_level(self, level: str, scale_factor: float) -> None:
        """Set the performance level and notify callbacks."""
        if level != self.current_performance_level:
            print(f"Performance level changed: {self.current_performance_level} -> {level} (scale: {scale_factor:.2f})")
            
            self.current_performance_level = level
            self.performance_scale_factor = scale_factor
            
            # Notify callbacks
            for callback in self.performance_callbacks:
                try:
                    callback(level, scale_factor)
                except Exception as e:
                    print(f"Error in performance callback: {e}")
    
    def report_audio_underrun(self) -> None:
        """Report an audio buffer underrun event."""
        with self._lock:
            self.underrun_events += 1
    
    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """Get the most recent performance metrics."""
        with self._lock:
            return self.metrics_history[-1] if self.metrics_history else None
    
    def get_average_stage_timing(self, stage_name: str, window_size: int = 10) -> float:
        """
        Get average timing for a specific stage over recent measurements.
        
        Args:
            stage_name: Name of the processing stage
            window_size: Number of recent measurements to average
            
        Returns:
            Average duration in seconds, or 0.0 if no data
        """
        with self._lock:
            if stage_name not in self.timing_history:
                return 0.0
            
            recent_timings = list(self.timing_history[stage_name])[-window_size:]
            return np.mean(recent_timings) if recent_timings else 0.0
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive performance summary.
        
        Returns:
            Dictionary with performance statistics and recommendations
        """
        with self._lock:
            if not self.metrics_history:
                return {'status': 'no_data'}
            
            recent_metrics = list(self.metrics_history)[-10:]  # Last 10 cycles
            
            # Calculate averages
            avg_duration = np.mean([m.total_duration for m in recent_metrics])
            avg_cpu = np.mean([m.cpu_percent for m in recent_metrics])
            avg_memory = np.mean([m.memory_mb for m in recent_metrics])
            avg_load = np.mean([m.processing_load for m in recent_metrics])
            
            # Calculate stage timing averages
            stage_averages = {}
            for stage_name, timings in self.timing_history.items():
                if timings:
                    stage_averages[stage_name] = {
                        'avg_ms': np.mean(list(timings)) * 1000,
                        'max_ms': np.max(list(timings)) * 1000,
                        'min_ms': np.min(list(timings)) * 1000
                    }
            
            # Performance statistics
            overrun_rate = (self.overrun_cycles / self.total_cycles) if self.total_cycles > 0 else 0.0
            
            return {
                'status': 'active',
                'current_performance_level': self.current_performance_level,
                'performance_scale_factor': self.performance_scale_factor,
                'total_cycles': self.total_cycles,
                'overrun_rate': overrun_rate,
                'underrun_events': self.underrun_events,
                'averages': {
                    'duration_ms': avg_duration * 1000,
                    'cpu_percent': avg_cpu,
                    'memory_mb': avg_memory,
                    'processing_load': avg_load
                },
                'stage_timings': stage_averages,
                'constraints': {
                    'max_processing_time_ms': self.constraints.max_processing_time_ms,
                    'target_processing_time_ms': self.constraints.target_processing_time_ms,
                    'max_cpu_percent': self.constraints.max_cpu_percent,
                    'max_memory_mb': self.constraints.max_memory_mb
                },
                'recommendations': self._generate_recommendations()
            }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        if not self.metrics_history:
            return recommendations
        
        recent_metrics = list(self.metrics_history)[-5:]
        avg_duration_ms = np.mean([m.total_duration * 1000 for m in recent_metrics])
        avg_cpu = np.mean([m.cpu_percent for m in recent_metrics])
        avg_memory = np.mean([m.memory_mb for m in recent_metrics])
        
        # Duration recommendations
        if avg_duration_ms > self.constraints.max_processing_time_ms:
            recommendations.append("Processing time exceeds audio block duration - reduce feature complexity")
        elif avg_duration_ms > self.constraints.target_processing_time_ms:
            recommendations.append("Processing time approaching limit - consider optimization")
        
        # CPU recommendations
        if avg_cpu > self.constraints.max_cpu_percent:
            recommendations.append("High CPU usage detected - reduce processing complexity")
        elif avg_cpu > self.constraints.cpu_warning_threshold:
            recommendations.append("CPU usage elevated - monitor for thermal throttling")
        
        # Memory recommendations
        if avg_memory > self.constraints.max_memory_mb:
            recommendations.append("Memory usage critical - reduce buffer sizes")
        elif avg_memory > self.constraints.memory_warning_threshold_mb:
            recommendations.append("Memory usage elevated - check for memory leaks")
        
        # Overrun recommendations
        overrun_rate = (self.overrun_cycles / self.total_cycles) if self.total_cycles > 0 else 0.0
        if overrun_rate > 0.1:
            recommendations.append("High overrun rate - enable performance scaling")
        
        # Underrun recommendations
        if self.underrun_events > 0:
            recommendations.append("Audio underruns detected - check audio system stability")
        
        return recommendations
    
    def save_performance_log(self, filepath: str) -> None:
        """
        Save performance metrics to a JSON file.
        
        Args:
            filepath: Path to save the performance log
        """
        with self._lock:
            log_data = {
                'timestamp': time.time(),
                'summary': self.get_performance_summary(),
                'recent_metrics': [
                    {
                        'timestamp': m.timestamp,
                        'total_duration_ms': m.total_duration * 1000,
                        'stage_timings_ms': {k: v * 1000 for k, v in m.stage_timings.items()},
                        'cpu_percent': m.cpu_percent,
                        'memory_mb': m.memory_mb,
                        'memory_peak_mb': m.memory_peak_mb,
                        'processing_load': m.processing_load,
                        'performance_level': m.performance_level
                    }
                    for m in list(self.metrics_history)[-20:]  # Last 20 measurements
                ]
            }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(log_data, f, indent=2)
            print(f"Performance log saved to {filepath}")
        except Exception as e:
            print(f"Error saving performance log: {e}")
    
    def reset_statistics(self) -> None:
        """Reset all performance statistics."""
        with self._lock:
            self.metrics_history.clear()
            self.timing_history.clear()
            self.total_cycles = 0
            self.overrun_cycles = 0
            self.underrun_events = 0
            self.consecutive_overruns = 0
            self.consecutive_good_performance = 0
            self.current_performance_level = 'high'
            self.performance_scale_factor = 1.0


class PerformanceScaler:
    """
    Performance scaling system that adjusts processing complexity based on available resources.
    """
    
    def __init__(self, monitor: PerformanceMonitor):
        self.monitor = monitor
        self.scaling_config = {
            'high': {
                'mfcc_coefficients': 4,
                'spectral_analysis_enabled': True,
                'pitch_analysis_enabled': True,
                'noise_filtering_enabled': True,
                'history_length': 10,
                'feature_smoothing': True
            },
            'medium': {
                'mfcc_coefficients': 2,
                'spectral_analysis_enabled': True,
                'pitch_analysis_enabled': False,
                'noise_filtering_enabled': True,
                'history_length': 5,
                'feature_smoothing': False
            },
            'low': {
                'mfcc_coefficients': 0,
                'spectral_analysis_enabled': False,
                'pitch_analysis_enabled': False,
                'noise_filtering_enabled': False,
                'history_length': 3,
                'feature_smoothing': False
            }
        }
        
        # Register with monitor
        self.monitor.register_performance_callback(self._on_performance_change)
        
        # Current configuration
        self.current_config = self.scaling_config['high'].copy()
        
        # Configuration change callbacks
        self.config_callbacks: List[Callable[[Dict[str, Any]], None]] = []
    
    def register_config_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Register a callback to be called when configuration changes.
        
        Args:
            callback: Function that takes the new configuration dictionary
        """
        self.config_callbacks.append(callback)
    
    def _on_performance_change(self, performance_level: str, scale_factor: float) -> None:
        """Handle performance level changes from the monitor."""
        new_config = self.scaling_config.get(performance_level, self.scaling_config['high']).copy()
        
        # Apply scale factor to numeric parameters
        new_config['history_length'] = max(1, int(new_config['history_length'] * scale_factor))
        
        if new_config != self.current_config:
            print(f"Updating performance configuration for level: {performance_level}")
            self.current_config = new_config
            
            # Notify callbacks
            for callback in self.config_callbacks:
                try:
                    callback(self.current_config)
                except Exception as e:
                    print(f"Error in config callback: {e}")
    
    def get_current_config(self) -> Dict[str, Any]:
        """Get the current performance configuration."""
        return self.current_config.copy()
    
    def should_enable_feature(self, feature_name: str) -> bool:
        """
        Check if a specific feature should be enabled based on current performance level.
        
        Args:
            feature_name: Name of the feature to check
            
        Returns:
            True if the feature should be enabled
        """
        return self.current_config.get(feature_name, True)
    
    def get_parameter_value(self, parameter_name: str, default_value: Any = None) -> Any:
        """
        Get a parameter value from the current configuration.
        
        Args:
            parameter_name: Name of the parameter
            default_value: Default value if parameter not found
            
        Returns:
            Parameter value or default
        """
        return self.current_config.get(parameter_name, default_value)


# Convenience function for creating a global performance monitor
_global_monitor: Optional[PerformanceMonitor] = None
_global_scaler: Optional[PerformanceScaler] = None

def get_global_monitor() -> PerformanceMonitor:
    """Get or create the global performance monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
    return _global_monitor

def get_global_scaler() -> PerformanceScaler:
    """Get or create the global performance scaler instance."""
    global _global_scaler, _global_monitor
    if _global_scaler is None:
        if _global_monitor is None:
            _global_monitor = PerformanceMonitor()
        _global_scaler = PerformanceScaler(_global_monitor)
    return _global_scaler

def reset_global_instances() -> None:
    """Reset global monitor and scaler instances."""
    global _global_monitor, _global_scaler
    _global_monitor = None
    _global_scaler = None