#!/usr/bin/env python3
"""
Demonstration of debugging and diagnostic tools for enhanced mood detection.
Shows how to use debug logging, real-time visualization, and diagnostic analysis.
"""

import time
import numpy as np
import threading
import sys
import os
from typing import Optional

# Import enhanced components
try:
    from enhanced_audio_features import EnhancedFeatureExtractor, AudioFeatures
    from advanced_mood_detector import AdvancedMoodDetector, MoodResult
    from mood_transition_smoother import MoodTransitionSmoother
    from mood_config import ConfigManager
    from mood_debug_tools import (
        MoodDebugLogger, RealTimeVisualizer, DiagnosticAnalyzer, 
        ConfigValidator, get_debug_logger
    )
    from performance_monitor import get_global_monitor
    ENHANCED_AVAILABLE = True
except ImportError as e:
    print(f"Enhanced components not available: {e}")
    ENHANCED_AVAILABLE = False


class DebugDemoSystem:
    """
    Demonstration system showing debug and diagnostic capabilities.
    """
    
    def __init__(self):
        if not ENHANCED_AVAILABLE:
            raise RuntimeError("Enhanced components required for demo")
        
        # Initialize components
        self.feature_extractor = EnhancedFeatureExtractor(
            samplerate=44100,
            frame_size=1024,
            enable_noise_filtering=True
        )
        self.mood_detector = AdvancedMoodDetector()
        self.transition_smoother = MoodTransitionSmoother()
        
        # Initialize debugging tools
        self.debug_logger = get_debug_logger()
        self.visualizer = RealTimeVisualizer(update_interval=0.1)
        self.diagnostic_analyzer = DiagnosticAnalyzer()
        self.config_validator = ConfigValidator()
        
        # Performance monitoring
        self.performance_monitor = get_global_monitor()
        
        # Demo state
        self.running = False
        self.demo_thread = None
    
    def generate_demo_audio(self, mood_type: str, duration: float = 1.0) -> np.ndarray:
        """Generate synthetic audio for different mood types."""
        samplerate = 44100
        samples = int(duration * samplerate)
        t = np.linspace(0, duration, samples)
        
        if mood_type == "calm":
            # Low amplitude, low frequency, stable
            frequency = 200.0
            amplitude = 0.015
            audio = amplitude * np.sin(2 * np.pi * frequency * t)
            # Add minimal variation
            variation = 0.002 * np.sin(2 * np.pi * 3 * t)
            audio += variation
            
        elif mood_type == "neutral":
            # Medium amplitude, normal speech frequency
            frequency = 400.0
            amplitude = 0.04
            audio = amplitude * np.sin(2 * np.pi * frequency * t)
            # Add speech-like harmonics
            harmonic = 0.01 * np.sin(2 * np.pi * frequency * 2 * t)
            audio += harmonic
            
        elif mood_type == "energetic":
            # Higher amplitude, variable frequency
            base_frequency = 600.0
            amplitude = 0.10
            # Add frequency modulation
            freq_mod = base_frequency + 100 * np.sin(2 * np.pi * 8 * t)
            audio = amplitude * np.sin(2 * np.pi * freq_mod * t)
            # Add energy bursts
            energy_bursts = 0.02 * np.sin(2 * np.pi * 15 * t) ** 2
            audio += energy_bursts
            
        elif mood_type == "excited":
            # High amplitude, very dynamic
            base_frequency = 800.0
            amplitude = 0.18
            # Complex modulation
            freq_mod = base_frequency + 200 * np.sin(2 * np.pi * 12 * t)
            amp_mod = amplitude * (1 + 0.3 * np.sin(2 * np.pi * 7 * t))
            audio = amp_mod * np.sin(2 * np.pi * freq_mod * t)
            # Add noise for excitement
            noise = np.random.normal(0, amplitude * 0.1, samples)
            audio += noise
            
        else:
            # Default to neutral
            return self.generate_demo_audio("neutral", duration)
        
        # Add some background noise for realism
        background_noise = np.random.normal(0, 0.005, samples)
        audio += background_noise
        
        return audio.astype(np.float32)
    
    def run_debug_logging_demo(self):
        """Demonstrate debug logging capabilities."""
        print("\n" + "="*60)
        print("DEBUG LOGGING DEMONSTRATION")
        print("="*60)
        
        print("Processing different mood types and logging results...")
        
        mood_types = ["calm", "neutral", "energetic", "excited"]
        
        for mood_type in mood_types:
            print(f"\nProcessing {mood_type} audio...")
            
            # Generate and process audio
            audio = self.generate_demo_audio(mood_type, duration=0.5)
            features = self.feature_extractor.extract_features(audio, timestamp=time.time())
            mood_result = self.mood_detector.detect_mood(features)
            
            # Log with performance metrics
            performance_metrics = {
                'processing_time_ms': 15.0,  # Simulated
                'performance_level': 'high'
            }
            
            self.debug_logger.log_mood_detection(mood_result, performance_metrics)
            
            print(f"  Detected: {mood_result.mood} (confidence: {mood_result.confidence:.3f})")
            print(f"  RMS: {features.rms:.4f}, ZCR: {features.zero_crossing_rate:.4f}")
            print(f"  Spectral Centroid: {features.spectral_centroid:.1f} Hz")
        
        # Show recent entries
        print(f"\nRecent debug entries:")
        recent_entries = self.debug_logger.get_recent_entries(4)
        for i, entry in enumerate(recent_entries):
            print(f"  {i+1}. {entry.mood} (conf: {entry.confidence:.3f}) at {time.ctime(entry.timestamp)}")
        
        # Export debug data
        export_file = "demo_debug_export.json"
        if self.debug_logger.export_debug_data(export_file):
            print(f"\nDebug data exported to {export_file}")
        
        # Analyze patterns
        analysis = self.debug_logger.analyze_mood_patterns()
        print(f"\nMood Pattern Analysis:")
        print(f"  Total detections: {analysis['total_detections']}")
        print(f"  Mood distribution: {analysis['mood_distribution']}")
        print(f"  Average confidence by mood: {analysis['average_confidence_by_mood']}")
    
    def run_real_time_visualization_demo(self):
        """Demonstrate real-time visualization."""
        print("\n" + "="*60)
        print("REAL-TIME VISUALIZATION DEMONSTRATION")
        print("="*60)
        
        print("Starting real-time visualizer...")
        self.visualizer.start()
        
        print("Simulating real-time mood detection (10 seconds)...")
        print("Watch the live updates below:\n")
        
        # Simulate changing moods over time
        mood_sequence = [
            ("calm", 2.0),
            ("neutral", 2.0),
            ("energetic", 3.0),
            ("excited", 2.0),
            ("neutral", 1.0)
        ]
        
        for mood_type, duration in mood_sequence:
            end_time = time.time() + duration
            
            while time.time() < end_time:
                # Generate audio for current mood
                audio = self.generate_demo_audio(mood_type, duration=0.1)
                features = self.feature_extractor.extract_features(audio)
                mood_result = self.mood_detector.detect_mood(features)
                
                # Update visualizer
                self.visualizer.update(mood_result)
                
                time.sleep(0.1)  # 10 Hz update rate
        
        print("\nStopping visualizer...")
        self.visualizer.stop()
        
        # Show feature statistics
        stats = self.visualizer.get_feature_statistics()
        if stats:
            print("\nFeature Statistics Summary:")
            for feature, stat in stats.items():
                print(f"  {feature}: mean={stat['mean']:.4f}, std={stat['std']:.4f}")
    
    def run_diagnostic_analysis_demo(self):
        """Demonstrate diagnostic analysis."""
        print("\n" + "="*60)
        print("DIAGNOSTIC ANALYSIS DEMONSTRATION")
        print("="*60)
        
        print("Running comprehensive system diagnostic...")
        
        # Run full diagnostic
        results = self.diagnostic_analyzer.run_full_diagnostic()
        
        # Print formatted report
        self.diagnostic_analyzer.print_diagnostic_report(results)
        
        # Save detailed results
        import json
        with open("demo_diagnostic_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nDetailed diagnostic results saved to demo_diagnostic_results.json")
    
    def run_config_validation_demo(self):
        """Demonstrate configuration validation."""
        print("\n" + "="*60)
        print("CONFIGURATION VALIDATION DEMONSTRATION")
        print("="*60)
        
        # Validate current configuration
        print("Validating current configuration...")
        validation_results = self.config_validator.validate_configuration()
        
        print(f"Configuration file exists: {'✓' if validation_results['file_exists'] else '✗'}")
        print(f"Configuration valid: {'✓' if validation_results['valid'] else '✗'}")
        
        if validation_results['errors']:
            print("\nErrors found:")
            for error in validation_results['errors']:
                print(f"  - {error}")
        
        if validation_results['warnings']:
            print("\nWarnings:")
            for warning in validation_results['warnings']:
                print(f"  - {warning}")
        
        if validation_results['recommendations']:
            print("\nRecommendations:")
            for rec in validation_results['recommendations']:
                print(f"  - {rec}")
        
        # Export configuration template
        template_file = "demo_config_template.json"
        if self.config_validator.export_config_template(template_file):
            print(f"\nConfiguration template exported to {template_file}")
        
        # Show optimal configuration suggestion
        print("\nGenerating optimal configuration suggestion...")
        try:
            optimal_config = self.config_validator.suggest_optimal_config()
            print("Optimal configuration generated successfully")
            print(f"  Energy calm_max: {optimal_config.energy.calm_max}")
            print(f"  Energy energetic_min: {optimal_config.energy.energetic_min}")
            print(f"  Smoothing confidence_threshold: {optimal_config.smoothing.confidence_threshold}")
        except Exception as e:
            print(f"Failed to generate optimal configuration: {e}")
    
    def run_performance_monitoring_demo(self):
        """Demonstrate performance monitoring integration."""
        print("\n" + "="*60)
        print("PERFORMANCE MONITORING DEMONSTRATION")
        print("="*60)
        
        print("Demonstrating performance monitoring during mood detection...")
        
        # Process multiple audio samples while monitoring performance
        for i in range(20):
            self.performance_monitor.start_cycle()
            
            # Generate audio
            mood_type = ["calm", "neutral", "energetic", "excited"][i % 4]
            audio = self.generate_demo_audio(mood_type, duration=0.1)
            
            # Process with performance monitoring
            with self.performance_monitor.measure_stage('feature_extraction'):
                features = self.feature_extractor.extract_features(audio)
            
            with self.performance_monitor.measure_stage('mood_detection'):
                mood_result = self.mood_detector.detect_mood(features)
            
            with self.performance_monitor.measure_stage('transition_smoothing'):
                smoothed_mood = self.transition_smoother.smooth_transition(
                    mood_result.mood, mood_result.confidence
                )
            
            metrics = self.performance_monitor.end_cycle()
            
            if i % 5 == 0:  # Print every 5th cycle
                print(f"Cycle {i+1}: {metrics.total_duration*1000:.1f}ms total, "
                      f"level: {metrics.performance_level}")
        
        # Get performance summary
        summary = self.performance_monitor.get_performance_summary()
        print(f"\nPerformance Summary:")
        print(f"  Average processing time: {summary.get('average_processing_time_ms', 0):.1f}ms")
        print(f"  Current performance level: {summary.get('current_performance_level', 'unknown')}")
        print(f"  Total cycles processed: {summary.get('total_cycles', 0)}")
        print(f"  Audio underruns: {summary.get('audio_underruns', 0)}")
        
        # Save performance log
        log_file = "demo_performance_log.json"
        self.performance_monitor.save_performance_log(log_file)
        print(f"  Performance log saved to {log_file}")
    
    def run_complete_demo(self):
        """Run complete demonstration of all debugging and diagnostic tools."""
        print("ENHANCED MOOD DETECTION - DEBUG & DIAGNOSTIC TOOLS DEMO")
        print("=" * 80)
        print("This demo shows the debugging and diagnostic capabilities")
        print("of the enhanced mood detection system.")
        print("=" * 80)
        
        try:
            # Run all demonstrations
            self.run_debug_logging_demo()
            self.run_real_time_visualization_demo()
            self.run_diagnostic_analysis_demo()
            self.run_config_validation_demo()
            self.run_performance_monitoring_demo()
            
            print("\n" + "="*80)
            print("DEMO COMPLETE")
            print("="*80)
            print("All debugging and diagnostic tools demonstrated successfully!")
            print("\nGenerated files:")
            print("  - demo_debug_export.json (debug data)")
            print("  - demo_diagnostic_results.json (diagnostic results)")
            print("  - demo_config_template.json (configuration template)")
            print("  - demo_performance_log.json (performance log)")
            print("  - mood_debug.log (debug log file)")
            
        except KeyboardInterrupt:
            print("\nDemo interrupted by user")
        except Exception as e:
            print(f"\nDemo failed with error: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main function for running the demo."""
    if not ENHANCED_AVAILABLE:
        print("Enhanced mood detection components are not available.")
        print("Please ensure all required modules are installed.")
        return 1
    
    if len(sys.argv) > 1:
        demo_type = sys.argv[1]
        
        demo_system = DebugDemoSystem()
        
        if demo_type == "logging":
            demo_system.run_debug_logging_demo()
        elif demo_type == "visualization":
            demo_system.run_real_time_visualization_demo()
        elif demo_type == "diagnostic":
            demo_system.run_diagnostic_analysis_demo()
        elif demo_type == "validation":
            demo_system.run_config_validation_demo()
        elif demo_type == "performance":
            demo_system.run_performance_monitoring_demo()
        elif demo_type == "all":
            demo_system.run_complete_demo()
        else:
            print(f"Unknown demo type: {demo_type}")
            print("Available demos: logging, visualization, diagnostic, validation, performance, all")
            return 1
    else:
        # Run complete demo by default
        demo_system = DebugDemoSystem()
        demo_system.run_complete_demo()
    
    return 0


if __name__ == "__main__":
    exit(main())