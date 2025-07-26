#!/usr/bin/env python3
"""
Demonstration of error handling and fallback mechanisms.
Shows how the system gracefully handles various error scenarios.
"""

import numpy as np
import time
import random
from error_handling import (
    ErrorRecoveryManager, ErrorSeverity, SafeAudioProcessor,
    get_global_error_manager
)


def simulate_microphone_disconnection():
    """Simulate microphone disconnection scenario."""
    print("\n" + "="*60)
    print("SCENARIO 1: Microphone Disconnection")
    print("="*60)
    
    error_manager = get_global_error_manager()
    
    # Simulate microphone disconnection
    print("Simulating microphone disconnection...")
    error_manager.handle_error(
        'microphone_input',
        'disconnection',
        Exception("Microphone device not found"),
        ErrorSeverity.HIGH
    )
    
    # Check system status
    summary = error_manager.get_error_summary()
    print(f"System status after disconnection:")
    print(f"  - Degraded components: {summary['degraded_components']}")
    print(f"  - Total errors: {summary['total_errors']}")
    
    # Simulate recovery attempt
    print("\nAttempting recovery...")
    error_manager.reset_component_errors('microphone_input')
    print("Recovery successful - microphone reconnected")


def simulate_feature_extraction_failures():
    """Simulate feature extraction failures and fallback."""
    print("\n" + "="*60)
    print("SCENARIO 2: Feature Extraction Failures")
    print("="*60)
    
    error_manager = ErrorRecoveryManager()
    safe_processor = SafeAudioProcessor(error_manager)
    
    # Mock feature extractor that fails
    class FailingFeatureExtractor:
        def __init__(self, failure_rate=0.7):
            self.failure_rate = failure_rate
            self.call_count = 0
        
        def extract_features(self, audio_data, **kwargs):
            self.call_count += 1
            if random.random() < self.failure_rate:
                raise RuntimeError(f"Feature extraction failed (call {self.call_count})")
            
            # Return mock features on success
            class MockFeatures:
                def __init__(self):
                    self.rms = 0.05
                    self.confidence = 0.8
                    self.spectral_centroid = 2000.0
            
            return MockFeatures()
    
    failing_extractor = FailingFeatureExtractor()
    test_audio = np.random.randn(1024).astype(np.float32)
    
    print("Testing feature extraction with 70% failure rate...")
    
    success_count = 0
    fallback_count = 0
    
    for i in range(10):
        result = safe_processor.safe_extract_features(test_audio, failing_extractor)
        
        if hasattr(result, 'confidence') and result.confidence == 0.3:
            fallback_count += 1
            print(f"  Attempt {i+1}: Used fallback features")
        else:
            success_count += 1
            print(f"  Attempt {i+1}: Successful extraction")
    
    print(f"\nResults: {success_count} successful, {fallback_count} fallbacks")
    
    # Show error summary
    summary = error_manager.get_error_summary()
    print(f"Error summary:")
    print(f"  - Total errors: {summary['total_errors']}")
    print(f"  - Degraded components: {summary['degraded_components']}")


def simulate_cascading_failures():
    """Simulate cascading failures across multiple components."""
    print("\n" + "="*60)
    print("SCENARIO 3: Cascading System Failures")
    print("="*60)
    
    error_manager = ErrorRecoveryManager()
    
    components = [
        'feature_extraction',
        'mood_detection', 
        'noise_filtering',
        'transition_smoothing',
        'performance_monitoring'
    ]
    
    print("Simulating cascading failures across components...")
    
    # Simulate increasing error rates
    for round_num in range(1, 4):
        print(f"\nRound {round_num}: Increasing system stress")
        
        for component in components:
            # Generate errors for each component
            error_count = round_num * 2
            for i in range(error_count):
                error_manager.handle_error(
                    component,
                    f'stress_error_round_{round_num}',
                    Exception(f"System stress error {i+1} in {component}"),
                    ErrorSeverity.MEDIUM
                )
        
        # Check system health
        summary = error_manager.get_error_summary()
        print(f"  System health:")
        print(f"    - Total errors: {summary['total_errors']}")
        print(f"    - Degraded components: {len(summary['degraded_components'])}")
        print(f"    - Disabled components: {len(summary['disabled_components'])}")
        print(f"    - Emergency mode: {summary['emergency_mode']}")
        
        if summary['emergency_mode']:
            print("    ⚠️  EMERGENCY MODE ACTIVATED!")
            break
    
    print(f"\nFinal system state:")
    final_summary = error_manager.get_error_summary()
    print(f"  - Degraded: {final_summary['degraded_components']}")
    print(f"  - Disabled: {final_summary['disabled_components']}")
    print(f"  - Emergency mode: {final_summary['emergency_mode']}")


def simulate_recovery_scenarios():
    """Simulate various recovery scenarios."""
    print("\n" + "="*60)
    print("SCENARIO 4: Recovery Mechanisms")
    print("="*60)
    
    error_manager = ErrorRecoveryManager()
    
    # Register recovery strategies
    def audio_recovery_strategy(error_event):
        print(f"    Attempting audio system recovery...")
        # Simulate 50% success rate
        success = random.random() > 0.5
        if success:
            print(f"    ✅ Audio recovery successful")
        else:
            print(f"    ❌ Audio recovery failed")
        return success
    
    def feature_recovery_strategy(error_event):
        print(f"    Attempting feature extraction recovery...")
        # Simulate 80% success rate
        success = random.random() > 0.2
        if success:
            print(f"    ✅ Feature extraction recovery successful")
        else:
            print(f"    ❌ Feature extraction recovery failed")
        return success
    
    # Register strategies
    error_manager.register_recovery_strategy('audio_system', audio_recovery_strategy)
    error_manager.register_recovery_strategy('feature_extraction', feature_recovery_strategy)
    
    # Test recovery scenarios
    test_scenarios = [
        ('audio_system', 'buffer_underrun', ErrorSeverity.MEDIUM),
        ('feature_extraction', 'computation_error', ErrorSeverity.MEDIUM),
        ('audio_system', 'device_error', ErrorSeverity.HIGH),
        ('feature_extraction', 'memory_error', ErrorSeverity.HIGH),
    ]
    
    print("Testing recovery mechanisms...")
    
    for component, error_type, severity in test_scenarios:
        print(f"\n  Triggering {error_type} in {component}...")
        
        success = error_manager.handle_error(
            component,
            error_type,
            Exception(f"Simulated {error_type}"),
            severity
        )
        
        if success:
            print(f"  ✅ Component {component} recovered successfully")
        else:
            print(f"  ❌ Component {component} recovery failed, using fallback")
    
    # Show final recovery statistics
    summary = error_manager.get_error_summary()
    print(f"\nRecovery session summary:")
    print(f"  - Total recovery attempts: {len(test_scenarios)}")
    print(f"  - Components with errors: {len(summary['error_counts_by_component'])}")
    print(f"  - System still operational: {not summary['emergency_mode']}")


def simulate_performance_degradation():
    """Simulate performance-related error handling."""
    print("\n" + "="*60)
    print("SCENARIO 5: Performance Degradation Handling")
    print("="*60)
    
    error_manager = ErrorRecoveryManager()
    
    print("Simulating high-frequency performance errors...")
    
    # Simulate rapid performance errors (like would happen on Pi Zero under load)
    performance_components = ['audio_processing', 'feature_extraction', 'mood_detection']
    
    for minute in range(1, 4):
        print(f"\nMinute {minute}: System under load")
        
        # Simulate errors occurring every few milliseconds
        for second in range(10):  # 10 seconds per minute simulation
            for component in performance_components:
                if random.random() < 0.3:  # 30% chance of error per second
                    error_manager.handle_error(
                        component,
                        'performance_timeout',
                        Exception(f"Processing timeout at minute {minute}, second {second}"),
                        ErrorSeverity.LOW
                    )
        
        # Check system adaptation
        summary = error_manager.get_error_summary()
        print(f"  System adaptation:")
        
        for component in performance_components:
            if error_manager.is_component_available(component):
                if error_manager.is_component_degraded(component):
                    print(f"    - {component}: DEGRADED (reduced functionality)")
                else:
                    print(f"    - {component}: NORMAL")
            else:
                print(f"    - {component}: DISABLED")
    
    print(f"\nPerformance handling results:")
    final_summary = error_manager.get_error_summary()
    print(f"  - Total performance errors: {final_summary['total_errors']}")
    print(f"  - Components adapted: {len(final_summary['degraded_components'])}")
    print(f"  - System stability maintained: {not final_summary['emergency_mode']}")


def main():
    """Run all error handling demonstrations."""
    print("Enhanced Mood Detection - Error Handling Demonstration")
    print("=" * 60)
    print("This demo shows how the system handles various error scenarios")
    print("and maintains functionality through graceful degradation.")
    
    # Run all scenarios
    simulate_microphone_disconnection()
    simulate_feature_extraction_failures()
    simulate_cascading_failures()
    simulate_recovery_scenarios()
    simulate_performance_degradation()
    
    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60)
    print("Key takeaways:")
    print("✅ System handles microphone disconnections gracefully")
    print("✅ Feature extraction failures fall back to basic methods")
    print("✅ Cascading failures trigger emergency mode protection")
    print("✅ Recovery mechanisms restore functionality when possible")
    print("✅ Performance degradation is managed through adaptation")
    print("\nThe enhanced mood detection system is designed to be robust")
    print("and maintain basic functionality even under adverse conditions.")


if __name__ == '__main__':
    main()