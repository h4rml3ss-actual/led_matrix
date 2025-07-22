#!/usr/bin/env python3
"""
Demonstration of the mood configuration management system.
Shows how to load, modify, and save configuration.
"""

import os
from mood_config import ConfigManager, MoodConfig

def main():
    """Demonstrate configuration management functionality."""
    print("=== Mood Detection Configuration System Demo ===\n")
    
    # Initialize config manager
    config_manager = ConfigManager("demo_mood_config.json")
    
    # 1. Load default configuration
    print("1. Loading default configuration...")
    config = config_manager.load_config()
    print(f"   Energy calm_max: {config.energy.calm_max}")
    print(f"   Energy neutral_range: {config.energy.neutral_range}")
    print(f"   Spectral bright_centroid_min: {config.spectral.bright_centroid_min}")
    print(f"   Smoothing transition_time: {config.smoothing.transition_time}")
    print(f"   Noise filtering adaptive_gain: {config.noise_filtering.adaptive_gain}")
    
    # 2. Save configuration to file
    print("\n2. Saving configuration to file...")
    success = config_manager.save_config(config)
    print(f"   Save successful: {success}")
    print(f"   Config file exists: {os.path.exists('demo_mood_config.json')}")
    
    # 3. Modify some thresholds
    print("\n3. Updating specific thresholds...")
    success = config_manager.update_thresholds(
        calm_max=0.025,
        transition_time=1.5,
        adaptive_gain=False,
        confidence_threshold=0.8
    )
    print(f"   Update successful: {success}")
    
    # 4. Load updated configuration
    print("\n4. Loading updated configuration...")
    updated_config = config_manager.load_config()
    print(f"   Energy calm_max: {updated_config.energy.calm_max}")
    print(f"   Smoothing transition_time: {updated_config.smoothing.transition_time}")
    print(f"   Noise filtering adaptive_gain: {updated_config.noise_filtering.adaptive_gain}")
    print(f"   Smoothing confidence_threshold: {updated_config.smoothing.confidence_threshold}")
    
    # 5. Demonstrate validation
    print("\n5. Testing configuration validation...")
    try:
        # Try invalid configuration
        invalid_config = MoodConfig()
        invalid_config.energy.calm_max = -0.01  # Invalid: negative value
        config_manager.validate_config(invalid_config)
        print("   ERROR: Invalid config passed validation!")
    except Exception as e:
        print(f"   Validation correctly caught error: {e}")
    
    # 6. Show configuration structure
    print("\n6. Configuration structure:")
    print("   Energy thresholds:")
    print(f"     - calm_max: {updated_config.energy.calm_max}")
    print(f"     - neutral_range: {updated_config.energy.neutral_range}")
    print(f"     - energetic_min: {updated_config.energy.energetic_min}")
    print(f"     - excited_min: {updated_config.energy.excited_min}")
    
    print("   Spectral thresholds:")
    print(f"     - calm_centroid_max: {updated_config.spectral.calm_centroid_max}")
    print(f"     - bright_centroid_min: {updated_config.spectral.bright_centroid_min}")
    print(f"     - rolloff_thresholds: {updated_config.spectral.rolloff_thresholds}")
    
    print("   Temporal thresholds:")
    print(f"     - calm_zcr_max: {updated_config.temporal.calm_zcr_max}")
    print(f"     - energetic_zcr_min: {updated_config.temporal.energetic_zcr_min}")
    
    print("   Smoothing configuration:")
    print(f"     - transition_time: {updated_config.smoothing.transition_time}")
    print(f"     - minimum_duration: {updated_config.smoothing.minimum_duration}")
    print(f"     - confidence_threshold: {updated_config.smoothing.confidence_threshold}")
    
    print("   Noise filtering configuration:")
    print(f"     - noise_gate_threshold: {updated_config.noise_filtering.noise_gate_threshold}")
    print(f"     - adaptive_gain: {updated_config.noise_filtering.adaptive_gain}")
    print(f"     - background_learning_rate: {updated_config.noise_filtering.background_learning_rate}")
    
    # Clean up demo file
    if os.path.exists("demo_mood_config.json"):
        os.remove("demo_mood_config.json")
        print("\n   Demo config file cleaned up.")
    
    print("\n=== Demo Complete ===")

if __name__ == "__main__":
    main()