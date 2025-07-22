#!/usr/bin/env python3
"""
Tests for mood configuration management system.
"""

import unittest
import tempfile
import os
import json
from unittest.mock import patch, mock_open
import logging

from mood_config import (
    MoodConfig, ConfigManager, ConfigValidationError,
    EnergyThresholds, SpectralThresholds, TemporalThresholds,
    SmoothingConfig, NoiseFilteringConfig
)


class TestMoodConfig(unittest.TestCase):
    """Test MoodConfig dataclass and its components."""
    
    def test_default_initialization(self):
        """Test that MoodConfig initializes with default values."""
        config = MoodConfig()
        
        # Check that all components are initialized
        self.assertIsInstance(config.energy, EnergyThresholds)
        self.assertIsInstance(config.spectral, SpectralThresholds)
        self.assertIsInstance(config.temporal, TemporalThresholds)
        self.assertIsInstance(config.smoothing, SmoothingConfig)
        self.assertIsInstance(config.noise_filtering, NoiseFilteringConfig)
        
        # Check default values match expected thresholds from led.py
        self.assertEqual(config.energy.calm_max, 0.02)
        self.assertEqual(config.energy.neutral_range, (0.02, 0.08))
        self.assertEqual(config.energy.energetic_min, 0.08)
        self.assertEqual(config.temporal.calm_zcr_max, 0.05)
        self.assertEqual(config.temporal.energetic_zcr_min, 0.15)
        self.assertEqual(config.spectral.bright_centroid_min, 3000.0)
    
    def test_custom_initialization(self):
        """Test MoodConfig with custom component values."""
        custom_energy = EnergyThresholds(calm_max=0.01, neutral_range=(0.01, 0.05))
        custom_smoothing = SmoothingConfig(transition_time=1.5, confidence_threshold=0.8)
        
        config = MoodConfig(energy=custom_energy, smoothing=custom_smoothing)
        
        self.assertEqual(config.energy.calm_max, 0.01)
        self.assertEqual(config.energy.neutral_range, (0.01, 0.05))
        self.assertEqual(config.smoothing.transition_time, 1.5)
        self.assertEqual(config.smoothing.confidence_threshold, 0.8)
        
        # Other components should use defaults
        self.assertEqual(config.spectral.bright_centroid_min, 3000.0)
        self.assertEqual(config.temporal.calm_zcr_max, 0.05)


class TestConfigManager(unittest.TestCase):
    """Test ConfigManager functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "test_config.json")
        self.config_manager = ConfigManager(self.config_path)
        
        # Suppress logging during tests
        logging.getLogger('mood_config').setLevel(logging.CRITICAL)
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.config_path):
            os.remove(self.config_path)
        os.rmdir(self.temp_dir)
    
    def test_get_default_config(self):
        """Test getting default configuration."""
        config = self.config_manager.get_default_config()
        
        self.assertIsInstance(config, MoodConfig)
        # Verify it matches the thresholds from led.py
        self.assertEqual(config.energy.calm_max, 0.02)
        self.assertEqual(config.energy.neutral_range, (0.02, 0.08))
        self.assertEqual(config.energy.energetic_min, 0.08)
        self.assertEqual(config.temporal.calm_zcr_max, 0.05)
        self.assertEqual(config.temporal.energetic_zcr_min, 0.15)
        self.assertEqual(config.spectral.bright_centroid_min, 3000.0)
    
    def test_load_config_file_not_exists(self):
        """Test loading config when file doesn't exist."""
        config = self.config_manager.load_config()
        
        # Should return default config
        default_config = self.config_manager.get_default_config()
        self.assertEqual(config.energy.calm_max, default_config.energy.calm_max)
        self.assertEqual(config.smoothing.transition_time, default_config.smoothing.transition_time)
    
    def test_save_and_load_config(self):
        """Test saving and loading configuration."""
        # Create custom config
        original_config = MoodConfig()
        original_config.energy.calm_max = 0.025
        original_config.smoothing.transition_time = 1.5
        original_config.noise_filtering.adaptive_gain = False
        
        # Save config
        success = self.config_manager.save_config(original_config)
        self.assertTrue(success)
        self.assertTrue(os.path.exists(self.config_path))
        
        # Load config
        loaded_config = self.config_manager.load_config()
        
        # Verify values match
        self.assertEqual(loaded_config.energy.calm_max, 0.025)
        self.assertEqual(loaded_config.smoothing.transition_time, 1.5)
        self.assertEqual(loaded_config.noise_filtering.adaptive_gain, False)
    
    def test_load_invalid_json(self):
        """Test loading config with invalid JSON."""
        # Write invalid JSON to file
        with open(self.config_path, 'w') as f:
            f.write("{ invalid json }")
        
        # Should return default config
        config = self.config_manager.load_config()
        default_config = self.config_manager.get_default_config()
        self.assertEqual(config.energy.calm_max, default_config.energy.calm_max)
    
    def test_load_partial_config(self):
        """Test loading config with missing fields."""
        # Write partial config
        partial_config = {
            "thresholds": {
                "energy": {
                    "calm_max": 0.03
                }
            },
            "smoothing": {
                "transition_time": 3.0
            }
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(partial_config, f)
        
        config = self.config_manager.load_config()
        
        # Should have custom values where specified
        self.assertEqual(config.energy.calm_max, 0.03)
        self.assertEqual(config.smoothing.transition_time, 3.0)
        
        # Should have defaults for missing values
        self.assertEqual(config.energy.neutral_range, (0.02, 0.08))
        self.assertEqual(config.smoothing.confidence_threshold, 0.7)
    
    def test_update_thresholds(self):
        """Test updating specific threshold values."""
        # Update some thresholds
        success = self.config_manager.update_thresholds(
            calm_max=0.03,
            transition_time=2.5,
            adaptive_gain=False
        )
        self.assertTrue(success)
        
        # Load and verify
        config = self.config_manager.load_config()
        self.assertEqual(config.energy.calm_max, 0.03)
        self.assertEqual(config.smoothing.transition_time, 2.5)
        self.assertEqual(config.noise_filtering.adaptive_gain, False)
        
        # Other values should remain default
        self.assertEqual(config.energy.energetic_min, 0.08)
        self.assertEqual(config.smoothing.confidence_threshold, 0.7)


class TestConfigValidation(unittest.TestCase):
    """Test configuration validation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config_manager = ConfigManager()
        # Suppress logging during tests
        logging.getLogger('mood_config').setLevel(logging.CRITICAL)
    
    def test_valid_config(self):
        """Test that valid config passes validation."""
        config = self.config_manager.get_default_config()
        
        # Should not raise exception
        try:
            self.config_manager.validate_config(config)
        except ConfigValidationError:
            self.fail("Valid config failed validation")
    
    def test_invalid_energy_thresholds(self):
        """Test validation of energy thresholds."""
        config = self.config_manager.get_default_config()
        
        # Test negative calm_max
        config.energy.calm_max = -0.01
        with self.assertRaises(ConfigValidationError):
            self.config_manager.validate_config(config)
        
        # Reset and test invalid neutral range
        config = self.config_manager.get_default_config()
        config.energy.neutral_range = (0.08, 0.02)  # min > max
        with self.assertRaises(ConfigValidationError):
            self.config_manager.validate_config(config)
        
        # Test energetic_min <= neutral_range max
        config = self.config_manager.get_default_config()
        config.energy.energetic_min = 0.05  # <= neutral_range[1] (0.08)
        with self.assertRaises(ConfigValidationError):
            self.config_manager.validate_config(config)
        
        # Test excited_min <= energetic_min
        config = self.config_manager.get_default_config()
        config.energy.excited_min = 0.07  # <= energetic_min (0.08)
        with self.assertRaises(ConfigValidationError):
            self.config_manager.validate_config(config)
    
    def test_invalid_spectral_thresholds(self):
        """Test validation of spectral thresholds."""
        config = self.config_manager.get_default_config()
        
        # Test negative calm_centroid_max
        config.spectral.calm_centroid_max = -100
        with self.assertRaises(ConfigValidationError):
            self.config_manager.validate_config(config)
        
        # Test bright_centroid_min <= calm_centroid_max
        config = self.config_manager.get_default_config()
        config.spectral.bright_centroid_min = 1000  # <= calm_centroid_max (2000)
        with self.assertRaises(ConfigValidationError):
            self.config_manager.validate_config(config)
        
        # Test invalid rolloff thresholds (not increasing)
        config = self.config_manager.get_default_config()
        config.spectral.rolloff_thresholds = (3000, 2000, 5000)  # not increasing
        with self.assertRaises(ConfigValidationError):
            self.config_manager.validate_config(config)
        
        # Test wrong number of rolloff thresholds
        config = self.config_manager.get_default_config()
        config.spectral.rolloff_thresholds = (1500, 3000)  # only 2 values
        with self.assertRaises(ConfigValidationError):
            self.config_manager.validate_config(config)
    
    def test_invalid_temporal_thresholds(self):
        """Test validation of temporal thresholds."""
        config = self.config_manager.get_default_config()
        
        # Test calm_zcr_max out of range
        config.temporal.calm_zcr_max = 1.5  # > 1
        with self.assertRaises(ConfigValidationError):
            self.config_manager.validate_config(config)
        
        config = self.config_manager.get_default_config()
        config.temporal.calm_zcr_max = -0.1  # < 0
        with self.assertRaises(ConfigValidationError):
            self.config_manager.validate_config(config)
        
        # Test energetic_zcr_min <= calm_zcr_max
        config = self.config_manager.get_default_config()
        config.temporal.energetic_zcr_min = 0.03  # <= calm_zcr_max (0.05)
        with self.assertRaises(ConfigValidationError):
            self.config_manager.validate_config(config)
        
        # Test energetic_zcr_min >= 1
        config = self.config_manager.get_default_config()
        config.temporal.energetic_zcr_min = 1.0  # >= 1
        with self.assertRaises(ConfigValidationError):
            self.config_manager.validate_config(config)
    
    def test_invalid_smoothing_config(self):
        """Test validation of smoothing configuration."""
        config = self.config_manager.get_default_config()
        
        # Test negative transition_time
        config.smoothing.transition_time = -1.0
        with self.assertRaises(ConfigValidationError):
            self.config_manager.validate_config(config)
        
        # Test negative minimum_duration
        config = self.config_manager.get_default_config()
        config.smoothing.minimum_duration = -2.0
        with self.assertRaises(ConfigValidationError):
            self.config_manager.validate_config(config)
        
        # Test confidence_threshold out of range
        config = self.config_manager.get_default_config()
        config.smoothing.confidence_threshold = 1.5  # > 1
        with self.assertRaises(ConfigValidationError):
            self.config_manager.validate_config(config)
        
        config = self.config_manager.get_default_config()
        config.smoothing.confidence_threshold = 0.0  # <= 0
        with self.assertRaises(ConfigValidationError):
            self.config_manager.validate_config(config)
    
    def test_invalid_noise_filtering_config(self):
        """Test validation of noise filtering configuration."""
        config = self.config_manager.get_default_config()
        
        # Test negative noise_gate_threshold
        config.noise_filtering.noise_gate_threshold = -0.01
        with self.assertRaises(ConfigValidationError):
            self.config_manager.validate_config(config)
        
        # Test background_learning_rate out of range
        config = self.config_manager.get_default_config()
        config.noise_filtering.background_learning_rate = 1.5  # > 1
        with self.assertRaises(ConfigValidationError):
            self.config_manager.validate_config(config)
        
        config = self.config_manager.get_default_config()
        config.noise_filtering.background_learning_rate = 0.0  # <= 0
        with self.assertRaises(ConfigValidationError):
            self.config_manager.validate_config(config)


class TestConfigSerialization(unittest.TestCase):
    """Test configuration serialization and deserialization."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config_manager = ConfigManager()
    
    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = self.config_manager.get_default_config()
        config_dict = self.config_manager._config_to_dict(config)
        
        # Check structure
        self.assertIn("thresholds", config_dict)
        self.assertIn("smoothing", config_dict)
        self.assertIn("noise_filtering", config_dict)
        
        # Check threshold values
        thresholds = config_dict["thresholds"]
        self.assertEqual(thresholds["energy"]["calm_max"], 0.02)
        self.assertEqual(thresholds["energy"]["neutral_range"], [0.02, 0.08])
        self.assertEqual(thresholds["spectral"]["bright_centroid_min"], 3000.0)
        self.assertEqual(thresholds["temporal"]["calm_zcr_max"], 0.05)
        
        # Check smoothing values
        smoothing = config_dict["smoothing"]
        self.assertEqual(smoothing["transition_time"], 2.0)
        self.assertEqual(smoothing["confidence_threshold"], 0.7)
        
        # Check noise filtering values
        noise = config_dict["noise_filtering"]
        self.assertEqual(noise["adaptive_gain"], True)
        self.assertEqual(noise["background_learning_rate"], 0.1)
    
    def test_dict_to_config(self):
        """Test converting dictionary to config."""
        config_dict = {
            "thresholds": {
                "energy": {
                    "calm_max": 0.025,
                    "neutral_range": [0.025, 0.09],
                    "energetic_min": 0.09,
                    "excited_min": 0.16
                },
                "spectral": {
                    "calm_centroid_max": 2100.0,
                    "bright_centroid_min": 3100.0,
                    "rolloff_thresholds": [1600.0, 3100.0, 5100.0]
                },
                "temporal": {
                    "calm_zcr_max": 0.06,
                    "energetic_zcr_min": 0.16
                }
            },
            "smoothing": {
                "transition_time": 2.5,
                "minimum_duration": 6.0,
                "confidence_threshold": 0.75
            },
            "noise_filtering": {
                "noise_gate_threshold": 0.015,
                "adaptive_gain": False,
                "background_learning_rate": 0.15
            }
        }
        
        config = self.config_manager._dict_to_config(config_dict)
        
        # Check energy values
        self.assertEqual(config.energy.calm_max, 0.025)
        self.assertEqual(config.energy.neutral_range, (0.025, 0.09))
        self.assertEqual(config.energy.energetic_min, 0.09)
        self.assertEqual(config.energy.excited_min, 0.16)
        
        # Check spectral values
        self.assertEqual(config.spectral.calm_centroid_max, 2100.0)
        self.assertEqual(config.spectral.bright_centroid_min, 3100.0)
        self.assertEqual(config.spectral.rolloff_thresholds, (1600.0, 3100.0, 5100.0))
        
        # Check temporal values
        self.assertEqual(config.temporal.calm_zcr_max, 0.06)
        self.assertEqual(config.temporal.energetic_zcr_min, 0.16)
        
        # Check smoothing values
        self.assertEqual(config.smoothing.transition_time, 2.5)
        self.assertEqual(config.smoothing.minimum_duration, 6.0)
        self.assertEqual(config.smoothing.confidence_threshold, 0.75)
        
        # Check noise filtering values
        self.assertEqual(config.noise_filtering.noise_gate_threshold, 0.015)
        self.assertEqual(config.noise_filtering.adaptive_gain, False)
        self.assertEqual(config.noise_filtering.background_learning_rate, 0.15)
    
    def test_roundtrip_serialization(self):
        """Test that config survives roundtrip serialization."""
        original_config = self.config_manager.get_default_config()
        
        # Modify some values
        original_config.energy.calm_max = 0.025
        original_config.smoothing.transition_time = 1.8
        original_config.noise_filtering.adaptive_gain = False
        
        # Convert to dict and back
        config_dict = self.config_manager._config_to_dict(original_config)
        restored_config = self.config_manager._dict_to_config(config_dict)
        
        # Check that values are preserved
        self.assertEqual(restored_config.energy.calm_max, original_config.energy.calm_max)
        self.assertEqual(restored_config.smoothing.transition_time, original_config.smoothing.transition_time)
        self.assertEqual(restored_config.noise_filtering.adaptive_gain, original_config.noise_filtering.adaptive_gain)
        
        # Check that other values are also preserved
        self.assertEqual(restored_config.energy.neutral_range, original_config.energy.neutral_range)
        self.assertEqual(restored_config.spectral.rolloff_thresholds, original_config.spectral.rolloff_thresholds)
        self.assertEqual(restored_config.temporal.energetic_zcr_min, original_config.temporal.energetic_zcr_min)


if __name__ == '__main__':
    # Set up logging for tests
    logging.basicConfig(level=logging.INFO)
    
    unittest.main()