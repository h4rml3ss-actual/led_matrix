#!/usr/bin/env python3
"""
Configuration management system for enhanced mood detection.
Handles loading, saving, and validation of mood detection parameters.
"""

import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
import logging


@dataclass
class EnergyThresholds:
    """Energy-based thresholds for mood detection."""
    calm_max: float = 0.02
    neutral_range: tuple = (0.02, 0.08)
    energetic_min: float = 0.08
    excited_min: float = 0.15


@dataclass
class SpectralThresholds:
    """Spectral feature thresholds for mood detection."""
    calm_centroid_max: float = 2000.0
    bright_centroid_min: float = 3000.0
    rolloff_thresholds: tuple = (1500.0, 3000.0, 5000.0)


@dataclass
class TemporalThresholds:
    """Temporal feature thresholds for mood detection."""
    calm_zcr_max: float = 0.05
    energetic_zcr_min: float = 0.15


@dataclass
class SmoothingConfig:
    """Configuration for mood transition smoothing."""
    transition_time: float = 2.0
    minimum_duration: float = 5.0
    confidence_threshold: float = 0.7


@dataclass
class NoiseFilteringConfig:
    """Configuration for noise filtering and adaptation."""
    noise_gate_threshold: float = 0.01
    adaptive_gain: bool = True
    background_learning_rate: float = 0.1


@dataclass
class MoodConfig:
    """
    Complete configuration for mood detection system.
    Contains all thresholds and parameters needed for mood detection.
    """
    energy: EnergyThresholds
    spectral: SpectralThresholds
    temporal: TemporalThresholds
    smoothing: SmoothingConfig
    noise_filtering: NoiseFilteringConfig
    
    def __init__(self, 
                 energy: Optional[EnergyThresholds] = None,
                 spectral: Optional[SpectralThresholds] = None,
                 temporal: Optional[TemporalThresholds] = None,
                 smoothing: Optional[SmoothingConfig] = None,
                 noise_filtering: Optional[NoiseFilteringConfig] = None):
        self.energy = energy or EnergyThresholds()
        self.spectral = spectral or SpectralThresholds()
        self.temporal = temporal or TemporalThresholds()
        self.smoothing = smoothing or SmoothingConfig()
        self.noise_filtering = noise_filtering or NoiseFilteringConfig()


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    pass


class ConfigManager:
    """
    Manages loading, saving, and validation of mood detection configuration.
    """
    
    DEFAULT_CONFIG_PATH = "mood_config.json"
    
    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH):
        self.config_path = config_path
        self.logger = logging.getLogger(__name__)
        
    def load_config(self) -> MoodConfig:
        """
        Load configuration from JSON file.
        Returns default configuration if file doesn't exist or is invalid.
        
        Returns:
            MoodConfig: Loaded or default configuration
        """
        if not os.path.exists(self.config_path):
            self.logger.info(f"Config file {self.config_path} not found, using defaults")
            return self.get_default_config()
        
        try:
            with open(self.config_path, 'r') as f:
                config_data = json.load(f)
            
            config = self._dict_to_config(config_data)
            self.validate_config(config)
            self.logger.info(f"Successfully loaded config from {self.config_path}")
            return config
            
        except (json.JSONDecodeError, KeyError, TypeError, ConfigValidationError) as e:
            self.logger.error(f"Error loading config from {self.config_path}: {e}")
            self.logger.info("Using default configuration")
            return self.get_default_config()
    
    def save_config(self, config: MoodConfig) -> bool:
        """
        Save configuration to JSON file.
        
        Args:
            config: MoodConfig to save
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.validate_config(config)
            config_dict = self._config_to_dict(config)
            
            # Create directory if it doesn't exist
            config_dir = os.path.dirname(self.config_path)
            if config_dir:
                os.makedirs(config_dir, exist_ok=True)
            
            with open(self.config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            self.logger.info(f"Successfully saved config to {self.config_path}")
            return True
            
        except (ConfigValidationError, IOError, OSError) as e:
            self.logger.error(f"Error saving config to {self.config_path}: {e}")
            return False
    
    def get_default_config(self) -> MoodConfig:
        """
        Get default configuration based on current threshold values from led.py.
        
        Returns:
            MoodConfig: Default configuration
        """
        # Based on the thresholds found in led.py detect_mood function:
        # TH_RMS_LOW, TH_RMS_HIGH = 0.02, 0.08
        # TH_ZCR_LOW, TH_ZCR_HIGH = 0.05, 0.15
        # TH_CENTROID_HIGH = 3000  # Hz
        
        return MoodConfig(
            energy=EnergyThresholds(
                calm_max=0.02,
                neutral_range=(0.02, 0.08),
                energetic_min=0.08,
                excited_min=0.15
            ),
            spectral=SpectralThresholds(
                calm_centroid_max=2000.0,
                bright_centroid_min=3000.0,
                rolloff_thresholds=(1500.0, 3000.0, 5000.0)
            ),
            temporal=TemporalThresholds(
                calm_zcr_max=0.05,
                energetic_zcr_min=0.15
            ),
            smoothing=SmoothingConfig(
                transition_time=2.0,
                minimum_duration=5.0,
                confidence_threshold=0.7
            ),
            noise_filtering=NoiseFilteringConfig(
                noise_gate_threshold=0.01,
                adaptive_gain=True,
                background_learning_rate=0.1
            )
        )
    
    def validate_config(self, config: MoodConfig) -> None:
        """
        Validate configuration parameters.
        
        Args:
            config: MoodConfig to validate
            
        Raises:
            ConfigValidationError: If validation fails
        """
        # Validate energy thresholds
        if config.energy.calm_max <= 0:
            raise ConfigValidationError("Energy calm_max must be positive")
        
        if config.energy.neutral_range[0] >= config.energy.neutral_range[1]:
            raise ConfigValidationError("Energy neutral_range must be (min, max) with min < max")
        
        if config.energy.energetic_min < config.energy.neutral_range[1]:
            raise ConfigValidationError("Energy energetic_min must be greater than or equal to neutral_range max")
        
        if config.energy.excited_min <= config.energy.energetic_min:
            raise ConfigValidationError("Energy excited_min must be greater than energetic_min")
        
        # Validate spectral thresholds
        if config.spectral.calm_centroid_max <= 0:
            raise ConfigValidationError("Spectral calm_centroid_max must be positive")
        
        if config.spectral.bright_centroid_min <= config.spectral.calm_centroid_max:
            raise ConfigValidationError("Spectral bright_centroid_min must be greater than calm_centroid_max")
        
        rolloff = config.spectral.rolloff_thresholds
        if len(rolloff) != 3 or not all(rolloff[i] < rolloff[i+1] for i in range(len(rolloff)-1)):
            raise ConfigValidationError("Spectral rolloff_thresholds must be 3 increasing values")
        
        # Validate temporal thresholds
        if config.temporal.calm_zcr_max <= 0 or config.temporal.calm_zcr_max >= 1:
            raise ConfigValidationError("Temporal calm_zcr_max must be between 0 and 1")
        
        if config.temporal.energetic_zcr_min <= config.temporal.calm_zcr_max or config.temporal.energetic_zcr_min >= 1:
            raise ConfigValidationError("Temporal energetic_zcr_min must be between calm_zcr_max and 1")
        
        # Validate smoothing config
        if config.smoothing.transition_time <= 0:
            raise ConfigValidationError("Smoothing transition_time must be positive")
        
        if config.smoothing.minimum_duration <= 0:
            raise ConfigValidationError("Smoothing minimum_duration must be positive")
        
        if config.smoothing.confidence_threshold <= 0 or config.smoothing.confidence_threshold > 1:
            raise ConfigValidationError("Smoothing confidence_threshold must be between 0 and 1")
        
        # Validate noise filtering config
        if config.noise_filtering.noise_gate_threshold < 0:
            raise ConfigValidationError("Noise filtering noise_gate_threshold must be non-negative")
        
        if config.noise_filtering.background_learning_rate <= 0 or config.noise_filtering.background_learning_rate > 1:
            raise ConfigValidationError("Noise filtering background_learning_rate must be between 0 and 1")
    
    def _config_to_dict(self, config: MoodConfig) -> Dict[str, Any]:
        """Convert MoodConfig to dictionary for JSON serialization."""
        return {
            "thresholds": {
                "energy": {
                    "calm_max": config.energy.calm_max,
                    "neutral_range": list(config.energy.neutral_range),
                    "energetic_min": config.energy.energetic_min,
                    "excited_min": config.energy.excited_min
                },
                "spectral": {
                    "calm_centroid_max": config.spectral.calm_centroid_max,
                    "bright_centroid_min": config.spectral.bright_centroid_min,
                    "rolloff_thresholds": list(config.spectral.rolloff_thresholds)
                },
                "temporal": {
                    "calm_zcr_max": config.temporal.calm_zcr_max,
                    "energetic_zcr_min": config.temporal.energetic_zcr_min
                }
            },
            "smoothing": {
                "transition_time": config.smoothing.transition_time,
                "minimum_duration": config.smoothing.minimum_duration,
                "confidence_threshold": config.smoothing.confidence_threshold
            },
            "noise_filtering": {
                "noise_gate_threshold": config.noise_filtering.noise_gate_threshold,
                "adaptive_gain": config.noise_filtering.adaptive_gain,
                "background_learning_rate": config.noise_filtering.background_learning_rate
            }
        }
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> MoodConfig:
        """Convert dictionary to MoodConfig object."""
        thresholds = config_dict.get("thresholds", {})
        
        energy = EnergyThresholds(
            calm_max=thresholds.get("energy", {}).get("calm_max", 0.02),
            neutral_range=tuple(thresholds.get("energy", {}).get("neutral_range", [0.02, 0.08])),
            energetic_min=thresholds.get("energy", {}).get("energetic_min", 0.08),
            excited_min=thresholds.get("energy", {}).get("excited_min", 0.15)
        )
        
        spectral = SpectralThresholds(
            calm_centroid_max=thresholds.get("spectral", {}).get("calm_centroid_max", 2000.0),
            bright_centroid_min=thresholds.get("spectral", {}).get("bright_centroid_min", 3000.0),
            rolloff_thresholds=tuple(thresholds.get("spectral", {}).get("rolloff_thresholds", [1500.0, 3000.0, 5000.0]))
        )
        
        temporal = TemporalThresholds(
            calm_zcr_max=thresholds.get("temporal", {}).get("calm_zcr_max", 0.05),
            energetic_zcr_min=thresholds.get("temporal", {}).get("energetic_zcr_min", 0.15)
        )
        
        smoothing_data = config_dict.get("smoothing", {})
        smoothing = SmoothingConfig(
            transition_time=smoothing_data.get("transition_time", 2.0),
            minimum_duration=smoothing_data.get("minimum_duration", 5.0),
            confidence_threshold=smoothing_data.get("confidence_threshold", 0.7)
        )
        
        noise_data = config_dict.get("noise_filtering", {})
        noise_filtering = NoiseFilteringConfig(
            noise_gate_threshold=noise_data.get("noise_gate_threshold", 0.01),
            adaptive_gain=noise_data.get("adaptive_gain", True),
            background_learning_rate=noise_data.get("background_learning_rate", 0.1)
        )
        
        return MoodConfig(
            energy=energy,
            spectral=spectral,
            temporal=temporal,
            smoothing=smoothing,
            noise_filtering=noise_filtering
        )
    
    def update_thresholds(self, **kwargs) -> bool:
        """
        Update specific threshold values and save configuration.
        
        Args:
            **kwargs: Threshold values to update
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            config = self.load_config()
            
            # Update energy thresholds
            if 'calm_max' in kwargs:
                config.energy.calm_max = kwargs['calm_max']
            if 'neutral_range' in kwargs:
                config.energy.neutral_range = tuple(kwargs['neutral_range'])
            if 'energetic_min' in kwargs:
                config.energy.energetic_min = kwargs['energetic_min']
            if 'excited_min' in kwargs:
                config.energy.excited_min = kwargs['excited_min']
            
            # Update spectral thresholds
            if 'calm_centroid_max' in kwargs:
                config.spectral.calm_centroid_max = kwargs['calm_centroid_max']
            if 'bright_centroid_min' in kwargs:
                config.spectral.bright_centroid_min = kwargs['bright_centroid_min']
            if 'rolloff_thresholds' in kwargs:
                config.spectral.rolloff_thresholds = tuple(kwargs['rolloff_thresholds'])
            
            # Update temporal thresholds
            if 'calm_zcr_max' in kwargs:
                config.temporal.calm_zcr_max = kwargs['calm_zcr_max']
            if 'energetic_zcr_min' in kwargs:
                config.temporal.energetic_zcr_min = kwargs['energetic_zcr_min']
            
            # Update smoothing config
            if 'transition_time' in kwargs:
                config.smoothing.transition_time = kwargs['transition_time']
            if 'minimum_duration' in kwargs:
                config.smoothing.minimum_duration = kwargs['minimum_duration']
            if 'confidence_threshold' in kwargs:
                config.smoothing.confidence_threshold = kwargs['confidence_threshold']
            
            # Update noise filtering config
            if 'noise_gate_threshold' in kwargs:
                config.noise_filtering.noise_gate_threshold = kwargs['noise_gate_threshold']
            if 'adaptive_gain' in kwargs:
                config.noise_filtering.adaptive_gain = kwargs['adaptive_gain']
            if 'background_learning_rate' in kwargs:
                config.noise_filtering.background_learning_rate = kwargs['background_learning_rate']
            
            return self.save_config(config)
            
        except Exception as e:
            self.logger.error(f"Error updating thresholds: {e}")
            return False