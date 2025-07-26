#!/usr/bin/env python3
"""
User calibration system for personalized mood detection.
Allows users to record baseline voice characteristics and adjust thresholds accordingly.
"""

import numpy as np
import json
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from enhanced_audio_features import AudioFeatures, EnhancedFeatureExtractor
from mood_config import MoodConfig, ConfigManager
from advanced_mood_detector import AdvancedMoodDetector


@dataclass
class CalibrationData:
    """
    User-specific calibration data containing baseline voice characteristics.
    """
    user_id: str
    timestamp: float
    
    # Baseline feature statistics
    baseline_rms: float
    baseline_rms_std: float
    baseline_spectral_centroid: float
    baseline_spectral_centroid_std: float
    baseline_zero_crossing_rate: float
    baseline_zero_crossing_rate_std: float
    baseline_fundamental_freq: float
    baseline_fundamental_freq_std: float
    
    # Voice range characteristics
    rms_range: Tuple[float, float]  # (min, max)
    spectral_centroid_range: Tuple[float, float]
    zcr_range: Tuple[float, float]
    f0_range: Tuple[float, float]
    
    # Mood-specific calibration factors
    calm_factor: float = 1.0
    neutral_factor: float = 1.0
    energetic_factor: float = 1.0
    excited_factor: float = 1.0
    
    # Calibration quality metrics
    sample_count: int = 0
    confidence_score: float = 0.0
    calibration_duration: float = 0.0


@dataclass
class CalibrationSession:
    """
    Represents an active calibration session.
    """
    user_id: str
    start_time: float
    target_duration: float
    mood_target: Optional[str] = None
    
    # Collected samples
    feature_samples: List[AudioFeatures] = None
    
    def __post_init__(self):
        if self.feature_samples is None:
            self.feature_samples = []


class UserCalibrator:
    """
    Manages user calibration for personalized mood detection.
    
    Provides functionality to:
    - Record baseline voice characteristics
    - Analyze user-specific patterns
    - Adjust detection thresholds
    - Store and load calibration data
    """
    
    def __init__(self, config_manager: Optional[ConfigManager] = None, 
                 feature_extractor: Optional[EnhancedFeatureExtractor] = None):
        """
        Initialize the user calibrator.
        
        Args:
            config_manager: ConfigManager instance for loading/saving configs
            feature_extractor: EnhancedFeatureExtractor for audio analysis
        """
        self.config_manager = config_manager or ConfigManager()
        self.feature_extractor = feature_extractor or EnhancedFeatureExtractor()
        
        # Calibration storage
        self.calibration_dir = Path("calibration_data")
        self.calibration_dir.mkdir(exist_ok=True)
        
        # Active session
        self.current_session: Optional[CalibrationSession] = None
        
        # Calibration parameters
        self.min_calibration_duration = 30.0  # seconds
        self.recommended_calibration_duration = 60.0  # seconds
        self.min_samples_required = 50
        self.confidence_threshold = 0.7
        
        # Voice activity requirements
        self.min_voice_activity_ratio = 0.6  # 60% of samples should be voice
        
    def start_calibration_session(self, user_id: str, duration: float = None, 
                                mood_target: str = None) -> CalibrationSession:
        """
        Start a new calibration session for a user.
        
        Args:
            user_id: Unique identifier for the user
            duration: Target duration in seconds (default: recommended duration)
            mood_target: Specific mood to calibrate for (optional)
            
        Returns:
            CalibrationSession: The active calibration session
        """
        if self.current_session is not None:
            raise RuntimeError("Calibration session already in progress")
        
        if duration is None:
            duration = self.recommended_calibration_duration
        
        if duration < self.min_calibration_duration:
            raise ValueError(f"Calibration duration must be at least {self.min_calibration_duration} seconds")
        
        self.current_session = CalibrationSession(
            user_id=user_id,
            start_time=time.time(),
            target_duration=duration,
            mood_target=mood_target
        )
        
        return self.current_session
    
    def add_audio_sample(self, audio_data: np.ndarray) -> bool:
        """
        Add an audio sample to the current calibration session.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            bool: True if sample was added successfully
        """
        if self.current_session is None:
            raise RuntimeError("No active calibration session")
        
        # Extract features from audio
        features = self.feature_extractor.extract_features(audio_data)
        
        # Only add samples with voice activity and reasonable confidence
        if features.voice_activity and features.confidence > 0.3:
            self.current_session.feature_samples.append(features)
            return True
        
        return False
    
    def get_calibration_progress(self) -> Dict[str, float]:
        """
        Get the progress of the current calibration session.
        
        Returns:
            Dict with progress information
        """
        if self.current_session is None:
            return {"progress": 0.0, "samples": 0, "voice_ratio": 0.0}
        
        elapsed_time = time.time() - self.current_session.start_time
        time_progress = min(elapsed_time / self.current_session.target_duration, 1.0)
        
        sample_count = len(self.current_session.feature_samples)
        sample_progress = min(sample_count / self.min_samples_required, 1.0)
        
        # Calculate voice activity ratio
        total_samples = sample_count
        voice_samples = sum(1 for f in self.current_session.feature_samples if f.voice_activity)
        voice_ratio = voice_samples / max(total_samples, 1)
        
        overall_progress = min(time_progress, sample_progress)
        
        return {
            "progress": overall_progress,
            "time_progress": time_progress,
            "sample_progress": sample_progress,
            "samples": sample_count,
            "voice_ratio": voice_ratio,
            "elapsed_time": elapsed_time,
            "remaining_time": max(0, self.current_session.target_duration - elapsed_time)
        }
    
    def is_calibration_complete(self) -> bool:
        """
        Check if the current calibration session is complete.
        
        Returns:
            bool: True if calibration meets completion criteria
        """
        if self.current_session is None:
            return False
        
        progress = self.get_calibration_progress()
        
        # Check minimum requirements
        time_complete = progress["elapsed_time"] >= self.min_calibration_duration
        sample_complete = progress["samples"] >= self.min_samples_required
        voice_ratio_ok = progress["voice_ratio"] >= self.min_voice_activity_ratio
        
        return time_complete and sample_complete and voice_ratio_ok
    
    def finish_calibration_session(self) -> CalibrationData:
        """
        Complete the current calibration session and generate calibration data.
        
        Returns:
            CalibrationData: Processed calibration data
        """
        if self.current_session is None:
            raise RuntimeError("No active calibration session")
        
        if not self.is_calibration_complete():
            raise RuntimeError("Calibration session does not meet completion criteria")
        
        # Analyze collected samples
        calibration_data = self._analyze_calibration_samples(self.current_session)
        
        # Save calibration data
        self._save_calibration_data(calibration_data)
        
        # Clear current session
        self.current_session = None
        
        return calibration_data
    
    def cancel_calibration_session(self) -> None:
        """Cancel the current calibration session."""
        self.current_session = None
    
    def _analyze_calibration_samples(self, session: CalibrationSession) -> CalibrationData:
        """
        Analyze collected calibration samples to generate calibration data.
        
        Args:
            session: Completed calibration session
            
        Returns:
            CalibrationData: Analyzed calibration data
        """
        features = session.feature_samples
        
        if not features:
            raise ValueError("No valid features collected during calibration")
        
        # Extract feature arrays
        rms_values = np.array([f.rms for f in features])
        centroid_values = np.array([f.spectral_centroid for f in features])
        zcr_values = np.array([f.zero_crossing_rate for f in features])
        f0_values = np.array([f.fundamental_freq for f in features if f.fundamental_freq > 0])
        
        # Calculate baseline statistics
        baseline_rms = float(np.mean(rms_values))
        baseline_rms_std = float(np.std(rms_values))
        
        baseline_centroid = float(np.mean(centroid_values))
        baseline_centroid_std = float(np.std(centroid_values))
        
        baseline_zcr = float(np.mean(zcr_values))
        baseline_zcr_std = float(np.std(zcr_values))
        
        if len(f0_values) > 0:
            baseline_f0 = float(np.mean(f0_values))
            baseline_f0_std = float(np.std(f0_values))
            f0_range = (float(np.min(f0_values)), float(np.max(f0_values)))
        else:
            baseline_f0 = 0.0
            baseline_f0_std = 0.0
            f0_range = (0.0, 0.0)
        
        # Calculate feature ranges
        rms_range = (float(np.min(rms_values)), float(np.max(rms_values)))
        centroid_range = (float(np.min(centroid_values)), float(np.max(centroid_values)))
        zcr_range = (float(np.min(zcr_values)), float(np.max(zcr_values)))
        
        # Calculate calibration quality
        confidence_scores = [f.confidence for f in features]
        avg_confidence = float(np.mean(confidence_scores))
        
        # Calculate mood-specific factors based on feature distributions
        mood_factors = self._calculate_mood_factors(
            baseline_rms, baseline_centroid, baseline_zcr, baseline_f0
        )
        
        calibration_duration = time.time() - session.start_time
        
        return CalibrationData(
            user_id=session.user_id,
            timestamp=time.time(),
            
            # Baseline statistics
            baseline_rms=baseline_rms,
            baseline_rms_std=baseline_rms_std,
            baseline_spectral_centroid=baseline_centroid,
            baseline_spectral_centroid_std=baseline_centroid_std,
            baseline_zero_crossing_rate=baseline_zcr,
            baseline_zero_crossing_rate_std=baseline_zcr_std,
            baseline_fundamental_freq=baseline_f0,
            baseline_fundamental_freq_std=baseline_f0_std,
            
            # Feature ranges
            rms_range=rms_range,
            spectral_centroid_range=centroid_range,
            zcr_range=zcr_range,
            f0_range=f0_range,
            
            # Mood factors
            calm_factor=mood_factors['calm'],
            neutral_factor=mood_factors['neutral'],
            energetic_factor=mood_factors['energetic'],
            excited_factor=mood_factors['excited'],
            
            # Quality metrics
            sample_count=len(features),
            confidence_score=avg_confidence,
            calibration_duration=calibration_duration
        )
    
    def _calculate_mood_factors(self, baseline_rms: float, baseline_centroid: float,
                              baseline_zcr: float, baseline_f0: float) -> Dict[str, float]:
        """
        Calculate mood-specific calibration factors based on user's baseline.
        
        Args:
            baseline_rms: User's baseline RMS energy
            baseline_centroid: User's baseline spectral centroid
            baseline_zcr: User's baseline zero-crossing rate
            baseline_f0: User's baseline fundamental frequency
            
        Returns:
            Dict with mood-specific calibration factors
        """
        # Load default configuration for reference
        default_config = self.config_manager.load_config()
        
        # Calculate scaling factors based on user's baseline vs. default expectations
        
        # RMS scaling (energy-based moods)
        default_neutral_rms = np.mean(default_config.energy.neutral_range)
        rms_scale = baseline_rms / max(default_neutral_rms, 0.01)
        
        # Spectral scaling (brightness-based moods)
        default_centroid = 2000.0  # Assumed default centroid
        centroid_scale = baseline_centroid / max(default_centroid, 100.0)
        
        # ZCR scaling (temporal activity)
        default_zcr = 0.1  # Assumed default ZCR
        zcr_scale = baseline_zcr / max(default_zcr, 0.01)
        
        # Calculate mood-specific factors
        factors = {
            'calm': min(2.0, max(0.5, 1.0 / rms_scale)),  # Inverse relationship with energy
            'neutral': 1.0,  # Neutral is the baseline
            'energetic': min(2.0, max(0.5, rms_scale * zcr_scale)),  # Direct relationship
            'excited': min(2.0, max(0.5, rms_scale * centroid_scale))  # Energy + brightness
        }
        
        return factors
    
    def apply_calibration(self, user_id: str, config: Optional[MoodConfig] = None) -> MoodConfig:
        """
        Apply user calibration to a mood configuration.
        
        Args:
            user_id: User identifier
            config: Base configuration to modify (default: load from file)
            
        Returns:
            MoodConfig: Calibrated configuration
        """
        # Load calibration data
        calibration_data = self.load_calibration_data(user_id)
        if calibration_data is None:
            raise ValueError(f"No calibration data found for user: {user_id}")
        
        # Load base configuration
        if config is None:
            config = self.config_manager.load_config()
        
        # Create a copy to modify
        calibrated_config = MoodConfig(
            energy=config.energy,
            spectral=config.spectral,
            temporal=config.temporal,
            smoothing=config.smoothing
        )
        
        # Apply calibration factors to energy thresholds
        calibrated_config.energy.calm_max *= calibration_data.calm_factor
        
        # Scale neutral range
        neutral_min, neutral_max = calibrated_config.energy.neutral_range
        calibrated_config.energy.neutral_range = (
            neutral_min * calibration_data.neutral_factor,
            neutral_max * calibration_data.neutral_factor
        )
        
        calibrated_config.energy.energetic_min *= calibration_data.energetic_factor
        calibrated_config.energy.excited_min *= calibration_data.excited_factor
        
        # Apply calibration to spectral thresholds
        centroid_scale = calibration_data.baseline_spectral_centroid / 2000.0
        calibrated_config.spectral.calm_centroid_max *= centroid_scale
        calibrated_config.spectral.bright_centroid_min *= centroid_scale
        
        # Apply calibration to temporal thresholds
        zcr_scale = calibration_data.baseline_zero_crossing_rate / 0.1
        calibrated_config.temporal.calm_zcr_max *= zcr_scale
        calibrated_config.temporal.energetic_zcr_min *= zcr_scale
        
        return calibrated_config
    
    def _save_calibration_data(self, calibration_data: CalibrationData) -> None:
        """
        Save calibration data to file.
        
        Args:
            calibration_data: Calibration data to save
        """
        filename = f"{calibration_data.user_id}_calibration.json"
        filepath = self.calibration_dir / filename
        
        # Convert to dictionary and handle tuples
        data_dict = asdict(calibration_data)
        
        with open(filepath, 'w') as f:
            json.dump(data_dict, f, indent=2)
    
    def load_calibration_data(self, user_id: str) -> Optional[CalibrationData]:
        """
        Load calibration data for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            CalibrationData or None if not found
        """
        filename = f"{user_id}_calibration.json"
        filepath = self.calibration_dir / filename
        
        if not filepath.exists():
            return None
        
        try:
            with open(filepath, 'r') as f:
                data_dict = json.load(f)
            
            return CalibrationData(**data_dict)
        except Exception as e:
            print(f"Error loading calibration data: {e}")
            return None
    
    def list_calibrated_users(self) -> List[str]:
        """
        Get list of users with calibration data.
        
        Returns:
            List of user IDs
        """
        users = []
        for filepath in self.calibration_dir.glob("*_calibration.json"):
            user_id = filepath.stem.replace("_calibration", "")
            users.append(user_id)
        return users
    
    def delete_calibration_data(self, user_id: str) -> bool:
        """
        Delete calibration data for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            bool: True if deleted successfully
        """
        filename = f"{user_id}_calibration.json"
        filepath = self.calibration_dir / filename
        
        if filepath.exists():
            filepath.unlink()
            return True
        return False
    
    def get_calibration_summary(self, user_id: str) -> Optional[Dict]:
        """
        Get a summary of calibration data for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dict with calibration summary or None if not found
        """
        calibration_data = self.load_calibration_data(user_id)
        if calibration_data is None:
            return None
        
        return {
            "user_id": calibration_data.user_id,
            "calibration_date": time.ctime(calibration_data.timestamp),
            "sample_count": calibration_data.sample_count,
            "confidence_score": calibration_data.confidence_score,
            "duration": calibration_data.calibration_duration,
            "baseline_rms": calibration_data.baseline_rms,
            "baseline_centroid": calibration_data.baseline_spectral_centroid,
            "mood_factors": {
                "calm": calibration_data.calm_factor,
                "neutral": calibration_data.neutral_factor,
                "energetic": calibration_data.energetic_factor,
                "excited": calibration_data.excited_factor
            }
        }


# Convenience functions for easy integration
def calibrate_user(user_id: str, audio_samples: List[np.ndarray], 
                  feature_extractor: Optional[EnhancedFeatureExtractor] = None) -> CalibrationData:
    """
    Convenience function to calibrate a user with provided audio samples.
    
    Args:
        user_id: User identifier
        audio_samples: List of audio data arrays
        feature_extractor: Optional feature extractor
        
    Returns:
        CalibrationData: Generated calibration data
    """
    calibrator = UserCalibrator(feature_extractor=feature_extractor)
    
    # Calculate duration based on samples (minimum 30 seconds)
    estimated_duration = max(30.0, len(audio_samples) * 0.5)
    
    # Start calibration session
    session = calibrator.start_calibration_session(user_id, duration=estimated_duration)
    
    # Add all samples
    for audio_data in audio_samples:
        calibrator.add_audio_sample(audio_data)
    
    # Force completion if we have enough samples
    if len(session.feature_samples) >= calibrator.min_samples_required:
        calibrator.current_session.start_time = time.time() - calibrator.min_calibration_duration
    
    # If we still don't meet criteria, adjust requirements temporarily
    if not calibrator.is_calibration_complete():
        original_min_samples = calibrator.min_samples_required
        original_voice_ratio = calibrator.min_voice_activity_ratio
        
        # Temporarily lower requirements for convenience function
        calibrator.min_samples_required = min(len(session.feature_samples), original_min_samples)
        calibrator.min_voice_activity_ratio = 0.3  # Lower voice activity requirement
        
        # Ensure minimum time has passed
        if time.time() - session.start_time < calibrator.min_calibration_duration:
            session.start_time = time.time() - calibrator.min_calibration_duration
        
        # Restore original requirements after use
        try:
            result = calibrator.finish_calibration_session()
            calibrator.min_samples_required = original_min_samples
            calibrator.min_voice_activity_ratio = original_voice_ratio
            return result
        except:
            calibrator.min_samples_required = original_min_samples
            calibrator.min_voice_activity_ratio = original_voice_ratio
            raise
    
    # Finish calibration
    return calibrator.finish_calibration_session()


def get_calibrated_detector(user_id: str, config_manager: Optional[ConfigManager] = None) -> AdvancedMoodDetector:
    """
    Convenience function to get a mood detector calibrated for a specific user.
    
    Args:
        user_id: User identifier
        config_manager: Optional config manager
        
    Returns:
        AdvancedMoodDetector: Calibrated mood detector
    """
    calibrator = UserCalibrator(config_manager=config_manager)
    calibrated_config = calibrator.apply_calibration(user_id)
    return AdvancedMoodDetector(calibrated_config)