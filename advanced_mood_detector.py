#!/usr/bin/env python3
"""
Advanced mood detection algorithm with multi-dimensional scoring and confidence calculation.
Replaces the simple detect_mood function with sophisticated feature analysis.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import time
from enhanced_audio_features import AudioFeatures
from mood_config import MoodConfig, ConfigManager
from performance_monitor import get_global_monitor, get_global_scaler
from error_handling import get_global_error_manager, ErrorSeverity


@dataclass
class MoodResult:
    """
    Result of mood detection with confidence scores and debugging information.
    """
    mood: str  # 'calm', 'neutral', 'energetic', 'excited'
    confidence: float  # 0.0 to 1.0
    features_used: AudioFeatures
    transition_recommended: bool
    debug_scores: Dict[str, float]


class AdvancedMoodDetector:
    """
    Advanced mood detection using multi-dimensional feature analysis.
    
    This class replaces the simple detect_mood function with a sophisticated
    algorithm that combines multiple audio features with configurable weights
    and provides confidence scoring.
    """
    
    def __init__(self, config: Optional[MoodConfig] = None):
        """
        Initialize the advanced mood detector.
        
        Args:
            config: MoodConfig object, or None to use default configuration
        """
        # Error handling
        self.error_manager = get_global_error_manager()
        self.consecutive_failures = 0
        self.max_consecutive_failures = 5
        self.fallback_mode = False
        
        # Load configuration with error handling
        try:
            if config is None:
                config_manager = ConfigManager()
                config = config_manager.load_config()
            self.config = config
        except Exception as e:
            self.error_manager.handle_error(
                'mood_config', 'config_load_error', e, ErrorSeverity.MEDIUM
            )
            # Use minimal default config
            self.config = self._get_minimal_config()
        
        self.feature_history: List[AudioFeatures] = []
        self.mood_history: List[str] = []
        self.max_history_length = 10
        
        # Performance monitoring integration with error handling
        try:
            self.performance_monitor = get_global_monitor()
            self.performance_scaler = get_global_scaler()
            
            # Register for performance configuration updates
            self.performance_scaler.register_config_callback(self._on_performance_config_change)
        except Exception as e:
            self.error_manager.handle_error(
                'performance_monitor', 'init_error', e, ErrorSeverity.LOW
            )
            self.performance_monitor = None
            self.performance_scaler = None
        
        # Mood detection weights for different feature categories
        self.feature_weights = {
            'energy': 0.35,      # Energy features are most important
            'spectral': 0.25,    # Spectral features for voice quality
            'temporal': 0.25,    # Temporal features for speech patterns
            'pitch': 0.15        # Pitch features for emotional content
        }
        
        # Mood-specific feature expectations
        self.mood_profiles = {
            'calm': {
                'energy_preference': 'low',
                'spectral_preference': 'low_centroid',
                'temporal_preference': 'low_zcr',
                'pitch_preference': 'stable'
            },
            'neutral': {
                'energy_preference': 'medium',
                'spectral_preference': 'medium_centroid',
                'temporal_preference': 'medium_zcr',
                'pitch_preference': 'moderate'
            },
            'energetic': {
                'energy_preference': 'high',
                'spectral_preference': 'high_centroid',
                'temporal_preference': 'high_zcr',
                'pitch_preference': 'variable'
            },
            'excited': {
                'energy_preference': 'very_high',
                'spectral_preference': 'bright',
                'temporal_preference': 'very_high_zcr',
                'pitch_preference': 'unstable'
            }
        }
    
    def detect_mood(self, features: AudioFeatures) -> MoodResult:
        """
        Detect mood from audio features using multi-dimensional scoring with error handling.
        
        Args:
            features: AudioFeatures object containing extracted audio features
            
        Returns:
            MoodResult: Detection result with mood, confidence, and debug info
        """
        try:
            # Validate input features
            if features is None:
                return self._create_fallback_mood_result()
            
            # Check if we should use fallback mode
            if self.fallback_mode:
                return self._detect_mood_simple(features)
            
            # Add features to history for consistency checking
            self._update_feature_history(features)
            
            # Calculate individual feature scores for each mood with error handling
            mood_scores = {}
            debug_scores = {}
            
            for mood in ['calm', 'neutral', 'energetic', 'excited']:
                try:
                    energy_score = self._calculate_energy_score(features, mood)
                    spectral_score = self._calculate_spectral_score(features, mood)
                    temporal_score = self._calculate_temporal_score(features, mood)
                    pitch_score = self._calculate_pitch_score(features, mood)
                    
                    # Weighted combination of feature scores
                    total_score = (
                        self.feature_weights['energy'] * energy_score +
                        self.feature_weights['spectral'] * spectral_score +
                        self.feature_weights['temporal'] * temporal_score +
                        self.feature_weights['pitch'] * pitch_score
                    )
                    
                    mood_scores[mood] = total_score
                    debug_scores[f'{mood}_energy'] = energy_score
                    debug_scores[f'{mood}_spectral'] = spectral_score
                    debug_scores[f'{mood}_temporal'] = temporal_score
                    debug_scores[f'{mood}_pitch'] = pitch_score
                    debug_scores[f'{mood}_total'] = total_score
                    
                except Exception as e:
                    self.error_manager.handle_error(
                        'mood_scoring', f'{mood}_score_error', e, ErrorSeverity.MEDIUM
                    )
                    # Use fallback score for this mood
                    mood_scores[mood] = 0.5
                    debug_scores[f'{mood}_total'] = 0.5
            
            # Determine best mood
            if mood_scores:
                best_mood = max(mood_scores, key=mood_scores.get)
                best_score = mood_scores[best_mood]
            else:
                # Fallback if no scores calculated
                best_mood = 'neutral'
                best_score = 0.5
            
            # Calculate confidence based on feature consistency and score separation
            try:
                confidence = self._calculate_confidence(features, mood_scores, best_mood)
            except Exception as e:
                self.error_manager.handle_error(
                    'confidence_calculation', 'confidence_error', e, ErrorSeverity.LOW
                )
                confidence = 0.5
            
            # Determine if transition is recommended
            try:
                transition_recommended = self._should_recommend_transition(
                    best_mood, confidence, features
                )
            except Exception as e:
                self.error_manager.handle_error(
                    'transition_logic', 'transition_error', e, ErrorSeverity.LOW
                )
                transition_recommended = True
            
            # Update mood history
            self._update_mood_history(best_mood)
            
            # Reset consecutive failures on success
            self.consecutive_failures = 0
            
            return MoodResult(
                mood=best_mood,
                confidence=confidence,
                features_used=features,
                transition_recommended=transition_recommended,
                debug_scores=debug_scores
            )
            
        except Exception as e:
            # Critical error in mood detection
            self.consecutive_failures += 1
            self.error_manager.handle_error(
                'mood_detection', 'critical_error', e, ErrorSeverity.HIGH
            )
            
            # If too many consecutive failures, enable fallback mode
            if self.consecutive_failures >= self.max_consecutive_failures:
                self.fallback_mode = True
                self.error_manager.handle_error(
                    'mood_detection', 'fallback_mode_activated',
                    Exception(f"Activating fallback mode after {self.consecutive_failures} failures"),
                    ErrorSeverity.CRITICAL
                )
            
            return self._create_fallback_mood_result(features)
    
    def _calculate_energy_score(self, features: AudioFeatures, mood: str) -> float:
        """Calculate energy-based score for a specific mood."""
        rms = features.rms
        peak_energy = features.peak_energy
        energy_variance = features.energy_variance
        
        if mood == 'calm':
            # Calm prefers low, stable energy
            if rms <= self.config.energy.calm_max:
                rms_score = 1.0 - (rms / self.config.energy.calm_max) * 0.3  # Give bonus for very low energy
            else:
                rms_score = max(0.0, 1.0 - (rms - self.config.energy.calm_max) / self.config.energy.calm_max)
            variance_score = 1.0 - min(energy_variance / 0.001, 1.0)  # Low variance
            return 0.7 * rms_score + 0.3 * variance_score
            
        elif mood == 'neutral':
            # Neutral prefers moderate energy in the neutral range
            neutral_min, neutral_max = self.config.energy.neutral_range
            if neutral_min <= rms <= neutral_max:
                rms_score = 1.0
            else:
                # Penalize being outside neutral range
                if rms < neutral_min:
                    rms_score = rms / neutral_min
                else:
                    rms_score = max(0.0, 1.0 - (rms - neutral_max) / neutral_max)
            
            # Moderate variance is acceptable
            variance_score = 1.0 - min(energy_variance / 0.005, 1.0)
            return 0.8 * rms_score + 0.2 * variance_score
            
        elif mood == 'energetic':
            # Energetic prefers high energy above energetic threshold
            if rms >= self.config.energy.energetic_min:
                rms_score = min((rms - self.config.energy.energetic_min) / 0.05, 1.0)
            else:
                rms_score = 0.0
            
            # Higher variance is good for energetic
            variance_score = min(energy_variance / 0.01, 1.0)
            return 0.6 * rms_score + 0.4 * variance_score
            
        elif mood == 'excited':
            # Excited prefers very high energy and high variance
            if rms >= self.config.energy.excited_min:
                rms_score = 1.0 + min((rms - self.config.energy.excited_min) / 0.05, 1.0)  # Bonus for very high energy
            else:
                rms_score = max(0.0, (rms - self.config.energy.energetic_min) / 
                               (self.config.energy.excited_min - self.config.energy.energetic_min))
            
            # Very high variance indicates excitement
            variance_score = min(energy_variance / 0.015, 1.0)  # More sensitive to variance
            peak_score = min(peak_energy / 0.3, 1.0)  # More sensitive to peaks
            return 0.4 * rms_score + 0.4 * variance_score + 0.2 * peak_score
        
        return 0.0
    
    def _calculate_spectral_score(self, features: AudioFeatures, mood: str) -> float:
        """Calculate spectral-based score for a specific mood."""
        centroid = features.spectral_centroid
        rolloff = features.spectral_rolloff
        flux = features.spectral_flux
        mfccs = features.mfccs
        
        if mood == 'calm':
            # Calm prefers low spectral centroid (darker sound)
            if centroid <= self.config.spectral.calm_centroid_max:
                centroid_score = 1.0 - (centroid / self.config.spectral.calm_centroid_max) * 0.5
            else:
                centroid_score = max(0.0, 1.0 - (centroid - self.config.spectral.calm_centroid_max) / 1000)
            flux_score = 1.0 - min(flux / 1000, 1.0)  # Low spectral change
            return 0.7 * centroid_score + 0.3 * flux_score
            
        elif mood == 'neutral':
            # Neutral prefers moderate spectral characteristics
            if centroid <= self.config.spectral.calm_centroid_max:
                centroid_score = centroid / self.config.spectral.calm_centroid_max
            elif centroid >= self.config.spectral.bright_centroid_min:
                centroid_score = max(0.0, 1.0 - (centroid - self.config.spectral.bright_centroid_min) / 2000)
            else:
                centroid_score = 1.0  # In the sweet spot
            
            flux_score = 1.0 - min(flux / 5000, 1.0)  # Moderate spectral change
            return 0.8 * centroid_score + 0.2 * flux_score
            
        elif mood == 'energetic':
            # Energetic prefers higher spectral centroid
            if centroid >= self.config.spectral.bright_centroid_min:
                centroid_score = min((centroid - self.config.spectral.bright_centroid_min) / 2000, 1.0)
            else:
                centroid_score = max(0.0, (centroid - self.config.spectral.calm_centroid_max) / 
                                   (self.config.spectral.bright_centroid_min - self.config.spectral.calm_centroid_max))
            
            flux_score = min(flux / 10000, 1.0)  # Higher spectral change is good
            return 0.6 * centroid_score + 0.4 * flux_score
            
        elif mood == 'excited':
            # Excited prefers very bright spectral characteristics
            if centroid >= 4000:  # Very bright threshold
                centroid_score = 1.0 + min((centroid - 4000) / 2000, 0.5)  # Bonus for very bright
            else:
                centroid_score = max(0.0, centroid / 4000)
            
            flux_score = min(flux / 15000, 1.0)  # High spectral change
            
            # MFCC variance indicates vocal excitement
            mfcc_variance = np.var(mfccs) if len(mfccs) > 1 else 0.0
            mfcc_score = min(mfcc_variance / 5, 1.0)  # More sensitive to MFCC variance
            
            return 0.5 * centroid_score + 0.3 * flux_score + 0.2 * mfcc_score
        
        return 0.0
    
    def _calculate_temporal_score(self, features: AudioFeatures, mood: str) -> float:
        """Calculate temporal-based score for a specific mood."""
        zcr = features.zero_crossing_rate
        tempo = features.tempo
        voice_activity = features.voice_activity
        
        # Voice activity bonus
        va_bonus = 1.0 if voice_activity else 0.5
        
        if mood == 'calm':
            # Calm prefers low zero-crossing rate
            if zcr <= self.config.temporal.calm_zcr_max:
                zcr_score = 1.0 - (zcr / self.config.temporal.calm_zcr_max) * 0.3  # Bonus for very low ZCR
            else:
                zcr_score = max(0.0, 1.0 - (zcr - self.config.temporal.calm_zcr_max) / self.config.temporal.calm_zcr_max)
            tempo_score = 1.0 - min(tempo / 60, 1.0)  # Slow tempo
            return va_bonus * (0.8 * zcr_score + 0.2 * tempo_score)
            
        elif mood == 'neutral':
            # Neutral prefers moderate ZCR
            if zcr <= self.config.temporal.calm_zcr_max:
                zcr_score = zcr / self.config.temporal.calm_zcr_max
            elif zcr >= self.config.temporal.energetic_zcr_min:
                zcr_score = max(0.0, 1.0 - (zcr - self.config.temporal.energetic_zcr_min) / 0.1)
            else:
                zcr_score = 1.0
            
            tempo_score = 1.0 - abs(tempo - 120) / 120  # Moderate tempo around 120 BPM
            tempo_score = max(0.0, tempo_score)
            return va_bonus * (0.7 * zcr_score + 0.3 * tempo_score)
            
        elif mood == 'energetic':
            # Energetic prefers higher ZCR
            if zcr >= self.config.temporal.energetic_zcr_min:
                zcr_score = min((zcr - self.config.temporal.energetic_zcr_min) / 0.1, 1.0)
            else:
                zcr_score = max(0.0, (zcr - self.config.temporal.calm_zcr_max) / 
                               (self.config.temporal.energetic_zcr_min - self.config.temporal.calm_zcr_max))
            
            tempo_score = min(tempo / 150, 1.0)  # Higher tempo
            return va_bonus * (0.7 * zcr_score + 0.3 * tempo_score)
            
        elif mood == 'excited':
            # Excited prefers very high ZCR
            zcr_score = min(zcr / 0.3, 1.0)  # Very high ZCR
            tempo_score = min(tempo / 180, 1.0)  # Very high tempo
            return va_bonus * (0.6 * zcr_score + 0.4 * tempo_score)
        
        return 0.0
    
    def _calculate_pitch_score(self, features: AudioFeatures, mood: str) -> float:
        """Calculate pitch-based score for a specific mood."""
        f0 = features.fundamental_freq
        pitch_stability = features.pitch_stability
        pitch_range = features.pitch_range
        
        # If no pitch detected, return neutral score
        if f0 <= 0:
            return 0.5
        
        if mood == 'calm':
            # Calm prefers stable, moderate pitch
            stability_score = pitch_stability  # Higher stability is better
            range_score = 1.0 - min(pitch_range / 100, 1.0)  # Lower range is better
            f0_score = 1.0 - abs(f0 - 150) / 150  # Prefer around 150 Hz
            f0_score = max(0.0, f0_score)
            return 0.5 * stability_score + 0.3 * range_score + 0.2 * f0_score
            
        elif mood == 'neutral':
            # Neutral accepts moderate pitch characteristics
            stability_score = min(pitch_stability + 0.3, 1.0)  # Moderate stability
            range_score = 1.0 - abs(pitch_range - 80) / 80  # Moderate range
            range_score = max(0.0, range_score)
            return 0.6 * stability_score + 0.4 * range_score
            
        elif mood == 'energetic':
            # Energetic prefers more variable pitch
            stability_score = 1.0 - pitch_stability  # Lower stability is better
            range_score = min(pitch_range / 150, 1.0)  # Higher range is better
            return 0.4 * stability_score + 0.6 * range_score
            
        elif mood == 'excited':
            # Excited prefers very unstable, wide-ranging pitch
            stability_score = 1.0 - pitch_stability  # Very unstable
            range_score = min(pitch_range / 200, 1.0)  # Very wide range
            f0_variance_score = min(abs(f0 - 200) / 200, 1.0)  # Extreme pitch values
            return 0.3 * stability_score + 0.5 * range_score + 0.2 * f0_variance_score
        
        return 0.0
    
    def _calculate_confidence(self, features: AudioFeatures, mood_scores: Dict[str, float], 
                            best_mood: str) -> float:
        """
        Calculate confidence based on feature consistency and score separation.
        """
        confidence_factors = []
        
        # 1. Feature quality confidence (from AudioFeatures)
        feature_confidence = features.confidence
        confidence_factors.append(feature_confidence)
        
        # 2. Score separation confidence
        sorted_scores = sorted(mood_scores.values(), reverse=True)
        if len(sorted_scores) >= 2:
            score_separation = sorted_scores[0] - sorted_scores[1]
            separation_confidence = min(score_separation / 0.3, 1.0)  # Normalize
            confidence_factors.append(separation_confidence)
        
        # 3. Historical consistency confidence
        if len(self.mood_history) >= 3:
            recent_moods = self.mood_history[-3:]
            consistency = sum(1 for mood in recent_moods if mood == best_mood) / len(recent_moods)
            confidence_factors.append(consistency)
        
        # 4. Feature consistency confidence
        if len(self.feature_history) >= 2:
            current_features = features
            prev_features = self.feature_history[-2]
            
            # Check consistency of key features
            rms_consistency = 1.0 - abs(current_features.rms - prev_features.rms) / max(current_features.rms, 0.01)
            centroid_consistency = 1.0 - abs(current_features.spectral_centroid - prev_features.spectral_centroid) / max(current_features.spectral_centroid, 100)
            
            feature_consistency = (rms_consistency + centroid_consistency) / 2
            feature_consistency = max(0.0, min(feature_consistency, 1.0))
            confidence_factors.append(feature_consistency)
        
        # 5. Voice activity confidence
        if features.voice_activity:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.3)
        
        # Calculate weighted average confidence
        final_confidence = np.mean(confidence_factors)
        return max(0.0, min(final_confidence, 1.0))
    
    def _should_recommend_transition(self, mood: str, confidence: float, 
                                   features: AudioFeatures) -> bool:
        """
        Determine if a mood transition should be recommended.
        """
        # Always recommend transition for first detection
        if len(self.mood_history) == 0:
            return True
        
        # Only recommend transition if confidence is above threshold
        if confidence < self.config.smoothing.confidence_threshold:
            return False
        
        # Check if this is a different mood from recent history
        if len(self.mood_history) > 0:
            last_mood = self.mood_history[-1]
            if mood != last_mood:
                return True
        
        # If it's the same mood but with very high confidence, still recommend
        # to maintain the current state
        if confidence > 0.9:
            return True
        
        return False
    
    def _update_feature_history(self, features: AudioFeatures) -> None:
        """Update the feature history buffer."""
        self.feature_history.append(features)
        if len(self.feature_history) > self.max_history_length:
            self.feature_history.pop(0)
    
    def _update_mood_history(self, mood: str) -> None:
        """Update the mood history buffer."""
        self.mood_history.append(mood)
        if len(self.mood_history) > self.max_history_length:
            self.mood_history.pop(0)
    
    def calibrate_for_user(self, calibration_data: List[AudioFeatures]) -> None:
        """
        Calibrate the detector for a specific user's voice characteristics.
        
        Args:
            calibration_data: List of AudioFeatures from user's baseline speech
        """
        if not calibration_data:
            return
        
        # Calculate baseline statistics
        rms_values = [f.rms for f in calibration_data]
        centroid_values = [f.spectral_centroid for f in calibration_data]
        zcr_values = [f.zero_crossing_rate for f in calibration_data]
        
        # Adjust thresholds based on user's baseline
        baseline_rms = np.mean(rms_values)
        baseline_centroid = np.mean(centroid_values)
        baseline_zcr = np.mean(zcr_values)
        
        # Scale thresholds relative to user's baseline
        scale_factor = baseline_rms / 0.05  # Assume 0.05 as reference
        
        self.config.energy.calm_max *= scale_factor
        self.config.energy.neutral_range = (
            self.config.energy.neutral_range[0] * scale_factor,
            self.config.energy.neutral_range[1] * scale_factor
        )
        self.config.energy.energetic_min *= scale_factor
        self.config.energy.excited_min *= scale_factor
        
        # Adjust spectral thresholds
        centroid_scale = baseline_centroid / 2000  # Assume 2000 Hz as reference
        self.config.spectral.calm_centroid_max *= centroid_scale
        self.config.spectral.bright_centroid_min *= centroid_scale
        
        # Adjust temporal thresholds
        zcr_scale = baseline_zcr / 0.1  # Assume 0.1 as reference
        self.config.temporal.calm_zcr_max *= zcr_scale
        self.config.temporal.energetic_zcr_min *= zcr_scale
    
    def update_thresholds(self, config: MoodConfig) -> None:
        """
        Update the detector's configuration thresholds.
        
        Args:
            config: New MoodConfig to use
        """
        self.config = config
    
    def _get_minimal_config(self) -> MoodConfig:
        """Get minimal configuration for fallback mode."""
        from mood_config import MoodConfig, EnergyThresholds, SpectralThresholds, TemporalThresholds, SmoothingConfig, NoiseFilteringConfig
        
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
    
    def _create_fallback_mood_result(self, features: Optional[AudioFeatures] = None) -> MoodResult:
        """Create fallback mood result when detection fails."""
        if features is None:
            # Create minimal features
            from enhanced_audio_features import AudioFeatures
            features = AudioFeatures(
                rms=0.05, peak_energy=0.1, energy_variance=0.001,
                spectral_centroid=2000.0, spectral_rolloff=4000.0, spectral_flux=100.0, mfccs=[],
                zero_crossing_rate=0.1, tempo=120.0, voice_activity=True,
                fundamental_freq=150.0, pitch_stability=0.5, pitch_range=50.0,
                timestamp=time.time(), confidence=0.3
            )
        
        return MoodResult(
            mood='neutral',  # Default to neutral
            confidence=0.3,  # Low confidence for fallback
            features_used=features,
            transition_recommended=True,
            debug_scores={'fallback': 1.0}
        )
    
    def _detect_mood_simple(self, features: AudioFeatures) -> MoodResult:
        """Simple mood detection for fallback mode."""
        try:
            # Use basic thresholds similar to original led.py
            rms = features.rms
            zcr = features.zero_crossing_rate
            centroid = features.spectral_centroid
            
            # Simple threshold-based detection
            if rms < 0.02 and zcr < 0.05:
                mood = "calm"
            elif rms > 0.08 and zcr > 0.15:
                mood = "energetic"
            elif centroid > 3000:
                mood = "excited"  # Map "bright" to "excited"
            else:
                mood = "neutral"
            
            return MoodResult(
                mood=mood,
                confidence=0.6,  # Medium confidence for simple detection
                features_used=features,
                transition_recommended=True,
                debug_scores={'simple_detection': 1.0}
            )
            
        except Exception as e:
            self.error_manager.handle_error(
                'simple_mood_detection', 'fallback_error', e, ErrorSeverity.HIGH
            )
            return self._create_fallback_mood_result(features)
    
    def reset_fallback_mode(self) -> None:
        """Reset fallback mode and try enhanced detection again."""
        self.fallback_mode = False
        self.consecutive_failures = 0
    
    def is_in_fallback_mode(self) -> bool:
        """Check if detector is in fallback mode."""
        return self.fallback_mode


# Convenience function to replace the simple detect_mood function
def detect_mood_advanced(features: AudioFeatures, detector: Optional[AdvancedMoodDetector] = None) -> MoodResult:
    """
    Advanced mood detection function that replaces the simple detect_mood.
    
    Args:
        features: AudioFeatures object
        detector: Optional AdvancedMoodDetector instance (creates new one if None)
        
    Returns:
        MoodResult: Advanced mood detection result
    """
    if detector is None:
        detector = AdvancedMoodDetector()
    
    return detector.detect_mood(features)


# Legacy compatibility function
def detect_mood_simple(rms: float, zcr: float, centroid: float) -> str:
    """
    Simple mood detection for backward compatibility with existing code.
    
    Args:
        rms: RMS energy
        zcr: Zero-crossing rate
        centroid: Spectral centroid
        
    Returns:
        str: Detected mood ('calm', 'neutral', 'energetic', 'bright')
    """
    # Use the same thresholds as in the original led.py
    TH_RMS_LOW, TH_RMS_HIGH = 0.02, 0.08
    TH_ZCR_LOW, TH_ZCR_HIGH = 0.05, 0.15
    TH_CENTROID_HIGH = 3000  # Hz

    if rms < TH_RMS_LOW and zcr < TH_ZCR_LOW:
        return "calm"
    elif rms > TH_RMS_HIGH and zcr > TH_ZCR_HIGH:
        return "energetic"
    elif centroid > TH_CENTROID_HIGH:
        return "bright"
    else:
        return "neutral"