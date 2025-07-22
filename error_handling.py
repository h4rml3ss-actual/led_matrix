#!/usr/bin/env python3
"""
Graceful error handling and fallback system for enhanced mood detection.
Implements error recovery, fallback mechanisms, and automatic degradation for low-resource situations.
"""

import logging
import time
import traceback
import threading
from dataclasses import dataclass
from typing import Optional, Dict, Any, Callable, List
from enum import Enum
import numpy as np


class ErrorSeverity(Enum):
    """Error severity levels for different types of failures."""
    LOW = "low"           # Minor issues, continue with degraded performance
    MEDIUM = "medium"     # Significant issues, fallback to simpler processing
    HIGH = "high"         # Critical issues, fallback to basic functionality
    CRITICAL = "critical" # System-threatening issues, emergency fallback


@dataclass
class ErrorEvent:
    """Represents an error event with context and recovery information."""
    timestamp: float
    component: str
    error_type: str
    severity: ErrorSeverity
    message: str
    exception: Optional[Exception]
    recovery_action: Optional[str]
    fallback_used: bool


class ErrorRecoveryManager:
    """
    Manages error recovery and fallback mechanisms for the mood detection system.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.error_history: List[ErrorEvent] = []
        self.max_error_history = 100
        
        # Error counters for different components
        self.error_counts: Dict[str, int] = {}
        self.consecutive_errors: Dict[str, int] = {}
        self.last_error_time: Dict[str, float] = {}
        
        # Recovery strategies
        self.recovery_strategies: Dict[str, Callable] = {}
        self.fallback_handlers: Dict[str, Callable] = {}
        
        # System state
        self.degraded_components: set = set()
        self.disabled_components: set = set()
        self.emergency_mode = False
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Configure logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup error logging configuration."""
        # Create error log handler if not exists
        error_handler = logging.FileHandler('mood_detection_errors.log')
        error_handler.setLevel(logging.ERROR)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        error_handler.setFormatter(formatter)
        self.logger.addHandler(error_handler)
        self.logger.setLevel(logging.INFO)
    
    def register_recovery_strategy(self, component: str, strategy: Callable) -> None:
        """
        Register a recovery strategy for a specific component.
        
        Args:
            component: Component name
            strategy: Recovery function that takes (error_event) and returns bool
        """
        self.recovery_strategies[component] = strategy
    
    def register_fallback_handler(self, component: str, handler: Callable) -> None:
        """
        Register a fallback handler for a specific component.
        
        Args:
            component: Component name
            handler: Fallback function that provides alternative functionality
        """
        self.fallback_handlers[component] = handler
    
    def handle_error(self, component: str, error_type: str, exception: Exception, 
                    severity: ErrorSeverity = ErrorSeverity.MEDIUM) -> bool:
        """
        Handle an error event and attempt recovery.
        
        Args:
            component: Component where error occurred
            error_type: Type of error
            exception: The exception that occurred
            severity: Severity level of the error
            
        Returns:
            True if error was handled successfully, False if fallback needed
        """
        current_time = time.time()
        
        with self._lock:
            # Create error event
            error_event = ErrorEvent(
                timestamp=current_time,
                component=component,
                error_type=error_type,
                severity=severity,
                message=str(exception),
                exception=exception,
                recovery_action=None,
                fallback_used=False
            )
            
            # Update error statistics
            self.error_counts[component] = self.error_counts.get(component, 0) + 1
            self.consecutive_errors[component] = self.consecutive_errors.get(component, 0) + 1
            self.last_error_time[component] = current_time
            
            # Add to history
            self.error_history.append(error_event)
            if len(self.error_history) > self.max_error_history:
                self.error_history.pop(0)
            
            # Log the error
            self.logger.error(f"Error in {component} ({error_type}): {exception}")
            if severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
                self.logger.error(f"Stack trace: {traceback.format_exc()}")
            
            # Determine recovery action based on severity and history
            recovery_successful = self._attempt_recovery(error_event)
            
            if not recovery_successful:
                # Recovery failed, use fallback
                self._activate_fallback(error_event)
                error_event.fallback_used = True
            
            # Check if component should be degraded or disabled
            self._evaluate_component_health(component)
            
            return recovery_successful
    
    def _attempt_recovery(self, error_event: ErrorEvent) -> bool:
        """
        Attempt to recover from an error using registered strategies.
        
        Args:
            error_event: The error event to recover from
            
        Returns:
            True if recovery was successful
        """
        component = error_event.component
        
        # Check if we have a recovery strategy for this component
        if component not in self.recovery_strategies:
            return False
        
        # Don't attempt recovery if too many consecutive errors
        if self.consecutive_errors.get(component, 0) > 5:
            self.logger.warning(f"Too many consecutive errors in {component}, skipping recovery")
            return False
        
        # Don't attempt recovery if component is disabled
        if component in self.disabled_components:
            return False
        
        try:
            recovery_strategy = self.recovery_strategies[component]
            success = recovery_strategy(error_event)
            
            if success:
                error_event.recovery_action = "strategy_successful"
                self.consecutive_errors[component] = 0  # Reset consecutive error count
                self.logger.info(f"Successfully recovered {component} from {error_event.error_type}")
                return True
            else:
                error_event.recovery_action = "strategy_failed"
                return False
                
        except Exception as recovery_exception:
            self.logger.error(f"Recovery strategy failed for {component}: {recovery_exception}")
            error_event.recovery_action = f"strategy_exception: {recovery_exception}"
            return False
    
    def _activate_fallback(self, error_event: ErrorEvent) -> None:
        """
        Activate fallback mechanism for a component.
        
        Args:
            error_event: The error event that triggered fallback
        """
        component = error_event.component
        
        if component in self.fallback_handlers:
            try:
                fallback_handler = self.fallback_handlers[component]
                fallback_handler(error_event)
                self.logger.info(f"Activated fallback for {component}")
                
                # Mark component as degraded
                self.degraded_components.add(component)
                
            except Exception as fallback_exception:
                self.logger.error(f"Fallback failed for {component}: {fallback_exception}")
                # If fallback fails, disable the component
                self.disabled_components.add(component)
        else:
            self.logger.warning(f"No fallback handler registered for {component}")
            self.degraded_components.add(component)
    
    def _evaluate_component_health(self, component: str) -> None:
        """
        Evaluate component health and take appropriate action.
        
        Args:
            component: Component to evaluate
        """
        consecutive_errors = self.consecutive_errors.get(component, 0)
        total_errors = self.error_counts.get(component, 0)
        
        # Disable component if too many consecutive errors
        if consecutive_errors >= 10:
            self.disabled_components.add(component)
            self.logger.error(f"Disabling {component} due to {consecutive_errors} consecutive errors")
        
        # Degrade component if moderate error rate
        elif consecutive_errors >= 5:
            self.degraded_components.add(component)
            self.logger.warning(f"Degrading {component} due to {consecutive_errors} consecutive errors")
        
        # Check for emergency mode conditions
        if len(self.disabled_components) >= 3 or total_errors >= 50:
            self._activate_emergency_mode()
    
    def _activate_emergency_mode(self) -> None:
        """Activate emergency mode with minimal functionality."""
        if not self.emergency_mode:
            self.emergency_mode = True
            self.logger.critical("Activating emergency mode - switching to minimal functionality")
            
            # Disable all non-essential components
            non_essential = ['pitch_analysis', 'mfcc_calculation', 'spectral_analysis', 'noise_filtering']
            for component in non_essential:
                self.disabled_components.add(component)
    
    def reset_component_errors(self, component: str) -> None:
        """
        Reset error counters for a component (e.g., after successful operation).
        
        Args:
            component: Component to reset
        """
        with self._lock:
            self.consecutive_errors[component] = 0
            if component in self.degraded_components:
                self.degraded_components.remove(component)
                self.logger.info(f"Component {component} recovered from degraded state")
    
    def is_component_available(self, component: str) -> bool:
        """
        Check if a component is available for use.
        
        Args:
            component: Component to check
            
        Returns:
            True if component is available (not disabled)
        """
        return component not in self.disabled_components
    
    def is_component_degraded(self, component: str) -> bool:
        """
        Check if a component is in degraded mode.
        
        Args:
            component: Component to check
            
        Returns:
            True if component is degraded
        """
        return component in self.degraded_components
    
    def get_error_summary(self) -> Dict[str, Any]:
        """
        Get a summary of error statistics and system health.
        
        Returns:
            Dictionary with error statistics and system status
        """
        with self._lock:
            recent_errors = [e for e in self.error_history if time.time() - e.timestamp < 300]  # Last 5 minutes
            
            return {
                'emergency_mode': self.emergency_mode,
                'disabled_components': list(self.disabled_components),
                'degraded_components': list(self.degraded_components),
                'total_errors': sum(self.error_counts.values()),
                'recent_errors': len(recent_errors),
                'error_counts_by_component': self.error_counts.copy(),
                'consecutive_errors': self.consecutive_errors.copy(),
                'recent_error_types': [e.error_type for e in recent_errors[-10:]]
            }


class SafeAudioProcessor:
    """
    Safe wrapper for audio processing with error handling and fallbacks.
    """
    
    def __init__(self, error_manager: ErrorRecoveryManager):
        self.error_manager = error_manager
        self.logger = logging.getLogger(__name__)
        
        # Fallback processing functions
        self._register_fallbacks()
    
    def _register_fallbacks(self):
        """Register fallback handlers for audio processing components."""
        self.error_manager.register_fallback_handler(
            'feature_extraction', self._fallback_feature_extraction
        )
        self.error_manager.register_fallback_handler(
            'mood_detection', self._fallback_mood_detection
        )
        self.error_manager.register_fallback_handler(
            'noise_filtering', self._fallback_noise_filtering
        )
        self.error_manager.register_fallback_handler(
            'microphone_input', self._fallback_microphone_input
        )
    
    def safe_extract_features(self, audio_data: np.ndarray, feature_extractor, **kwargs):
        """
        Safely extract audio features with error handling.
        
        Args:
            audio_data: Input audio data
            feature_extractor: Feature extraction object
            **kwargs: Additional arguments for feature extraction
            
        Returns:
            AudioFeatures object or fallback features
        """
        try:
            features = feature_extractor.extract_features(audio_data, **kwargs)
            # Reset error count on success
            self.error_manager.reset_component_errors('feature_extraction')
            return features
            
        except Exception as e:
            self.error_manager.handle_error(
                'feature_extraction', 
                type(e).__name__, 
                e, 
                ErrorSeverity.MEDIUM
            )
            # Return fallback features
            return self._fallback_feature_extraction(None)
    
    def safe_detect_mood(self, features, mood_detector, **kwargs):
        """
        Safely detect mood with error handling.
        
        Args:
            features: Audio features
            mood_detector: Mood detection object
            **kwargs: Additional arguments for mood detection
            
        Returns:
            MoodResult object or fallback mood
        """
        try:
            mood_result = mood_detector.detect_mood(features, **kwargs)
            self.error_manager.reset_component_errors('mood_detection')
            return mood_result
            
        except Exception as e:
            self.error_manager.handle_error(
                'mood_detection',
                type(e).__name__,
                e,
                ErrorSeverity.MEDIUM
            )
            return self._fallback_mood_detection(None)
    
    def safe_filter_audio(self, audio_data: np.ndarray, noise_filter, **kwargs):
        """
        Safely filter audio with error handling.
        
        Args:
            audio_data: Input audio data
            noise_filter: Noise filter object
            **kwargs: Additional arguments for filtering
            
        Returns:
            Tuple of (filtered_audio, vad_result) or fallback
        """
        try:
            filtered_audio, vad_result = noise_filter.filter_audio(audio_data, **kwargs)
            self.error_manager.reset_component_errors('noise_filtering')
            return filtered_audio, vad_result
            
        except Exception as e:
            self.error_manager.handle_error(
                'noise_filtering',
                type(e).__name__,
                e,
                ErrorSeverity.LOW  # Noise filtering failure is less critical
            )
            return self._fallback_noise_filtering(audio_data)
    
    def _fallback_feature_extraction(self, error_event):
        """Fallback feature extraction using basic methods."""
        try:
            from enhanced_audio_features import AudioFeatures
            
            # Return minimal features for compatibility
            return AudioFeatures(
                # Energy features
                rms=0.05,
                peak_energy=0.1,
                energy_variance=0.001,
                
                # Spectral features
                spectral_centroid=2000.0,
                spectral_rolloff=4000.0,
                spectral_flux=100.0,
                mfccs=[],  # Empty MFCC for fallback
                
                # Temporal features
                zero_crossing_rate=0.1,
                tempo=120.0,
                voice_activity=True,
                
                # Pitch features
                fundamental_freq=150.0,
                pitch_stability=0.5,
                pitch_range=50.0,
                
                # Metadata
                timestamp=time.time(),
                confidence=0.3  # Low confidence for fallback
            )
        except ImportError:
            # Create a simple fallback object if AudioFeatures can't be imported
            class FallbackFeatures:
                def __init__(self):
                    self.rms = 0.05
                    self.peak_energy = 0.1
                    self.energy_variance = 0.001
                    self.spectral_centroid = 2000.0
                    self.spectral_rolloff = 4000.0
                    self.spectral_flux = 100.0
                    self.mfccs = []
                    self.zero_crossing_rate = 0.1
                    self.tempo = 120.0
                    self.voice_activity = True
                    self.fundamental_freq = 150.0
                    self.pitch_stability = 0.5
                    self.pitch_range = 50.0
                    self.timestamp = time.time()
                    self.confidence = 0.3
            
            return FallbackFeatures()
    
    def _fallback_mood_detection(self, error_event):
        """Fallback mood detection using simple logic."""
        try:
            from advanced_mood_detector import MoodResult
            
            # Create fallback features
            fallback_features = self._fallback_feature_extraction(None)
            
            # Simple mood detection based on basic heuristics
            mood = "neutral"  # Default to neutral
            confidence = 0.5
            
            return MoodResult(
                mood=mood,
                confidence=confidence,
                features_used=fallback_features,
                transition_recommended=True,
                debug_scores={'fallback': 1.0}
            )
        except ImportError:
            # Create a simple fallback object if MoodResult can't be imported
            class FallbackMoodResult:
                def __init__(self):
                    self.mood = "neutral"
                    self.confidence = 0.5
                    self.features_used = self._fallback_feature_extraction(None)
                    self.transition_recommended = True
                    self.debug_scores = {'fallback': 1.0}
            
            return FallbackMoodResult()
    
    def _fallback_noise_filtering(self, audio_data: np.ndarray):
        """Fallback noise filtering using simple methods."""
        try:
            from noise_filter import VoiceActivityResult
            
            # Simple voice activity detection based on energy
            rms = np.sqrt(np.mean(audio_data**2))
            is_voice = rms > 0.01
            
            # Simple high-pass filter to remove low-frequency noise
            if len(audio_data) > 1:
                filtered_audio = audio_data - np.mean(audio_data)  # Remove DC offset
            else:
                filtered_audio = audio_data
            
            vad_result = VoiceActivityResult(
                is_voice=is_voice,
                confidence=0.6,
                energy_ratio=rms / 0.01,
                spectral_ratio=1.0,
                zero_crossing_ratio=1.0
            )
            
            return filtered_audio, vad_result
        except ImportError:
            # Create a simple fallback object if VoiceActivityResult can't be imported
            class FallbackVADResult:
                def __init__(self, is_voice, confidence):
                    self.is_voice = is_voice
                    self.confidence = confidence
                    self.energy_ratio = 1.0
                    self.spectral_ratio = 1.0
                    self.zero_crossing_ratio = 1.0
            
            # Simple voice activity detection based on energy
            rms = np.sqrt(np.mean(audio_data**2))
            is_voice = rms > 0.01
            
            # Simple high-pass filter to remove low-frequency noise
            if len(audio_data) > 1:
                filtered_audio = audio_data - np.mean(audio_data)  # Remove DC offset
            else:
                filtered_audio = audio_data
            
            vad_result = FallbackVADResult(is_voice=is_voice, confidence=0.6)
            
            return filtered_audio, vad_result
    
    def _fallback_microphone_input(self, error_event):
        """Handle microphone disconnection by generating silence."""
        self.logger.warning("Microphone disconnected, generating silence")
        # Return silence - the system will continue with neutral mood
        return np.zeros(1024, dtype=np.float32)


class MicrophoneErrorHandler:
    """
    Specialized error handler for microphone and audio input issues.
    """
    
    def __init__(self, error_manager: ErrorRecoveryManager):
        self.error_manager = error_manager
        self.logger = logging.getLogger(__name__)
        self.reconnection_attempts = 0
        self.max_reconnection_attempts = 5
        self.last_reconnection_attempt = 0
        self.reconnection_delay = 2.0  # seconds
        
        # Register recovery strategy
        self.error_manager.register_recovery_strategy(
            'microphone_input', self._recover_microphone
        )
    
    def _recover_microphone(self, error_event: ErrorEvent) -> bool:
        """
        Attempt to recover from microphone errors.
        
        Args:
            error_event: The microphone error event
            
        Returns:
            True if recovery was successful
        """
        current_time = time.time()
        
        # Don't attempt recovery too frequently
        if current_time - self.last_reconnection_attempt < self.reconnection_delay:
            return False
        
        # Don't exceed maximum attempts
        if self.reconnection_attempts >= self.max_reconnection_attempts:
            self.logger.error("Maximum microphone reconnection attempts exceeded")
            return False
        
        self.last_reconnection_attempt = current_time
        self.reconnection_attempts += 1
        
        try:
            # Attempt to reinitialize audio input
            import sounddevice as sd
            
            # Test if we can query devices
            devices = sd.query_devices()
            if not devices:
                return False
            
            # Try to create a test stream
            with sd.InputStream(channels=1, samplerate=44100, blocksize=1024):
                pass
            
            # If we get here, microphone is working
            self.reconnection_attempts = 0
            self.logger.info("Microphone reconnection successful")
            return True
            
        except Exception as e:
            self.logger.warning(f"Microphone reconnection attempt {self.reconnection_attempts} failed: {e}")
            return False
    
    def handle_microphone_disconnection(self) -> bool:
        """
        Handle microphone disconnection event.
        
        Returns:
            True if handled successfully
        """
        return self.error_manager.handle_error(
            'microphone_input',
            'disconnection',
            Exception("Microphone disconnected"),
            ErrorSeverity.HIGH
        )


# Global error manager instance
_global_error_manager: Optional[ErrorRecoveryManager] = None

def get_global_error_manager() -> ErrorRecoveryManager:
    """Get or create the global error recovery manager."""
    global _global_error_manager
    if _global_error_manager is None:
        _global_error_manager = ErrorRecoveryManager()
    return _global_error_manager

def safe_execute(func: Callable, component: str, *args, **kwargs):
    """
    Safely execute a function with error handling.
    
    Args:
        func: Function to execute
        component: Component name for error tracking
        *args: Function arguments
        **kwargs: Function keyword arguments
        
    Returns:
        Function result or None if error occurred
    """
    error_manager = get_global_error_manager()
    
    try:
        result = func(*args, **kwargs)
        error_manager.reset_component_errors(component)
        return result
    except Exception as e:
        error_manager.handle_error(
            component,
            type(e).__name__,
            e,
            ErrorSeverity.MEDIUM
        )
        return None