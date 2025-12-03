#!/usr/bin/env python3
"""
Debugging and diagnostic tools for enhanced mood detection system.
Provides debug logging, real-time visualization, diagnostic analysis, and configuration validation.
"""

import json
import time
import logging
import os
import sys
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import numpy as np
from datetime import datetime
import threading
import queue
from scipy.io import wavfile

# Import enhanced components
try:
    from enhanced_audio_features import AudioFeatures, EnhancedFeatureExtractor
    from advanced_mood_detector import MoodResult, AdvancedMoodDetector
    from mood_config import MoodConfig, ConfigManager, ConfigValidationError
    from mood_transition_smoother import MoodTransitionSmoother
    from performance_monitor import get_global_monitor
    ENHANCED_AVAILABLE = True
except ImportError as e:
    print(f"Enhanced components not available: {e}")
    ENHANCED_AVAILABLE = False


@dataclass
class DebugLogEntry:
    """Single debug log entry for mood detection."""
    timestamp: float
    mood: str
    confidence: float
    features: Dict[str, Any]
    debug_scores: Dict[str, float]
    transition_recommended: bool
    performance_metrics: Optional[Dict[str, Any]] = None


class MoodDebugLogger:
    """
    Debug logger for mood detection decisions and confidence scores.
    Logs detailed information about each mood detection cycle.
    """
    
    def __init__(self, log_file: str = "mood_debug.log", max_entries: int = 1000):
        self.log_file = log_file
        self.max_entries = max_entries
        self.entries: List[DebugLogEntry] = []
        self.lock = threading.Lock()
        
        # Setup file logging
        self.logger = logging.getLogger('mood_debug')
        self.logger.setLevel(logging.DEBUG)
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler for important messages
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info("Mood debug logger initialized")
    
    def log_mood_detection(self, mood_result: 'MoodResult', performance_metrics: Optional[Dict] = None):
        """
        Log a mood detection result with detailed information.
        
        Args:
            mood_result: MoodResult from advanced mood detector
            performance_metrics: Optional performance metrics
        """
        with self.lock:
            # Create debug entry
            entry = DebugLogEntry(
                timestamp=time.time(),
                mood=mood_result.mood,
                confidence=mood_result.confidence,
                features=self._features_to_dict(mood_result.features_used),
                debug_scores=mood_result.debug_scores.copy(),
                transition_recommended=mood_result.transition_recommended,
                performance_metrics=performance_metrics
            )
            
            # Add to memory buffer
            self.entries.append(entry)
            if len(self.entries) > self.max_entries:
                self.entries.pop(0)
            
            # Log to file
            self.logger.debug(
                f"MOOD_DETECTION: {mood_result.mood} "
                f"(conf={mood_result.confidence:.3f}, "
                f"transition={mood_result.transition_recommended}) "
                f"RMS={mood_result.features_used.rms:.4f} "
                f"ZCR={mood_result.features_used.zero_crossing_rate:.4f} "
                f"Centroid={mood_result.features_used.spectral_centroid:.1f}Hz"
            )
            
            # Log detailed scores at debug level
            score_details = ", ".join([
                f"{k}={v:.3f}" for k, v in mood_result.debug_scores.items()
                if k.endswith('_total')
            ])
            self.logger.debug(f"MOOD_SCORES: {score_details}")
            
            # Log confidence factors
            if mood_result.confidence < 0.5:
                self.logger.warning(f"LOW_CONFIDENCE: {mood_result.confidence:.3f} for {mood_result.mood}")
            
            # Log performance issues
            if performance_metrics and performance_metrics.get('total_duration', 0) > 0.02:  # 20ms threshold
                self.logger.warning(
                    f"PERFORMANCE_SLOW: {performance_metrics['total_duration']*1000:.1f}ms processing time"
                )
    
    def _features_to_dict(self, features: 'AudioFeatures') -> Dict[str, Any]:
        """Convert AudioFeatures to dictionary for logging."""
        return {
            'rms': features.rms,
            'peak_energy': features.peak_energy,
            'energy_variance': features.energy_variance,
            'spectral_centroid': features.spectral_centroid,
            'spectral_rolloff': features.spectral_rolloff,
            'spectral_flux': features.spectral_flux,
            'mfccs': features.mfccs,
            'zero_crossing_rate': features.zero_crossing_rate,
            'tempo': features.tempo,
            'voice_activity': features.voice_activity,
            'fundamental_freq': features.fundamental_freq,
            'pitch_stability': features.pitch_stability,
            'pitch_range': features.pitch_range,
            'timestamp': features.timestamp,
            'confidence': features.confidence
        }
    
    def get_recent_entries(self, count: int = 10) -> List[DebugLogEntry]:
        """Get recent debug entries."""
        with self.lock:
            return self.entries[-count:] if self.entries else []
    
    def export_debug_data(self, filepath: str) -> bool:
        """
        Export debug data to JSON file.
        
        Args:
            filepath: Path to export file
            
        Returns:
            bool: True if successful
        """
        try:
            with self.lock:
                data = {
                    'export_time': datetime.now().isoformat(),
                    'total_entries': len(self.entries),
                    'entries': [asdict(entry) for entry in self.entries]
                }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.logger.info(f"Debug data exported to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export debug data: {e}")
            return False
    
    def analyze_mood_patterns(self) -> Dict[str, Any]:
        """
        Analyze mood detection patterns from logged data.
        
        Returns:
            Dictionary with analysis results
        """
        if not self.entries:
            return {'error': 'No debug data available'}
        
        with self.lock:
            entries = self.entries.copy()
        
        # Mood distribution
        mood_counts = {}
        confidence_by_mood = {}
        transition_counts = 0
        
        for entry in entries:
            mood = entry.mood
            mood_counts[mood] = mood_counts.get(mood, 0) + 1
            
            if mood not in confidence_by_mood:
                confidence_by_mood[mood] = []
            confidence_by_mood[mood].append(entry.confidence)
            
            if entry.transition_recommended:
                transition_counts += 1
        
        # Calculate statistics
        total_entries = len(entries)
        mood_percentages = {
            mood: (count / total_entries) * 100
            for mood, count in mood_counts.items()
        }
        
        avg_confidence_by_mood = {
            mood: np.mean(confidences)
            for mood, confidences in confidence_by_mood.items()
        }
        
        # Recent performance
        recent_entries = entries[-50:] if len(entries) >= 50 else entries
        recent_avg_confidence = np.mean([e.confidence for e in recent_entries])
        
        return {
            'total_detections': total_entries,
            'mood_distribution': mood_percentages,
            'average_confidence_by_mood': avg_confidence_by_mood,
            'transition_rate': (transition_counts / total_entries) * 100,
            'recent_average_confidence': recent_avg_confidence,
            'time_span_minutes': (entries[-1].timestamp - entries[0].timestamp) / 60 if entries else 0
        }


class RealTimeVisualizer:
    """
    Real-time feature visualization for tuning mood detection parameters.
    Displays live audio features and mood detection results.
    """
    
    def __init__(self, update_interval: float = 0.1):
        self.update_interval = update_interval
        self.running = False
        self.data_queue = queue.Queue(maxsize=100)
        self.display_thread = None
        
        # Feature history for visualization
        self.feature_history = {
            'rms': [],
            'zcr': [],
            'centroid': [],
            'confidence': [],
            'mood': []
        }
        self.max_history = 50
    
    def start(self):
        """Start real-time visualization."""
        if self.running:
            return
        
        self.running = True
        self.display_thread = threading.Thread(target=self._display_loop, daemon=True)
        self.display_thread.start()
        print("Real-time visualizer started")
    
    def stop(self):
        """Stop real-time visualization."""
        self.running = False
        if self.display_thread:
            self.display_thread.join(timeout=1.0)
        print("Real-time visualizer stopped")
    
    def update(self, mood_result: 'MoodResult'):
        """
        Update visualization with new mood detection result.
        
        Args:
            mood_result: Latest MoodResult from detector
        """
        if not self.running:
            return
        
        try:
            self.data_queue.put_nowait({
                'timestamp': time.time(),
                'mood': mood_result.mood,
                'confidence': mood_result.confidence,
                'features': mood_result.features_used,
                'debug_scores': mood_result.debug_scores
            })
        except queue.Full:
            # Drop oldest data if queue is full
            try:
                self.data_queue.get_nowait()
                self.data_queue.put_nowait({
                    'timestamp': time.time(),
                    'mood': mood_result.mood,
                    'confidence': mood_result.confidence,
                    'features': mood_result.features_used,
                    'debug_scores': mood_result.debug_scores
                })
            except queue.Empty:
                pass
    
    def _display_loop(self):
        """Main display loop for real-time visualization."""
        while self.running:
            try:
                # Get latest data
                data = self.data_queue.get(timeout=self.update_interval)
                
                # Update feature history
                self._update_history(data)
                
                # Display current status
                self._display_current_status(data)
                
                # Display feature trends every 10 updates
                if len(self.feature_history['rms']) % 10 == 0:
                    self._display_feature_trends()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Visualization error: {e}")
    
    def _update_history(self, data: Dict):
        """Update feature history buffers."""
        features = data['features']
        
        self.feature_history['rms'].append(features.rms)
        self.feature_history['zcr'].append(features.zero_crossing_rate)
        self.feature_history['centroid'].append(features.spectral_centroid)
        self.feature_history['confidence'].append(data['confidence'])
        self.feature_history['mood'].append(data['mood'])
        
        # Trim history to max length
        for key in self.feature_history:
            if len(self.feature_history[key]) > self.max_history:
                self.feature_history[key].pop(0)
    
    def _display_current_status(self, data: Dict):
        """Display current mood detection status."""
        features = data['features']
        
        # Create status line
        status = (
            f"[{datetime.now().strftime('%H:%M:%S')}] "
            f"Mood: {data['mood'].upper():>10} | "
            f"Conf: {data['confidence']:5.3f} | "
            f"RMS: {features.rms:6.4f} | "
            f"ZCR: {features.zero_crossing_rate:6.4f} | "
            f"Cent: {features.spectral_centroid:6.0f}Hz | "
            f"Voice: {'YES' if features.voice_activity else 'NO'}"
        )
        
        # Print with color coding based on confidence
        if data['confidence'] > 0.8:
            print(f"\033[92m{status}\033[0m")  # Green for high confidence
        elif data['confidence'] > 0.5:
            print(f"\033[93m{status}\033[0m")  # Yellow for medium confidence
        else:
            print(f"\033[91m{status}\033[0m")  # Red for low confidence
    
    def _display_feature_trends(self):
        """Display feature trends and statistics."""
        if len(self.feature_history['rms']) < 5:
            return
        
        print("\n" + "="*80)
        print("FEATURE TRENDS (last 50 samples)")
        print("="*80)
        
        # RMS trend
        rms_values = self.feature_history['rms'][-10:]
        rms_trend = "↑" if rms_values[-1] > rms_values[0] else "↓"
        print(f"RMS Energy:     {np.mean(rms_values):6.4f} ± {np.std(rms_values):6.4f} {rms_trend}")
        
        # ZCR trend
        zcr_values = self.feature_history['zcr'][-10:]
        zcr_trend = "↑" if zcr_values[-1] > zcr_values[0] else "↓"
        print(f"Zero Cross:     {np.mean(zcr_values):6.4f} ± {np.std(zcr_values):6.4f} {zcr_trend}")
        
        # Centroid trend
        cent_values = self.feature_history['centroid'][-10:]
        cent_trend = "↑" if cent_values[-1] > cent_values[0] else "↓"
        print(f"Spectral Cent:  {np.mean(cent_values):6.0f} ± {np.std(cent_values):6.0f}Hz {cent_trend}")
        
        # Confidence trend
        conf_values = self.feature_history['confidence'][-10:]
        conf_trend = "↑" if conf_values[-1] > conf_values[0] else "↓"
        print(f"Confidence:     {np.mean(conf_values):6.3f} ± {np.std(conf_values):6.3f} {conf_trend}")
        
        # Mood distribution
        recent_moods = self.feature_history['mood'][-20:]
        mood_counts = {}
        for mood in recent_moods:
            mood_counts[mood] = mood_counts.get(mood, 0) + 1
        
        print(f"Recent Moods:   ", end="")
        for mood, count in sorted(mood_counts.items()):
            percentage = (count / len(recent_moods)) * 100
            print(f"{mood}:{percentage:4.1f}% ", end="")
        print()
        print("="*80 + "\n")
    
    def get_feature_statistics(self) -> Dict[str, Any]:
        """Get current feature statistics."""
        if not self.feature_history['rms']:
            return {}
        
        stats = {}
        for feature, values in self.feature_history.items():
            if feature == 'mood':
                continue
            if values:
                stats[feature] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'current': float(values[-1]) if values else 0.0
                }
        
        return stats


class DiagnosticAnalyzer:
    """
    Diagnostic mode that outputs detailed analysis of mood detection system.
    Provides comprehensive analysis of system performance and recommendations.
    """
    
    def __init__(self):
        self.config_manager = ConfigManager() if ENHANCED_AVAILABLE else None
        self.feature_extractor = None
        self.mood_detector = None
        
        if ENHANCED_AVAILABLE:
            try:
                self.feature_extractor = EnhancedFeatureExtractor()
                self.mood_detector = AdvancedMoodDetector()
            except Exception as e:
                print(f"Failed to initialize diagnostic components: {e}")
    
    def run_full_diagnostic(self, audio_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Run comprehensive diagnostic analysis.
        
        Args:
            audio_file: Optional audio file to analyze
            
        Returns:
            Dictionary with diagnostic results
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'system_info': self._get_system_info(),
            'configuration_analysis': self._analyze_configuration(),
            'component_status': self._check_component_status(),
            'performance_analysis': self._analyze_performance(),
            'recommendations': []
        }
        
        if audio_file and os.path.exists(audio_file):
            results['audio_analysis'] = self._analyze_audio_file(audio_file)
        
        # Generate recommendations based on analysis
        results['recommendations'] = self._generate_recommendations(results)
        
        return results
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        return {
            'enhanced_available': ENHANCED_AVAILABLE,
            'python_version': sys.version,
            'platform': sys.platform,
            'working_directory': os.getcwd(),
            'config_file_exists': os.path.exists('mood_config.json'),
            'calibration_data_dir': os.path.exists('calibration_data')
        }
    
    def _analyze_configuration(self) -> Dict[str, Any]:
        """Analyze current configuration."""
        if not self.config_manager:
            return {'error': 'Configuration manager not available'}
        
        try:
            config = self.config_manager.load_config()
            
            analysis = {
                'config_valid': True,
                'energy_thresholds': {
                    'calm_max': config.energy.calm_max,
                    'neutral_range': config.energy.neutral_range,
                    'energetic_min': config.energy.energetic_min,
                    'excited_min': config.energy.excited_min
                },
                'spectral_thresholds': {
                    'calm_centroid_max': config.spectral.calm_centroid_max,
                    'bright_centroid_min': config.spectral.bright_centroid_min
                },
                'temporal_thresholds': {
                    'calm_zcr_max': config.temporal.calm_zcr_max,
                    'energetic_zcr_min': config.temporal.energetic_zcr_min
                },
                'smoothing_config': {
                    'transition_time': config.smoothing.transition_time,
                    'minimum_duration': config.smoothing.minimum_duration,
                    'confidence_threshold': config.smoothing.confidence_threshold
                }
            }
            
            # Validate configuration
            try:
                self.config_manager.validate_config(config)
                analysis['validation_errors'] = []
            except ConfigValidationError as e:
                analysis['config_valid'] = False
                analysis['validation_errors'] = [str(e)]
            
            return analysis
            
        except Exception as e:
            return {
                'error': f'Configuration analysis failed: {e}',
                'config_valid': False
            }
    
    def _check_component_status(self) -> Dict[str, Any]:
        """Check status of all system components."""
        status = {
            'enhanced_feature_extractor': False,
            'advanced_mood_detector': False,
            'mood_transition_smoother': False,
            'noise_filter': False,
            'performance_monitor': False,
            'user_calibration': False
        }
        
        if not ENHANCED_AVAILABLE:
            return status
        
        # Check each component
        try:
            from enhanced_audio_features import EnhancedFeatureExtractor
            status['enhanced_feature_extractor'] = True
        except ImportError:
            pass
        
        try:
            from advanced_mood_detector import AdvancedMoodDetector
            status['advanced_mood_detector'] = True
        except ImportError:
            pass
        
        try:
            from mood_transition_smoother import MoodTransitionSmoother
            status['mood_transition_smoother'] = True
        except ImportError:
            pass
        
        try:
            from noise_filter import NoiseFilter
            status['noise_filter'] = True
        except ImportError:
            pass
        
        try:
            from performance_monitor import get_global_monitor
            status['performance_monitor'] = True
        except ImportError:
            pass
        
        try:
            from user_calibration import get_calibrated_detector
            status['user_calibration'] = True
        except ImportError:
            pass
        
        return status
    
    def _analyze_performance(self) -> Dict[str, Any]:
        """Analyze system performance."""
        if not ENHANCED_AVAILABLE:
            return {'error': 'Performance analysis requires enhanced components'}
        
        try:
            monitor = get_global_monitor()
            summary = monitor.get_performance_summary()
            
            return {
                'performance_monitoring_active': True,
                'current_performance_level': summary.get('current_performance_level', 'unknown'),
                'average_processing_time_ms': summary.get('average_processing_time_ms', 0),
                'total_cycles': summary.get('total_cycles', 0),
                'audio_underruns': summary.get('audio_underruns', 0),
                'performance_scaling_active': summary.get('performance_scaling_active', False)
            }
            
        except Exception as e:
            return {
                'error': f'Performance analysis failed: {e}',
                'performance_monitoring_active': False
            }
    
    def _analyze_audio_file(self, audio_file: str) -> Dict[str, Any]:
        """Analyze audio file for diagnostic purposes."""
        if not os.path.exists(audio_file):
            return {
                'file_path': audio_file,
                'error': 'Audio file not found',
                'message': 'Please provide a valid path to an audio file.'
            }

        if not ENHANCED_AVAILABLE:
            return {
                'file_path': audio_file,
                'error': 'Enhanced analysis components are unavailable',
                'message': 'Install enhanced dependencies to enable file diagnostics.'
            }

        try:
            samplerate, data = wavfile.read(audio_file)
        except Exception as e:
            return {
                'file_path': audio_file,
                'error': f'Failed to load audio file: {e}',
                'message': 'Unsupported or corrupt audio file.'
            }

        try:
            audio = np.asarray(data, dtype=np.float32)
            messages: List[str] = []

            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
                messages.append('Input audio was multi-channel and has been downmixed to mono.')

            if audio.dtype.kind in {'i', 'u'}:
                max_val = np.iinfo(data.dtype).max or 1
                audio = audio / float(max_val)
            elif audio.dtype.kind == 'f':
                audio = np.clip(audio, -1.0, 1.0)
            else:
                return {
                    'file_path': audio_file,
                    'error': f'Unsupported audio dtype: {audio.dtype}',
                    'message': 'Provide PCM integer or float audio formats.'
                }

            if len(audio) == 0:
                return {
                    'file_path': audio_file,
                    'error': 'Audio file is empty',
                    'message': 'Provide an audio file with samples for analysis.'
                }

            # Initialize components if necessary
            if self.feature_extractor is None or getattr(self.feature_extractor, 'samplerate', None) != samplerate:
                self.feature_extractor = EnhancedFeatureExtractor(samplerate=samplerate)

            if self.mood_detector is None:
                self.mood_detector = AdvancedMoodDetector()

            frame_size = getattr(self.feature_extractor, 'frame_size', 1024)
            features_list: List[AudioFeatures] = []
            mood_results: List[MoodResult] = []

            for start in range(0, len(audio), frame_size):
                block = audio[start:start + frame_size]
                if len(block) == 0:
                    continue

                timestamp = start / float(samplerate)
                features = self.feature_extractor.extract_features(block, timestamp=timestamp)
                features_list.append(features)

                try:
                    mood_results.append(self.mood_detector.detect_mood(features))
                except Exception as e:
                    messages.append(f"Mood detection failed for frame at {timestamp:.3f}s: {e}")

            if not features_list:
                return {
                    'file_path': audio_file,
                    'error': 'No audio frames could be processed',
                    'message': 'Ensure the file contains valid PCM audio data.'
                }

            feature_stats = self._summarize_features(features_list)
            mood_summary = self._summarize_moods(mood_results)

            return {
                'file_path': audio_file,
                'samplerate': samplerate,
                'duration_seconds': len(audio) / float(samplerate),
                'frames_analyzed': len(features_list),
                'feature_stats': feature_stats,
                'mood_summary': mood_summary,
                'messages': messages
            }
        except Exception as e:
            return {
                'file_path': audio_file,
                'error': f'Unexpected audio analysis failure: {e}',
                'message': 'Try using a different audio file or check diagnostic logs.'
            }

    def _summarize_features(self, features_list: List['AudioFeatures']) -> Dict[str, Any]:
        """Compute summary statistics for extracted features."""
        def stats(values: List[float]) -> Dict[str, float]:
            arr = np.asarray(values, dtype=np.float32)
            return {
                'min': float(np.min(arr)),
                'max': float(np.max(arr)),
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr))
            }

        feature_fields = [
            'rms', 'peak_energy', 'energy_variance', 'spectral_centroid',
            'spectral_rolloff', 'spectral_flux', 'zero_crossing_rate',
            'tempo', 'fundamental_freq', 'pitch_stability', 'pitch_range', 'confidence'
        ]

        summary: Dict[str, Any] = {}
        for field in feature_fields:
            values = [getattr(f, field) for f in features_list]
            summary[field] = stats(values)

        # Handle MFCCs separately to retain per-coefficient statistics
        mfcc_matrix = np.array([f.mfccs for f in features_list if hasattr(f, 'mfccs')])
        if mfcc_matrix.size:
            summary['mfccs'] = [stats(mfcc_matrix[:, i]) for i in range(mfcc_matrix.shape[1])]

        # Voice activity ratio
        voice_activity_flags = [getattr(f, 'voice_activity', False) for f in features_list]
        if voice_activity_flags:
            activity_ratio = sum(1 for v in voice_activity_flags if v) / len(voice_activity_flags)
            summary['voice_activity_ratio'] = activity_ratio

        return summary

    def _summarize_moods(self, mood_results: List['MoodResult']) -> Dict[str, Any]:
        """Summarize mood classifications across analyzed frames."""
        if not mood_results:
            return {'error': 'Mood classification unavailable'}

        mood_counts: Dict[str, int] = {}
        confidences: List[float] = []
        for result in mood_results:
            mood_counts[result.mood] = mood_counts.get(result.mood, 0) + 1
            confidences.append(result.confidence)

        dominant_mood = max(mood_counts, key=mood_counts.get)
        average_confidence = float(np.mean(confidences)) if confidences else 0.0

        return {
            'dominant_mood': dominant_mood,
            'mood_counts': mood_counts,
            'average_confidence': average_confidence,
            'last_debug_scores': mood_results[-1].debug_scores if mood_results else {}
        }
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on diagnostic results."""
        recommendations = []
        
        # System recommendations
        if not results['system_info']['enhanced_available']:
            recommendations.append("Install enhanced mood detection components for improved accuracy")
        
        if not results['system_info']['config_file_exists']:
            recommendations.append("Create mood_config.json file for customizable thresholds")
        
        if not results['system_info']['calibration_data_dir']:
            recommendations.append("Create calibration_data directory for user-specific tuning")
        
        # Configuration recommendations
        config_analysis = results.get('configuration_analysis', {})
        if not config_analysis.get('config_valid', True):
            recommendations.append("Fix configuration validation errors")
        
        # Performance recommendations
        perf_analysis = results.get('performance_analysis', {})
        if perf_analysis.get('average_processing_time_ms', 0) > 20:
            recommendations.append("Consider reducing feature complexity for better real-time performance")
        
        if perf_analysis.get('audio_underruns', 0) > 0:
            recommendations.append("Audio underruns detected - check system load and audio buffer settings")
        
        # Component recommendations
        component_status = results.get('component_status', {})
        missing_components = [name for name, available in component_status.items() if not available]
        if missing_components:
            recommendations.append(f"Missing components: {', '.join(missing_components)}")
        
        return recommendations
    
    def print_diagnostic_report(self, results: Dict[str, Any]):
        """Print formatted diagnostic report."""
        print("\n" + "="*80)
        print("MOOD DETECTION SYSTEM DIAGNOSTIC REPORT")
        print("="*80)
        print(f"Generated: {results['timestamp']}")
        print()
        
        # System Info
        print("SYSTEM INFORMATION:")
        print("-" * 40)
        sys_info = results['system_info']
        print(f"Enhanced Mode:      {'✓' if sys_info['enhanced_available'] else '✗'}")
        print(f"Config File:        {'✓' if sys_info['config_file_exists'] else '✗'}")
        print(f"Calibration Data:   {'✓' if sys_info['calibration_data_dir'] else '✗'}")
        print(f"Python Version:     {sys_info['python_version'].split()[0]}")
        print()
        
        # Component Status
        print("COMPONENT STATUS:")
        print("-" * 40)
        comp_status = results['component_status']
        for component, available in comp_status.items():
            status = "✓" if available else "✗"
            print(f"{component:25} {status}")
        print()
        
        # Configuration Analysis
        print("CONFIGURATION ANALYSIS:")
        print("-" * 40)
        config_analysis = results.get('configuration_analysis', {})
        if 'error' in config_analysis:
            print(f"Error: {config_analysis['error']}")
        else:
            print(f"Configuration Valid: {'✓' if config_analysis.get('config_valid', False) else '✗'}")
            if config_analysis.get('validation_errors'):
                for error in config_analysis['validation_errors']:
                    print(f"  - {error}")
        print()
        
        # Performance Analysis
        print("PERFORMANCE ANALYSIS:")
        print("-" * 40)
        perf_analysis = results.get('performance_analysis', {})
        if 'error' in perf_analysis:
            print(f"Error: {perf_analysis['error']}")
        else:
            print(f"Monitoring Active:  {'✓' if perf_analysis.get('performance_monitoring_active', False) else '✗'}")
            print(f"Performance Level:  {perf_analysis.get('current_performance_level', 'unknown')}")
            print(f"Avg Processing:     {perf_analysis.get('average_processing_time_ms', 0):.1f}ms")
            print(f"Total Cycles:       {perf_analysis.get('total_cycles', 0)}")
            print(f"Audio Underruns:    {perf_analysis.get('audio_underruns', 0)}")
        print()
        
        # Recommendations
        print("RECOMMENDATIONS:")
        print("-" * 40)
        recommendations = results.get('recommendations', [])
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                print(f"{i}. {rec}")
        else:
            print("No recommendations - system appears to be functioning optimally")
        
        print("\n" + "="*80)


class ConfigValidator:
    """
    Configuration validation and recommendation tools.
    Provides validation, optimization suggestions, and configuration tuning.
    """
    
    def __init__(self):
        self.config_manager = ConfigManager() if ENHANCED_AVAILABLE else None
    
    def validate_configuration(self, config_path: str = "mood_config.json") -> Dict[str, Any]:
        """
        Validate configuration file and provide detailed feedback.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'file_exists': os.path.exists(config_path),
            'valid': False,
            'errors': [],
            'warnings': [],
            'recommendations': []
        }
        
        if not self.config_manager:
            results['errors'].append('Configuration manager not available')
            return results
        
        if not results['file_exists']:
            results['errors'].append(f"Configuration file {config_path} not found")
            results['recommendations'].append("Create configuration file using default values")
            return results
        
        try:
            # Load and validate configuration
            config = self.config_manager.load_config()
            self.config_manager.validate_config(config)
            results['valid'] = True
            
            # Check for potential issues
            results['warnings'].extend(self._check_threshold_ranges(config))
            results['recommendations'].extend(self._generate_config_recommendations(config))
            
        except ConfigValidationError as e:
            results['errors'].append(str(e))
        except Exception as e:
            results['errors'].append(f"Unexpected error: {e}")
        
        return results
    
    def _check_threshold_ranges(self, config: 'MoodConfig') -> List[str]:
        """Check for potential threshold range issues."""
        warnings = []
        
        # Check energy threshold gaps
        neutral_min, neutral_max = config.energy.neutral_range
        if config.energy.calm_max > neutral_min:
            warnings.append("Calm and neutral energy ranges overlap")
        
        if config.energy.energetic_min < neutral_max:
            warnings.append("Neutral and energetic energy ranges overlap")
        
        # Check spectral threshold gaps
        if config.spectral.bright_centroid_min - config.spectral.calm_centroid_max < 500:
            warnings.append("Small gap between calm and bright spectral centroids")
        
        # Check temporal threshold gaps
        if config.temporal.energetic_zcr_min - config.temporal.calm_zcr_max < 0.05:
            warnings.append("Small gap between calm and energetic ZCR thresholds")
        
        # Check smoothing parameters
        if config.smoothing.confidence_threshold > 0.9:
            warnings.append("Very high confidence threshold may prevent mood transitions")
        
        if config.smoothing.minimum_duration > 10.0:
            warnings.append("Long minimum duration may make system feel unresponsive")
        
        return warnings
    
    def _generate_config_recommendations(self, config: 'MoodConfig') -> List[str]:
        """Generate configuration optimization recommendations."""
        recommendations = []
        
        # Energy threshold recommendations
        neutral_range_size = config.energy.neutral_range[1] - config.energy.neutral_range[0]
        if neutral_range_size < 0.02:
            recommendations.append("Consider widening neutral energy range for more stable detection")
        
        # Spectral recommendations
        if config.spectral.calm_centroid_max > 2500:
            recommendations.append("High calm centroid threshold may misclassify calm speech")
        
        if config.spectral.bright_centroid_min < 2500:
            recommendations.append("Low bright centroid threshold may over-detect bright moods")
        
        # Smoothing recommendations
        if config.smoothing.transition_time < 1.0:
            recommendations.append("Short transition time may cause jarring mood changes")
        
        if config.smoothing.confidence_threshold < 0.5:
            recommendations.append("Low confidence threshold may cause unstable mood detection")
        
        return recommendations
    
    def suggest_optimal_config(self, user_voice_samples: Optional[List] = None) -> 'MoodConfig':
        """
        Suggest optimal configuration based on user voice samples or defaults.
        
        Args:
            user_voice_samples: Optional list of user voice feature samples
            
        Returns:
            Optimized MoodConfig
        """
        if not self.config_manager:
            raise RuntimeError("Configuration manager not available")
        
        # Start with default configuration
        config = self.config_manager.get_default_config()
        
        if user_voice_samples and ENHANCED_AVAILABLE:
            # Analyze user samples to optimize thresholds
            config = self._optimize_for_user_samples(config, user_voice_samples)
        
        return config
    
    def _optimize_for_user_samples(self, config: 'MoodConfig', samples: List) -> 'MoodConfig':
        """Optimize configuration based on user voice samples."""
        # This would analyze user samples and adjust thresholds
        # For now, return the original config
        return config
    
    def export_config_template(self, filepath: str = "mood_config_template.json") -> bool:
        """
        Export a configuration template with comments.
        
        Args:
            filepath: Path to export template
            
        Returns:
            bool: True if successful
        """
        if not self.config_manager:
            return False
        
        template = {
            "_comment": "Mood Detection Configuration Template",
            "_description": "Adjust these values to tune mood detection for your voice and environment",
            "thresholds": {
                "energy": {
                    "_comment": "Energy-based thresholds (RMS values)",
                    "calm_max": 0.02,
                    "neutral_range": [0.02, 0.08],
                    "energetic_min": 0.08,
                    "excited_min": 0.15
                },
                "spectral": {
                    "_comment": "Spectral feature thresholds (Hz)",
                    "calm_centroid_max": 2000.0,
                    "bright_centroid_min": 3000.0,
                    "rolloff_thresholds": [1500.0, 3000.0, 5000.0]
                },
                "temporal": {
                    "_comment": "Temporal feature thresholds",
                    "calm_zcr_max": 0.05,
                    "energetic_zcr_min": 0.15
                }
            },
            "smoothing": {
                "_comment": "Mood transition smoothing parameters",
                "transition_time": 2.0,
                "minimum_duration": 5.0,
                "confidence_threshold": 0.7
            },
            "noise_filtering": {
                "_comment": "Noise filtering and adaptation settings",
                "noise_gate_threshold": 0.01,
                "adaptive_gain": True,
                "background_learning_rate": 0.1
            }
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(template, f, indent=2)
            print(f"Configuration template exported to {filepath}")
            return True
        except Exception as e:
            print(f"Failed to export template: {e}")
            return False


# Global debug logger instance
_debug_logger = None

def get_debug_logger() -> MoodDebugLogger:
    """Get global debug logger instance."""
    global _debug_logger
    if _debug_logger is None:
        _debug_logger = MoodDebugLogger()
    return _debug_logger


def main():
    """Main function for running diagnostic tools."""
    if len(sys.argv) < 2:
        print("Usage: python mood_debug_tools.py <command> [options]")
        print("Commands:")
        print("  diagnostic - Run full system diagnostic")
        print("  validate - Validate configuration")
        print("  visualize - Start real-time visualization")
        print("  template - Export configuration template")
        return
    
    command = sys.argv[1]
    
    if command == "diagnostic":
        analyzer = DiagnosticAnalyzer()
        results = analyzer.run_full_diagnostic()
        analyzer.print_diagnostic_report(results)
        
        # Export results to file
        with open("diagnostic_report.json", 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to diagnostic_report.json")
    
    elif command == "validate":
        validator = ConfigValidator()
        results = validator.validate_configuration()
        
        print("Configuration Validation Results:")
        print("=" * 40)
        print(f"File exists: {'✓' if results['file_exists'] else '✗'}")
        print(f"Valid: {'✓' if results['valid'] else '✗'}")
        
        if results['errors']:
            print("\nErrors:")
            for error in results['errors']:
                print(f"  - {error}")
        
        if results['warnings']:
            print("\nWarnings:")
            for warning in results['warnings']:
                print(f"  - {warning}")
        
        if results['recommendations']:
            print("\nRecommendations:")
            for rec in results['recommendations']:
                print(f"  - {rec}")
    
    elif command == "template":
        validator = ConfigValidator()
        if validator.export_config_template():
            print("Configuration template exported successfully")
        else:
            print("Failed to export configuration template")
    
    elif command == "visualize":
        print("Real-time visualization mode")
        print("This would start the real-time visualizer")
        print("(Integration with audio system required)")
    
    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()