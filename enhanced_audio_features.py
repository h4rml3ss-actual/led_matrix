#!/usr/bin/env python3
"""
Enhanced audio feature extraction system for mood detection.
Extends the basic RMS/ZCR/spectral centroid features with comprehensive audio analysis.
"""

import numpy as np
import queue
import time
from dataclasses import dataclass
from typing import List, Optional
import scipy.signal
from scipy.fft import fft, fftfreq
from noise_filter import NoiseFilter, VoiceActivityResult
from performance_monitor import get_global_monitor, get_global_scaler
from error_handling import get_global_error_manager, ErrorSeverity


@dataclass
class AudioFeatures:
    """
    Comprehensive audio features for mood detection.
    """
    # Energy features
    rms: float
    peak_energy: float
    energy_variance: float
    
    # Spectral features
    spectral_centroid: float
    spectral_rolloff: float
    spectral_flux: float
    mfccs: List[float]  # First 4 coefficients
    
    # Temporal features
    zero_crossing_rate: float
    tempo: float
    voice_activity: bool
    
    # Pitch features
    fundamental_freq: float
    pitch_stability: float
    pitch_range: float
    
    # Metadata
    timestamp: float
    confidence: float


class EnhancedFeatureExtractor:
    """
    Enhanced audio feature extractor that extends the basic extract_features function
    with comprehensive audio analysis for improved mood detection.
    """
    
    def __init__(self, samplerate: int = 44100, frame_size: int = 1024, enable_noise_filtering: bool = True):
        self.samplerate = samplerate
        self.frame_size = frame_size
        self.previous_spectrum = None
        self.noise_profile = None
        
        # Error handling
        self.error_manager = get_global_error_manager()
        self.consecutive_failures = 0
        self.max_consecutive_failures = 5
        
        # Initialize noise filter with error handling
        self.enable_noise_filtering = enable_noise_filtering
        if enable_noise_filtering:
            try:
                self.noise_filter = NoiseFilter(samplerate, frame_size)
            except Exception as e:
                self.error_manager.handle_error(
                    'noise_filter_init', 'initialization_error', e, ErrorSeverity.MEDIUM
                )
                self.noise_filter = None
                self.enable_noise_filtering = False
        else:
            self.noise_filter = None
        
        # Pre-compute mel filter bank for MFCC calculation
        try:
            self._init_mel_filterbank()
        except Exception as e:
            self.error_manager.handle_error(
                'mfcc_init', 'filterbank_error', e, ErrorSeverity.LOW
            )
            # Continue without MFCC capability
        
        # Performance monitoring integration
        try:
            self.performance_monitor = get_global_monitor()
            self.performance_scaler = get_global_scaler()
            
            # Register for performance configuration updates
            self.performance_scaler.register_config_callback(self._on_performance_config_change)
            self.current_performance_config = self.performance_scaler.get_current_config()
        except Exception as e:
            self.error_manager.handle_error(
                'performance_monitor_init', 'initialization_error', e, ErrorSeverity.LOW
            )
            self.performance_monitor = None
            self.performance_scaler = None
            self.current_performance_config = {}
        
    def _init_mel_filterbank(self, n_mels: int = 13, fmin: float = 80, fmax: float = 8000):
        """Initialize mel-scale filter bank parameters for MFCC computation."""
        # Store parameters for dynamic filter bank creation
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = min(fmax, self.samplerate // 2)
        
        # Convert to mel scale
        self.mel_min = self._hz_to_mel(self.fmin)
        self.mel_max = self._hz_to_mel(self.fmax)
    
    def _create_mel_filterbank(self, n_fft_bins: int) -> np.ndarray:
        """Create mel-scale filter bank for given FFT size."""
        # Create equally spaced mel points
        mel_points = np.linspace(self.mel_min, self.mel_max, self.n_mels + 2)
        hz_points = self._mel_to_hz(mel_points)
        
        # Convert to FFT bin indices
        bin_points = np.floor(n_fft_bins * hz_points / self.samplerate).astype(int)
        bin_points = np.clip(bin_points, 0, n_fft_bins - 1)
        
        # Create filter bank
        mel_filterbank = np.zeros((self.n_mels, n_fft_bins))
        
        for i in range(1, self.n_mels + 1):
            left = bin_points[i - 1]
            center = bin_points[i]
            right = bin_points[i + 1]
            
            # Triangular filter
            for j in range(left, center):
                if center != left:
                    mel_filterbank[i - 1, j] = (j - left) / (center - left)
            for j in range(center, right):
                if right != center:
                    mel_filterbank[i - 1, j] = (right - j) / (right - center)
        
        return mel_filterbank
    
    @staticmethod
    def _hz_to_mel(hz: float) -> float:
        """Convert frequency in Hz to mel scale."""
        return 2595 * np.log10(1 + hz / 700)
    
    @staticmethod
    def _mel_to_hz(mel: float) -> float:
        """Convert mel scale to frequency in Hz."""
        return 700 * (10**(mel / 2595) - 1)
    
    def extract_features(self, pcm_block: np.ndarray, timestamp: float = 0.0) -> AudioFeatures:
        """
        Extract comprehensive audio features from PCM audio block with error handling.
        
        Args:
            pcm_block: Audio samples as numpy array
            timestamp: Timestamp for this audio block
            
        Returns:
            AudioFeatures object with all computed features
        """
        try:
            # Validate input
            if pcm_block is None or len(pcm_block) == 0:
                return self._create_fallback_features(timestamp)
            
            # Normalize audio to [-1, 1] range with error handling
            try:
                if self.performance_monitor:
                    with self.performance_monitor.measure_stage('audio_preprocessing'):
                        audio = self._safe_normalize_audio(pcm_block)
                else:
                    audio = self._safe_normalize_audio(pcm_block)
            except Exception as e:
                self.error_manager.handle_error(
                    'audio_preprocessing', 'normalization_error', e, ErrorSeverity.MEDIUM
                )
                return self._create_fallback_features(timestamp)
            
            # Apply noise filtering if enabled and performance allows
            vad_result = None
            if (self.enable_noise_filtering and 
                self.noise_filter is not None and 
                (not self.performance_scaler or self.performance_scaler.should_enable_feature('noise_filtering_enabled'))):
                try:
                    if self.performance_monitor:
                        with self.performance_monitor.measure_stage('noise_filtering'):
                            audio, vad_result = self.noise_filter.filter_audio(audio)
                    else:
                        audio, vad_result = self.noise_filter.filter_audio(audio)
                except Exception as e:
                    self.error_manager.handle_error(
                        'noise_filtering', 'filter_error', e, ErrorSeverity.LOW
                    )
                    # Continue with unfiltered audio
            
            # Extract feature categories with error handling
            try:
                if self.performance_monitor:
                    with self.performance_monitor.measure_stage('energy_features'):
                        energy_features = self._extract_energy_features(audio)
                else:
                    energy_features = self._extract_energy_features(audio)
            except Exception as e:
                self.error_manager.handle_error(
                    'energy_features', 'extraction_error', e, ErrorSeverity.HIGH
                )
                energy_features = self._get_fallback_energy_features()
            
            # Spectral features - may be disabled for low performance
            try:
                if (not self.performance_scaler or self.performance_scaler.should_enable_feature('spectral_analysis_enabled')):
                    if self.performance_monitor:
                        with self.performance_monitor.measure_stage('spectral_features'):
                            spectral_features = self._extract_spectral_features(audio)
                    else:
                        spectral_features = self._extract_spectral_features(audio)
                else:
                    # Minimal spectral features for compatibility
                    spectral_features = self._extract_minimal_spectral_features(audio)
            except Exception as e:
                self.error_manager.handle_error(
                    'spectral_features', 'extraction_error', e, ErrorSeverity.MEDIUM
                )
                spectral_features = self._get_fallback_spectral_features()
            
            try:
                if self.performance_monitor:
                    with self.performance_monitor.measure_stage('temporal_features'):
                        temporal_features = self._extract_temporal_features(audio, vad_result)
                else:
                    temporal_features = self._extract_temporal_features(audio, vad_result)
            except Exception as e:
                self.error_manager.handle_error(
                    'temporal_features', 'extraction_error', e, ErrorSeverity.MEDIUM
                )
                temporal_features = self._get_fallback_temporal_features()
            
            # Pitch features - may be disabled for medium/low performance
            try:
                if (not self.performance_scaler or self.performance_scaler.should_enable_feature('pitch_analysis_enabled')):
                    if self.performance_monitor:
                        with self.performance_monitor.measure_stage('pitch_features'):
                            pitch_features = self._extract_pitch_features(audio)
                    else:
                        pitch_features = self._extract_pitch_features(audio)
                else:
                    # Minimal pitch features
                    pitch_features = {
                        'fundamental_freq': 0.0,
                        'pitch_stability': 0.5,
                        'pitch_range': 0.0
                    }
            except Exception as e:
                self.error_manager.handle_error(
                    'pitch_features', 'extraction_error', e, ErrorSeverity.LOW
                )
                pitch_features = self._get_fallback_pitch_features()
            
            # Calculate overall confidence based on signal quality
            try:
                confidence = self._calculate_confidence(audio, energy_features, spectral_features, vad_result)
            except Exception as e:
                self.error_manager.handle_error(
                    'confidence_calculation', 'calculation_error', e, ErrorSeverity.LOW
                )
                confidence = 0.5  # Default confidence
            
            # Reset consecutive failures on success
            self.consecutive_failures = 0
            
            return AudioFeatures(
                # Energy features
                rms=energy_features['rms'],
                peak_energy=energy_features['peak_energy'],
                energy_variance=energy_features['energy_variance'],
                
                # Spectral features
                spectral_centroid=spectral_features['spectral_centroid'],
                spectral_rolloff=spectral_features['spectral_rolloff'],
                spectral_flux=spectral_features['spectral_flux'],
                mfccs=spectral_features['mfccs'],
                
                # Temporal features
                zero_crossing_rate=temporal_features['zero_crossing_rate'],
                tempo=temporal_features['tempo'],
                voice_activity=temporal_features['voice_activity'],
                
                # Pitch features
                fundamental_freq=pitch_features['fundamental_freq'],
                pitch_stability=pitch_features['pitch_stability'],
                pitch_range=pitch_features['pitch_range'],
                
                # Metadata
                timestamp=timestamp,
                confidence=confidence
            )
            
        except Exception as e:
            # Critical error in feature extraction
            self.consecutive_failures += 1
            self.error_manager.handle_error(
                'feature_extraction', 'critical_error', e, ErrorSeverity.HIGH
            )
            
            # If too many consecutive failures, disable enhanced features
            if self.consecutive_failures >= self.max_consecutive_failures:
                self.error_manager.handle_error(
                    'feature_extraction', 'too_many_failures', 
                    Exception(f"Too many consecutive failures: {self.consecutive_failures}"),
                    ErrorSeverity.CRITICAL
                )
            
            return self._create_fallback_features(timestamp)
    
    def _extract_energy_features(self, audio: np.ndarray) -> dict:
        """Extract energy-based features."""
        # RMS energy (root mean square)
        rms = np.sqrt(np.mean(audio**2))
        
        # Peak energy
        peak_energy = np.max(np.abs(audio))
        
        # Energy variance (measure of dynamic range)
        frame_length = len(audio) // 10  # Divide into 10 sub-frames
        if frame_length > 0:
            sub_energies = []
            for i in range(0, len(audio) - frame_length, frame_length):
                sub_frame = audio[i:i + frame_length]
                sub_energies.append(np.mean(sub_frame**2))
            energy_variance = np.var(sub_energies) if sub_energies else 0.0
        else:
            energy_variance = 0.0
        
        return {
            'rms': float(rms),
            'peak_energy': float(peak_energy),
            'energy_variance': float(energy_variance)
        }
    
    def _extract_spectral_features(self, audio: np.ndarray) -> dict:
        """Extract spectral features including MFCC."""
        # Compute FFT
        fft_data = fft(audio)
        magnitude_spectrum = np.abs(fft_data[:len(fft_data)//2 + 1])
        freqs = fftfreq(len(audio), 1/self.samplerate)[:len(magnitude_spectrum)]
        
        # Spectral centroid
        if np.sum(magnitude_spectrum) > 1e-6:
            spectral_centroid = np.sum(freqs * magnitude_spectrum) / np.sum(magnitude_spectrum)
        else:
            spectral_centroid = 0.0
        
        # Spectral rolloff (frequency below which 85% of energy is contained)
        cumulative_energy = np.cumsum(magnitude_spectrum**2)
        total_energy = cumulative_energy[-1]
        if total_energy > 1e-6:
            rolloff_threshold = 0.85 * total_energy
            rolloff_idx = np.where(cumulative_energy >= rolloff_threshold)[0]
            spectral_rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else freqs[-1]
        else:
            spectral_rolloff = 0.0
        
        # Spectral flux (measure of spectral change)
        if self.previous_spectrum is not None and len(self.previous_spectrum) == len(magnitude_spectrum):
            spectral_flux = np.sum((magnitude_spectrum - self.previous_spectrum)**2)
        else:
            spectral_flux = 0.0
        self.previous_spectrum = magnitude_spectrum.copy()
        
        # MFCC calculation
        mfccs = self._calculate_mfcc(magnitude_spectrum)
        
        return {
            'spectral_centroid': float(spectral_centroid),
            'spectral_rolloff': float(spectral_rolloff),
            'spectral_flux': float(spectral_flux),
            'mfccs': mfccs
        }
    
    def _calculate_mfcc(self, magnitude_spectrum: np.ndarray, n_mfcc: int = 4) -> List[float]:
        """Calculate Mel-Frequency Cepstral Coefficients."""
        # Get MFCC count from performance configuration
        configured_mfcc_count = self.performance_scaler.get_parameter_value('mfcc_coefficients', n_mfcc)
        
        # Skip MFCC calculation if disabled for performance
        if configured_mfcc_count == 0:
            return []
        
        # Use the configured count
        n_mfcc = min(configured_mfcc_count, n_mfcc)
        
        # Create mel filter bank for current spectrum size
        mel_filterbank = self._create_mel_filterbank(len(magnitude_spectrum))
        
        # Apply mel filter bank
        mel_energies = np.dot(mel_filterbank, magnitude_spectrum)
        
        # Avoid log of zero
        mel_energies = np.maximum(mel_energies, 1e-10)
        
        # Log mel energies
        log_mel_energies = np.log(mel_energies)
        
        # DCT to get cepstral coefficients
        mfccs = scipy.fft.dct(log_mel_energies, type=2, norm='ortho')
        
        # Return first n_mfcc coefficients
        return mfccs[:n_mfcc].tolist()
    
    def _extract_temporal_features(self, audio: np.ndarray, vad_result: Optional[VoiceActivityResult] = None) -> dict:
        """Extract temporal features."""
        # Zero crossing rate
        zero_crossings = np.where(np.diff(np.sign(audio)))[0]
        zero_crossing_rate = len(zero_crossings) / len(audio)
        
        # Simple tempo estimation using autocorrelation
        tempo = self._estimate_tempo(audio)
        
        # Voice activity detection - use enhanced VAD if available, otherwise fallback to simple
        if vad_result is not None:
            voice_activity = vad_result.is_voice
        else:
            # Fallback to simple energy-based VAD
            rms = np.sqrt(np.mean(audio**2))
            voice_activity = rms > 0.01
        
        return {
            'zero_crossing_rate': float(zero_crossing_rate),
            'tempo': float(tempo),
            'voice_activity': bool(voice_activity)
        }
    
    def _estimate_tempo(self, audio: np.ndarray) -> float:
        """Estimate tempo using autocorrelation."""
        # Simple tempo estimation - look for periodicity in energy
        frame_length = self.samplerate // 10  # 100ms frames
        if len(audio) < frame_length * 2:
            return 0.0
        
        # Calculate energy in overlapping frames
        hop_length = frame_length // 2
        energies = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frame_energy = np.mean(audio[i:i + frame_length]**2)
            energies.append(frame_energy)
        
        if len(energies) < 4:
            return 0.0
        
        # Autocorrelation of energy signal
        energies = np.array(energies)
        autocorr = np.correlate(energies, energies, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Find peaks in autocorrelation (excluding zero lag)
        if len(autocorr) > 1:
            peaks, _ = scipy.signal.find_peaks(autocorr[1:], height=np.max(autocorr) * 0.3)
            if len(peaks) > 0:
                # Convert lag to tempo (beats per minute)
                lag_samples = (peaks[0] + 1) * hop_length
                lag_seconds = lag_samples / self.samplerate
                tempo = 60.0 / lag_seconds if lag_seconds > 0 else 0.0
                return min(tempo, 200.0)  # Cap at reasonable tempo
        
        return 0.0
    
    def _extract_pitch_features(self, audio: np.ndarray) -> dict:
        """Extract pitch-related features."""
        # Fundamental frequency estimation using autocorrelation
        fundamental_freq = self._estimate_f0_autocorr(audio)
        
        # Pitch stability (variance in F0 over time)
        pitch_stability = self._calculate_pitch_stability(audio)
        
        # Pitch range (difference between max and min F0)
        pitch_range = self._calculate_pitch_range(audio)
        
        return {
            'fundamental_freq': float(fundamental_freq),
            'pitch_stability': float(pitch_stability),
            'pitch_range': float(pitch_range)
        }
    
    def _estimate_f0_autocorr(self, audio: np.ndarray) -> float:
        """Estimate fundamental frequency using autocorrelation."""
        # Apply window to reduce edge effects
        windowed = audio * np.hanning(len(audio))
        
        # Autocorrelation
        autocorr = np.correlate(windowed, windowed, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Find the first significant peak (excluding zero lag)
        min_period = int(self.samplerate / 800)  # 800 Hz max
        max_period = int(self.samplerate / 80)   # 80 Hz min
        
        if len(autocorr) > max_period:
            search_range = autocorr[min_period:max_period]
            if len(search_range) > 0:
                peak_idx = np.argmax(search_range) + min_period
                if autocorr[peak_idx] > 0.3 * autocorr[0]:  # Significant peak
                    f0 = self.samplerate / peak_idx
                    return f0
        
        return 0.0
    
    def _calculate_pitch_stability(self, audio: np.ndarray) -> float:
        """Calculate pitch stability by analyzing F0 variance over time."""
        # Divide audio into overlapping frames
        frame_length = self.frame_size // 4
        hop_length = frame_length // 2
        
        f0_values = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i + frame_length]
            f0 = self._estimate_f0_autocorr(frame)
            if f0 > 0:  # Only consider voiced frames
                f0_values.append(f0)
        
        if len(f0_values) < 2:
            return 0.0
        
        # Calculate coefficient of variation (std/mean)
        f0_array = np.array(f0_values)
        mean_f0 = np.mean(f0_array)
        std_f0 = np.std(f0_array)
        
        if mean_f0 > 0:
            stability = 1.0 - (std_f0 / mean_f0)  # Higher value = more stable
            return max(0.0, stability)
        
        return 0.0
    
    def _calculate_pitch_range(self, audio: np.ndarray) -> float:
        """Calculate pitch range (max F0 - min F0)."""
        # Similar to pitch stability but return range
        frame_length = self.frame_size // 4
        hop_length = frame_length // 2
        
        f0_values = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i + frame_length]
            f0 = self._estimate_f0_autocorr(frame)
            if f0 > 0:
                f0_values.append(f0)
        
        if len(f0_values) < 2:
            return 0.0
        
        f0_array = np.array(f0_values)
        return float(np.max(f0_array) - np.min(f0_array))
    
    def _on_performance_config_change(self, config: dict) -> None:
        """Handle performance configuration changes."""
        self.current_performance_config = config
        
        # Log configuration changes for debugging
        if hasattr(self, 'performance_monitor'):
            print(f"Feature extractor config updated: "
                  f"MFCC={config.get('mfcc_coefficients', 0)}, "
                  f"spectral={config.get('spectral_analysis_enabled', False)}, "
                  f"pitch={config.get('pitch_analysis_enabled', False)}")
    
    def _calculate_confidence(self, audio: np.ndarray, energy_features: dict, spectral_features: dict, vad_result: Optional[VoiceActivityResult] = None) -> float:
        """Calculate confidence score based on signal quality."""
        confidence_factors = []
        
        # Energy-based confidence
        rms = energy_features['rms']
        if rms > 0.001:  # Minimum signal level
            confidence_factors.append(min(rms * 10, 1.0))
        else:
            confidence_factors.append(0.0)
        
        # Spectral clarity
        spectral_centroid = spectral_features['spectral_centroid']
        if 100 < spectral_centroid < 8000:  # Reasonable speech range
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.3)
        
        # Signal-to-noise ratio estimate
        signal_power = np.mean(audio**2)
        if signal_power > 1e-6:
            snr_factor = np.log10(signal_power * 1000) / 2
            confidence_factors.append(max(0.0, min(snr_factor, 1.0)))
        else:
            confidence_factors.append(0.0)
        
        # Voice activity confidence from enhanced VAD
        if vad_result is not None:
            # Use VAD confidence directly
            confidence_factors.append(vad_result.confidence)
            
            # Bonus for voice activity detection
            if vad_result.is_voice:
                confidence_factors.append(0.8)
            else:
                confidence_factors.append(0.3)
        else:
            # Fallback confidence based on simple energy
            if rms > 0.01:
                confidence_factors.append(0.6)
            else:
                confidence_factors.append(0.2)
        
        # Ensure confidence is always between 0 and 1
        confidence = float(np.mean(confidence_factors))
        return max(0.0, min(confidence, 1.0))
    
    def get_baseline_features(self, duration_seconds: int = 5, samplerate: Optional[int] = None,
                              frame_size: Optional[int] = None) -> Optional[AudioFeatures]:
        """
        Record audio for a baseline window and return aggregated features.

        Args:
            duration_seconds: Duration to record for baseline capture.
            samplerate: Optional override for the recording sample rate.
            frame_size: Optional override for frame size during capture.

        Returns:
            AudioFeatures with averaged metrics over the capture window, or None on failure.
        """
        effective_samplerate = samplerate or self.samplerate
        effective_frame_size = frame_size or self.frame_size

        # Temporarily override samplerate/frame_size to align normalization with capture
        original_samplerate = self.samplerate
        original_frame_size = self.frame_size
        self.samplerate = effective_samplerate
        self.frame_size = effective_frame_size

        try:
            recorded_audio = self._record_audio(
                duration_seconds=duration_seconds,
                samplerate=effective_samplerate,
                frame_size=effective_frame_size,
            )

            if recorded_audio is None or len(recorded_audio) == 0:
                return None

            feature_samples = []
            hop = effective_frame_size
            timestamp = 0.0

            for start in range(0, len(recorded_audio) - effective_frame_size + 1, hop):
                segment = recorded_audio[start:start + effective_frame_size]
                features = self.extract_features(segment, timestamp=timestamp)
                feature_samples.append(features)
                timestamp += hop / float(effective_samplerate)

            return self._aggregate_baseline_features(feature_samples)

        except Exception as e:
            self.error_manager.handle_error(
                'baseline_capture', 'baseline_error', e, ErrorSeverity.MEDIUM
            )
            return None
        finally:
            self.samplerate = original_samplerate
            self.frame_size = original_frame_size

    def _record_audio(self, duration_seconds: int, samplerate: int, frame_size: int,
                      channels: int = 1) -> np.ndarray:
        """Record audio from the microphone for the given duration."""
        try:
            import sounddevice as sd
        except Exception as e:
            self.error_manager.handle_error(
                'baseline_capture', 'microphone_unavailable', e, ErrorSeverity.HIGH
            )
            return np.array([])

        audio_queue: "queue.Queue[np.ndarray]" = queue.Queue()
        blocks = []

        def _callback(indata, frames, time_info, status):
            if status:
                self.error_manager.handle_error(
                    'baseline_capture', 'stream_status', Exception(str(status)), ErrorSeverity.LOW
                )
            audio_queue.put(indata.copy())

        stream = None
        try:
            stream = sd.InputStream(
                channels=channels,
                samplerate=samplerate,
                blocksize=frame_size,
                callback=_callback,
            )
            stream.start()

            start_time = time.time()
            while time.time() - start_time < duration_seconds:
                try:
                    block = audio_queue.get(timeout=0.5)
                    blocks.append(block)
                except queue.Empty:
                    continue
        except Exception as e:
            self.error_manager.handle_error(
                'baseline_capture', 'recording_error', e, ErrorSeverity.MEDIUM
            )
            return np.array([])
        finally:
            if stream is not None:
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    pass

        if not blocks:
            return np.array([])

        audio_data = np.concatenate(blocks, axis=0)
        return audio_data.flatten().astype(np.float32)

    def _aggregate_baseline_features(self, feature_samples: List[AudioFeatures]) -> Optional[AudioFeatures]:
        """Aggregate a list of feature samples into a single baseline profile."""
        if not feature_samples:
            return None

        def _avg(field: str) -> float:
            return float(np.mean([getattr(sample, field) for sample in feature_samples]))

        def _avg_list(field: str) -> List[float]:
            values = np.array([getattr(sample, field) for sample in feature_samples], dtype=np.float32)
            return np.mean(values, axis=0).tolist()

        voice_activity_ratio = float(np.mean([1.0 if f.voice_activity else 0.0 for f in feature_samples]))

        return AudioFeatures(
            rms=_avg('rms'),
            peak_energy=_avg('peak_energy'),
            energy_variance=_avg('energy_variance'),
            spectral_centroid=_avg('spectral_centroid'),
            spectral_rolloff=_avg('spectral_rolloff'),
            spectral_flux=_avg('spectral_flux'),
            mfccs=_avg_list('mfccs'),
            zero_crossing_rate=_avg('zero_crossing_rate'),
            tempo=_avg('tempo'),
            voice_activity=voice_activity_ratio >= 0.5,
            fundamental_freq=_avg('fundamental_freq'),
            pitch_stability=_avg('pitch_stability'),
            pitch_range=_avg('pitch_range'),
            timestamp=_avg('timestamp'),
            confidence=_avg('confidence')
        )
    
    def update_noise_profile(self, pcm_block: np.ndarray) -> None:
        """
        Update background noise profile for adaptive filtering.
        """
        if pcm_block.dtype == np.int16:
            audio = pcm_block.astype(np.float32) / 32768.0
        else:
            audio = pcm_block.astype(np.float32)
        
        # Use the enhanced noise filter if available
        if self.enable_noise_filtering and self.noise_filter is not None:
            # The noise filter will update its profile automatically during filtering
            # But we can also manually update it with known noise samples
            self.noise_filter._update_noise_profile(audio)
        
        # Keep the simple noise profile for backward compatibility
        current_energy = np.mean(audio**2)
        
        if self.noise_profile is None:
            self.noise_profile = current_energy
        else:
            # Exponential moving average
            alpha = 0.1
            self.noise_profile = alpha * current_energy + (1 - alpha) * self.noise_profile
    
    def set_noise_gate_threshold(self, threshold: float) -> None:
        """
        Set the noise gate threshold for voice activity detection.
        
        Args:
            threshold: New threshold value for VAD energy detection
        """
        if self.noise_filter is not None:
            self.noise_filter.set_noise_gate_threshold(threshold)
    
    def set_adaptive_gain_target(self, target_rms: float) -> None:
        """
        Set the target RMS level for adaptive gain control.
        
        Args:
            target_rms: Target RMS level for gain adjustment
        """
        if self.noise_filter is not None:
            self.noise_filter.set_adaptive_gain_target(target_rms)
    
    def reset_noise_profile(self) -> None:
        """Reset the noise profile to start learning from scratch."""
        if self.noise_filter is not None:
            self.noise_filter.reset_noise_profile()
        self.noise_profile = None
    
    def get_noise_filter_info(self) -> dict:
        """
        Get information about the current noise filter state.
        
        Returns:
            Dictionary with noise filter statistics
        """
        if self.noise_filter is not None:
            info = self.noise_filter.get_noise_profile_info()
            info['current_gain'] = self.noise_filter.get_current_gain()
            info['noise_filtering_enabled'] = True
            return info
        else:
            return {
                'noise_filtering_enabled': False,
                'initialized': False,
                'update_count': 0,
                'noise_energy': float(self.noise_profile) if self.noise_profile is not None else 0.0,
                'current_gain': 1.0
            }
    
    def _safe_normalize_audio(self, pcm_block: np.ndarray) -> np.ndarray:
        """Safely normalize audio with error handling."""
        try:
            # Normalize audio to [-1, 1] range
            if pcm_block.dtype == np.int16:
                audio = pcm_block.astype(np.float32) / 32768.0
            else:
                audio = pcm_block.astype(np.float32)
            
            # Flatten if stereo
            if len(audio.shape) > 1:
                audio = audio.flatten()
            
            # Handle NaN or inf values
            audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Ensure we have the right frame size
            if len(audio) != self.frame_size:
                if len(audio) < self.frame_size:
                    # Pad with zeros
                    padded = np.zeros(self.frame_size, dtype=np.float32)
                    padded[:len(audio)] = audio
                    audio = padded
                else:
                    # Truncate
                    audio = audio[:self.frame_size]
            
            return audio
            
        except Exception as e:
            # Return silence if normalization fails
            return np.zeros(self.frame_size, dtype=np.float32)
    
    def _create_fallback_features(self, timestamp: float = 0.0) -> AudioFeatures:
        """Create fallback features when extraction fails."""
        return AudioFeatures(
            # Energy features
            rms=0.05,
            peak_energy=0.1,
            energy_variance=0.001,
            
            # Spectral features
            spectral_centroid=2000.0,
            spectral_rolloff=4000.0,
            spectral_flux=100.0,
            mfccs=[],
            
            # Temporal features
            zero_crossing_rate=0.1,
            tempo=120.0,
            voice_activity=True,
            
            # Pitch features
            fundamental_freq=150.0,
            pitch_stability=0.5,
            pitch_range=50.0,
            
            # Metadata
            timestamp=timestamp,
            confidence=0.2  # Low confidence for fallback
        )
    
    def _get_fallback_energy_features(self) -> dict:
        """Get fallback energy features."""
        return {
            'rms': 0.05,
            'peak_energy': 0.1,
            'energy_variance': 0.001
        }
    
    def _get_fallback_spectral_features(self) -> dict:
        """Get fallback spectral features."""
        return {
            'spectral_centroid': 2000.0,
            'spectral_rolloff': 4000.0,
            'spectral_flux': 100.0,
            'mfccs': []
        }
    
    def _get_fallback_temporal_features(self) -> dict:
        """Get fallback temporal features."""
        return {
            'zero_crossing_rate': 0.1,
            'tempo': 120.0,
            'voice_activity': True
        }
    
    def _get_fallback_pitch_features(self) -> dict:
        """Get fallback pitch features."""
        return {
            'fundamental_freq': 150.0,
            'pitch_stability': 0.5,
            'pitch_range': 50.0
        }
    
    def _extract_minimal_spectral_features(self, audio: np.ndarray) -> dict:
        """Extract minimal spectral features for low-performance mode."""
        try:
            # Simple spectral centroid calculation
            fft_data = fft(audio)
            magnitude_spectrum = np.abs(fft_data[:len(fft_data)//2 + 1])
            freqs = fftfreq(len(audio), 1/self.samplerate)[:len(magnitude_spectrum)]
            
            if np.sum(magnitude_spectrum) > 1e-6:
                spectral_centroid = np.sum(freqs * magnitude_spectrum) / np.sum(magnitude_spectrum)
            else:
                spectral_centroid = 2000.0
            
            return {
                'spectral_centroid': float(spectral_centroid),
                'spectral_rolloff': spectral_centroid * 1.5,  # Estimate
                'spectral_flux': 0.0,  # Skip flux calculation
                'mfccs': []  # Skip MFCC calculation
            }
        except Exception:
            return self._get_fallback_spectral_features()