#!/usr/bin/env python3
"""
Noise filtering and voice activity detection system for enhanced mood detection.
Implements background noise reduction, voice activity detection, adaptive gain control,
and spectral subtraction for consistent audio processing.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List
import scipy.signal
from scipy.fft import fft, ifft, fftfreq
import time


@dataclass
class NoiseProfile:
    """
    Background noise profile for spectral subtraction.
    """
    noise_spectrum: np.ndarray
    noise_energy: float
    update_count: int
    last_update: float


@dataclass
class VoiceActivityResult:
    """
    Result of voice activity detection.
    """
    is_voice: bool
    confidence: float
    energy_ratio: float
    spectral_ratio: float
    zero_crossing_ratio: float


class NoiseFilter:
    """
    Comprehensive noise filtering system with voice activity detection,
    adaptive gain control, and spectral subtraction.
    """
    
    def __init__(self, samplerate: int = 44100, frame_size: int = 1024):
        """
        Initialize the noise filter.
        
        Args:
            samplerate: Audio sample rate in Hz
            frame_size: Audio frame size in samples
        """
        self.samplerate = samplerate
        self.frame_size = frame_size
        
        # Noise profile for spectral subtraction
        self.noise_profile: Optional[NoiseProfile] = None
        self.noise_learning_rate = 0.1
        self.noise_update_threshold = 0.5  # Seconds between updates
        
        # Voice activity detection parameters
        self.vad_energy_threshold = 0.01
        self.vad_spectral_threshold = 1500  # Hz
        self.vad_zcr_threshold = 0.1
        self.vad_history: List[bool] = []
        self.vad_history_length = 5
        
        # Adaptive gain control
        self.target_rms = 0.05
        self.gain_smoothing = 0.9
        self.current_gain = 1.0
        self.min_gain = 0.1
        self.max_gain = 5.0
        
        # Spectral subtraction parameters
        self.alpha = 2.0  # Over-subtraction factor
        self.beta = 0.01  # Spectral floor factor
        self.gamma = 1.0  # Magnitude averaging factor
        
        # Multi-band filtering (focus on human voice range)
        self.voice_freq_min = 80   # Hz
        self.voice_freq_max = 8000 # Hz
        
        # Pre-compute frequency bins for voice range
        self.freqs = fftfreq(frame_size, 1/samplerate)[:frame_size//2 + 1]
        self.voice_bins = np.where(
            (self.freqs >= self.voice_freq_min) & 
            (self.freqs <= self.voice_freq_max)
        )[0]
        
        # Smoothing buffers
        self.prev_magnitude = None
        self.prev_phase = None
    
    def filter_audio(self, audio: np.ndarray, update_noise_profile: bool = True) -> Tuple[np.ndarray, VoiceActivityResult]:
        """
        Apply comprehensive noise filtering to audio signal.
        
        Args:
            audio: Input audio signal
            update_noise_profile: Whether to update noise profile during non-voice periods
            
        Returns:
            Tuple of (filtered_audio, voice_activity_result)
        """
        # Normalize input audio
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        else:
            audio = audio.astype(np.float32)
        
        # Flatten if stereo
        if len(audio.shape) > 1:
            audio = audio.flatten()
        
        # Ensure we have the right frame size
        if len(audio) != self.frame_size:
            if len(audio) < self.frame_size:
                # Pad with zeros
                padded = np.zeros(self.frame_size)
                padded[:len(audio)] = audio
                audio = padded
            else:
                # Truncate
                audio = audio[:self.frame_size]
        
        # Step 1: Voice Activity Detection
        vad_result = self.detect_voice_activity(audio)
        
        # Step 2: Update noise profile during non-voice periods
        if update_noise_profile and not vad_result.is_voice:
            self._update_noise_profile(audio)
        
        # Step 3: Apply spectral subtraction if noise profile is available
        if self.noise_profile is not None:
            audio = self._apply_spectral_subtraction(audio)
        
        # Step 4: Apply multi-band filtering (focus on voice frequencies)
        audio = self._apply_voice_band_filter(audio)
        
        # Step 5: Apply adaptive gain control
        audio = self._apply_adaptive_gain(audio, vad_result.is_voice)
        
        # Step 6: Apply smoothing to reduce artifacts
        audio = self._apply_temporal_smoothing(audio)
        
        return audio, vad_result
    
    def detect_voice_activity(self, audio: np.ndarray) -> VoiceActivityResult:
        """
        Detect voice activity using multiple features.
        
        Args:
            audio: Input audio signal
            
        Returns:
            VoiceActivityResult: Detection result with confidence
        """
        # Calculate basic features
        rms_energy = np.sqrt(np.mean(audio**2))
        
        # Zero-crossing rate
        if len(audio) == 0:
            zcr = 0.0
        else:
            zero_crossings = np.where(np.diff(np.sign(audio)))[0]
            zcr = len(zero_crossings) / len(audio)
        
        # Spectral features
        fft_data = fft(audio * np.hanning(len(audio)))
        magnitude_spectrum = np.abs(fft_data[:len(fft_data)//2 + 1])
        
        # Spectral centroid
        if np.sum(magnitude_spectrum) > 1e-6:
            spectral_centroid = np.sum(self.freqs * magnitude_spectrum) / np.sum(magnitude_spectrum)
        else:
            spectral_centroid = 0.0
        
        # Voice-specific spectral energy (focus on voice frequencies)
        voice_energy = np.sum(magnitude_spectrum[self.voice_bins])
        total_energy = np.sum(magnitude_spectrum)
        voice_energy_ratio = voice_energy / max(total_energy, 1e-6)
        
        # Feature-based decisions
        energy_decision = rms_energy > self.vad_energy_threshold
        spectral_decision = spectral_centroid > self.vad_spectral_threshold
        zcr_decision = zcr > self.vad_zcr_threshold
        voice_band_decision = voice_energy_ratio > 0.6
        
        # Combine decisions with weights
        feature_scores = {
            'energy': 1.0 if energy_decision else 0.0,
            'spectral': 1.0 if spectral_decision else 0.0,
            'zcr': 1.0 if zcr_decision else 0.0,
            'voice_band': 1.0 if voice_band_decision else 0.0
        }
        
        # Weighted combination
        weights = {'energy': 0.4, 'spectral': 0.2, 'zcr': 0.2, 'voice_band': 0.2}
        confidence = sum(weights[feature] * score for feature, score in feature_scores.items())
        
        # Final decision with hysteresis
        is_voice = confidence > 0.5
        
        # Apply temporal smoothing using history
        self.vad_history.append(is_voice)
        if len(self.vad_history) > self.vad_history_length:
            self.vad_history.pop(0)
        
        # Smooth decision based on recent history
        if len(self.vad_history) >= 3:
            recent_voice_count = sum(self.vad_history[-3:])
            if recent_voice_count >= 2:
                is_voice = True
            elif recent_voice_count == 0:
                is_voice = False
            # Otherwise keep current decision
        
        return VoiceActivityResult(
            is_voice=is_voice,
            confidence=confidence,
            energy_ratio=rms_energy / max(self.vad_energy_threshold, 1e-6),
            spectral_ratio=spectral_centroid / max(self.vad_spectral_threshold, 1e-6),
            zero_crossing_ratio=zcr / max(self.vad_zcr_threshold, 1e-6)
        )
    
    def _update_noise_profile(self, audio: np.ndarray) -> None:
        """
        Update the background noise profile using current audio.
        
        Args:
            audio: Audio signal assumed to contain only noise
        """
        current_time = time.time()
        
        # Calculate current spectrum
        windowed_audio = audio * np.hanning(len(audio))
        fft_data = fft(windowed_audio)
        current_spectrum = np.abs(fft_data[:len(fft_data)//2 + 1])
        current_energy = np.mean(audio**2)
        
        if self.noise_profile is None:
            # Initialize noise profile
            self.noise_profile = NoiseProfile(
                noise_spectrum=current_spectrum.copy(),
                noise_energy=current_energy,
                update_count=1,
                last_update=current_time
            )
        else:
            # Check if enough time has passed for update
            if current_time - self.noise_profile.last_update > self.noise_update_threshold:
                # Exponential moving average update
                alpha = self.noise_learning_rate
                self.noise_profile.noise_spectrum = (
                    alpha * current_spectrum + 
                    (1 - alpha) * self.noise_profile.noise_spectrum
                )
                self.noise_profile.noise_energy = (
                    alpha * current_energy + 
                    (1 - alpha) * self.noise_profile.noise_energy
                )
                self.noise_profile.update_count += 1
                self.noise_profile.last_update = current_time
    
    def _apply_spectral_subtraction(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply spectral subtraction to remove background noise.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Noise-reduced audio signal
        """
        if self.noise_profile is None:
            return audio
        
        # Apply window to reduce edge effects
        windowed_audio = audio * np.hanning(len(audio))
        
        # Compute FFT
        fft_data = fft(windowed_audio)
        magnitude = np.abs(fft_data)
        phase = np.angle(fft_data)
        
        # Only process positive frequencies
        half_len = len(magnitude) // 2 + 1
        magnitude_half = magnitude[:half_len]
        
        # Spectral subtraction
        noise_magnitude = self.noise_profile.noise_spectrum
        
        # Ensure noise profile matches current spectrum length
        if len(noise_magnitude) != len(magnitude_half):
            # Interpolate noise profile to match current length
            noise_magnitude = np.interp(
                np.linspace(0, 1, len(magnitude_half)),
                np.linspace(0, 1, len(noise_magnitude)),
                noise_magnitude
            )
        
        # Over-subtraction with spectral floor
        subtracted_magnitude = magnitude_half - self.alpha * noise_magnitude
        
        # Apply spectral floor to prevent over-subtraction artifacts
        spectral_floor = self.beta * magnitude_half
        subtracted_magnitude = np.maximum(subtracted_magnitude, spectral_floor)
        
        # Smooth the magnitude spectrum to reduce musical noise
        if self.prev_magnitude is not None and len(self.prev_magnitude) == len(subtracted_magnitude):
            subtracted_magnitude = (
                self.gamma * subtracted_magnitude + 
                (1 - self.gamma) * self.prev_magnitude
            )
        self.prev_magnitude = subtracted_magnitude.copy()
        
        # Reconstruct full spectrum (mirror for negative frequencies)
        full_magnitude = np.concatenate([
            subtracted_magnitude,
            subtracted_magnitude[-2:0:-1]  # Mirror, excluding DC and Nyquist
        ])
        
        # Reconstruct complex spectrum
        reconstructed_fft = full_magnitude * np.exp(1j * phase)
        
        # Inverse FFT
        filtered_audio = np.real(ifft(reconstructed_fft))
        
        # Remove window effect (avoid division by zero)
        window = np.hanning(len(filtered_audio))
        # Only divide where window is not zero
        nonzero_mask = window > 1e-10
        filtered_audio[nonzero_mask] = filtered_audio[nonzero_mask] / window[nonzero_mask]
        
        # Handle potential NaN or inf values
        filtered_audio = np.nan_to_num(filtered_audio, nan=0.0, posinf=0.0, neginf=0.0)
        
        return filtered_audio
    
    def _apply_voice_band_filter(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply band-pass filter to focus on human voice frequencies.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Band-pass filtered audio
        """
        # Design Butterworth band-pass filter
        nyquist = self.samplerate / 2
        low_freq = self.voice_freq_min / nyquist
        high_freq = min(self.voice_freq_max / nyquist, 0.99)  # Avoid Nyquist frequency
        
        try:
            # 4th order Butterworth filter
            b, a = scipy.signal.butter(4, [low_freq, high_freq], btype='band')
            
            # Apply filter with zero-phase filtering to avoid delay
            filtered_audio = scipy.signal.filtfilt(b, a, audio)
            
            return filtered_audio
        except Exception:
            # If filter design fails, return original audio
            return audio
    
    def _apply_adaptive_gain(self, audio: np.ndarray, is_voice: bool) -> np.ndarray:
        """
        Apply adaptive gain control based on ambient noise levels.
        
        Args:
            audio: Input audio signal
            is_voice: Whether voice activity is detected
            
        Returns:
            Gain-adjusted audio signal
        """
        current_rms = np.sqrt(np.mean(audio**2))
        
        if is_voice and current_rms > 1e-6:
            # Calculate desired gain to reach target RMS
            desired_gain = self.target_rms / current_rms
            
            # Smooth gain changes to avoid artifacts
            self.current_gain = (
                self.gain_smoothing * self.current_gain + 
                (1 - self.gain_smoothing) * desired_gain
            )
            
            # Clamp gain to reasonable limits
            self.current_gain = np.clip(self.current_gain, self.min_gain, self.max_gain)
        
        # Apply gain
        gained_audio = audio * self.current_gain
        
        # Prevent clipping
        max_val = np.max(np.abs(gained_audio))
        if max_val > 0.95:
            gained_audio = gained_audio * (0.95 / max_val)
        
        return gained_audio
    
    def _apply_temporal_smoothing(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply temporal smoothing to reduce processing artifacts.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Smoothed audio signal
        """
        # Simple moving average smoothing
        kernel_size = 3
        kernel = np.ones(kernel_size) / kernel_size
        
        # Pad audio to handle edges
        padded_audio = np.pad(audio, (kernel_size//2, kernel_size//2), mode='edge')
        
        # Apply convolution
        smoothed_audio = np.convolve(padded_audio, kernel, mode='valid')
        
        return smoothed_audio
    
    def set_noise_gate_threshold(self, threshold: float) -> None:
        """
        Set the noise gate threshold for voice activity detection.
        
        Args:
            threshold: New threshold value
        """
        self.vad_energy_threshold = max(0.001, threshold)
    
    def set_adaptive_gain_target(self, target_rms: float) -> None:
        """
        Set the target RMS level for adaptive gain control.
        
        Args:
            target_rms: Target RMS level
        """
        self.target_rms = max(0.01, min(target_rms, 0.5))
    
    def reset_noise_profile(self) -> None:
        """Reset the noise profile to start learning from scratch."""
        self.noise_profile = None
    
    def get_noise_profile_info(self) -> dict:
        """
        Get information about the current noise profile.
        
        Returns:
            Dictionary with noise profile statistics
        """
        if self.noise_profile is None:
            return {
                'initialized': False,
                'update_count': 0,
                'noise_energy': 0.0,
                'last_update': 0.0
            }
        
        return {
            'initialized': True,
            'update_count': self.noise_profile.update_count,
            'noise_energy': float(self.noise_profile.noise_energy),
            'last_update': self.noise_profile.last_update,
            'spectrum_length': len(self.noise_profile.noise_spectrum)
        }
    
    def get_current_gain(self) -> float:
        """Get the current adaptive gain value."""
        return self.current_gain


# Convenience function for simple noise filtering
def filter_audio_simple(audio: np.ndarray, noise_filter: Optional[NoiseFilter] = None) -> Tuple[np.ndarray, bool]:
    """
    Simple noise filtering function for backward compatibility.
    
    Args:
        audio: Input audio signal
        noise_filter: Optional NoiseFilter instance
        
    Returns:
        Tuple of (filtered_audio, is_voice)
    """
    if noise_filter is None:
        noise_filter = NoiseFilter()
    
    filtered_audio, vad_result = noise_filter.filter_audio(audio)
    return filtered_audio, vad_result.is_voice