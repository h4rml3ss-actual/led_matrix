#!/usr/bin/env python3
"""
Mood transition smoothing system for anti-flickering logic.
Implements confidence threshold checking, minimum duration holds, and smooth transition timing.
"""

import time
from dataclasses import dataclass
from typing import Optional, List, Dict
from collections import deque
from mood_config import MoodConfig, ConfigManager


@dataclass
class TransitionState:
    """
    Represents the current state of mood transitions.
    """
    current_mood: str
    last_transition_time: float
    mood_start_time: float
    transition_in_progress: bool
    transition_progress: float  # 0.0 to 1.0
    confidence_buffer: deque  # Buffer of recent confidence scores
    mood_buffer: deque  # Buffer of recent mood detections


class MoodTransitionSmoother:
    """
    Handles smooth mood transitions with anti-flickering logic.
    
    This class prevents jarring mood changes by implementing:
    - Confidence threshold checking before transitions
    - Minimum duration holds for each mood
    - Transition buffers to require consistent detections
    - Smooth transition timing for frame changes
    """
    
    def __init__(self, config: Optional[MoodConfig] = None):
        """
        Initialize the mood transition smoother.
        
        Args:
            config: MoodConfig object, or None to use default configuration
        """
        if config is None:
            config_manager = ConfigManager()
            config = config_manager.load_config()
        
        self.config = config
        
        # Initialize transition state
        self.state = TransitionState(
            current_mood="neutral",
            last_transition_time=0.0,
            mood_start_time=time.time(),
            transition_in_progress=False,
            transition_progress=0.0,
            confidence_buffer=deque(maxlen=5),  # Keep last 5 confidence scores
            mood_buffer=deque(maxlen=3)  # Keep last 3 mood detections
        )
        
        # Transition timing parameters
        self.transition_start_time = 0.0
        self.target_mood = None
        
        # Statistics for debugging
        self.transition_count = 0
        self.rejected_transitions = 0
        
    def smooth_mood_transition(self, new_mood: str, confidence: float) -> str:
        """
        Process a new mood detection and return the smoothed mood.
        
        Args:
            new_mood: Newly detected mood
            confidence: Confidence score for the detection (0.0 to 1.0)
            
        Returns:
            str: The current smoothed mood to use
        """
        current_time = time.time()
        
        # Update buffers
        self.state.confidence_buffer.append(confidence)
        self.state.mood_buffer.append(new_mood)
        
        # Check if we should transition to the new mood
        if self.should_transition(self.state.current_mood, new_mood, confidence):
            self._start_transition(new_mood, current_time)
        
        # Update transition progress if in progress
        if self.state.transition_in_progress:
            self._update_transition_progress(current_time)
        
        return self.state.current_mood
    
    def should_transition(self, current_mood: str, new_mood: str, confidence: float) -> bool:
        """
        Determine if a mood transition should occur based on smoothing rules.
        
        Args:
            current_mood: Current mood state
            new_mood: Proposed new mood
            confidence: Confidence score for the new mood
            
        Returns:
            bool: True if transition should occur, False otherwise
        """
        current_time = time.time()
        
        # Rule 1: Don't transition if already in progress
        if self.state.transition_in_progress:
            return False
        
        # Rule 2: Don't transition to the same mood
        if new_mood == current_mood:
            return False
        
        # Rule 3: Check confidence threshold
        if confidence < self.config.smoothing.confidence_threshold:
            self.rejected_transitions += 1
            return False
        
        # Rule 4: Check minimum duration hold
        time_in_current_mood = current_time - self.state.mood_start_time
        if time_in_current_mood < self.config.smoothing.minimum_duration:
            self.rejected_transitions += 1
            return False
        
        # Rule 5: Check transition buffer consistency
        if not self._check_transition_buffer_consistency(new_mood):
            self.rejected_transitions += 1
            return False
        
        # Rule 6: Check average confidence over buffer
        if len(self.state.confidence_buffer) >= 3:
            avg_confidence = sum(self.state.confidence_buffer) / len(self.state.confidence_buffer)
            if avg_confidence < self.config.smoothing.confidence_threshold * 0.8:  # Slightly lower threshold for average
                self.rejected_transitions += 1
                return False
        
        return True
    
    def get_transition_progress(self) -> float:
        """
        Get the current transition progress.
        
        Returns:
            float: Progress from 0.0 (start) to 1.0 (complete)
        """
        return self.state.transition_progress
    
    def is_transitioning(self) -> bool:
        """
        Check if a transition is currently in progress.
        
        Returns:
            bool: True if transitioning, False otherwise
        """
        return self.state.transition_in_progress
    
    def get_current_mood(self) -> str:
        """
        Get the current smoothed mood.
        
        Returns:
            str: Current mood
        """
        return self.state.current_mood
    
    def get_target_mood(self) -> Optional[str]:
        """
        Get the target mood if transitioning.
        
        Returns:
            Optional[str]: Target mood if transitioning, None otherwise
        """
        return self.target_mood if self.state.transition_in_progress else None
    
    def force_transition(self, new_mood: str) -> None:
        """
        Force an immediate transition to a new mood, bypassing smoothing rules.
        
        Args:
            new_mood: Mood to transition to immediately
        """
        current_time = time.time()
        self.state.current_mood = new_mood
        self.state.mood_start_time = current_time
        self.state.last_transition_time = current_time
        self.state.transition_in_progress = False
        self.state.transition_progress = 0.0
        self.target_mood = None
        
        # Clear buffers
        self.state.confidence_buffer.clear()
        self.state.mood_buffer.clear()
    
    def reset(self) -> None:
        """
        Reset the smoother to initial state.
        """
        current_time = time.time()
        self.state = TransitionState(
            current_mood="neutral",
            last_transition_time=0.0,
            mood_start_time=current_time,
            transition_in_progress=False,
            transition_progress=0.0,
            confidence_buffer=deque(maxlen=5),
            mood_buffer=deque(maxlen=3)
        )
        self.target_mood = None
        self.transition_start_time = 0.0
        self.transition_count = 0
        self.rejected_transitions = 0
    
    def get_statistics(self) -> Dict[str, any]:
        """
        Get transition statistics for debugging and monitoring.
        
        Returns:
            Dict: Statistics about transitions
        """
        current_time = time.time()
        time_in_current_mood = current_time - self.state.mood_start_time
        
        return {
            'current_mood': self.state.current_mood,
            'time_in_current_mood': time_in_current_mood,
            'transition_in_progress': self.state.transition_in_progress,
            'transition_progress': self.state.transition_progress,
            'target_mood': self.target_mood,
            'total_transitions': self.transition_count,
            'rejected_transitions': self.rejected_transitions,
            'recent_confidences': list(self.state.confidence_buffer),
            'recent_moods': list(self.state.mood_buffer),
            'minimum_duration_remaining': max(0, self.config.smoothing.minimum_duration - time_in_current_mood)
        }
    
    def update_config(self, config: MoodConfig) -> None:
        """
        Update the smoother's configuration.
        
        Args:
            config: New MoodConfig to use
        """
        self.config = config
        
        # Resize buffers if needed
        if self.state.confidence_buffer.maxlen != 5:
            new_buffer = deque(self.state.confidence_buffer, maxlen=5)
            self.state.confidence_buffer = new_buffer
        
        if self.state.mood_buffer.maxlen != 3:
            new_buffer = deque(self.state.mood_buffer, maxlen=3)
            self.state.mood_buffer = new_buffer
    
    def _start_transition(self, new_mood: str, current_time: float) -> None:
        """
        Start a transition to a new mood.
        
        Args:
            new_mood: Target mood for transition
            current_time: Current timestamp
        """
        self.target_mood = new_mood
        self.state.transition_in_progress = True
        self.state.transition_progress = 0.0
        self.transition_start_time = current_time
        self.transition_count += 1
    
    def _update_transition_progress(self, current_time: float) -> None:
        """
        Update the progress of an ongoing transition.
        
        Args:
            current_time: Current timestamp
        """
        if not self.state.transition_in_progress:
            return
        
        # Calculate progress based on elapsed time
        elapsed_time = current_time - self.transition_start_time
        progress = elapsed_time / self.config.smoothing.transition_time
        
        # Use easing function for smoother transitions (ease-in-out)
        if progress <= 1.0:
            progress = self._ease_in_out(progress)
        
        self.state.transition_progress = min(progress, 1.0)
        
        # Complete transition if progress reaches 1.0
        if progress >= 1.0:
            self._complete_transition(current_time)
    
    def _complete_transition(self, current_time: float) -> None:
        """
        Complete the current transition.
        
        Args:
            current_time: Current timestamp
        """
        if self.target_mood:
            self.state.current_mood = self.target_mood
            self.state.mood_start_time = current_time
            self.state.last_transition_time = current_time
        
        self.state.transition_in_progress = False
        self.state.transition_progress = 0.0
        self.target_mood = None
    
    def _check_transition_buffer_consistency(self, new_mood: str) -> bool:
        """
        Check if the mood buffer shows consistent detection of the new mood.
        
        Args:
            new_mood: Mood to check for consistency
            
        Returns:
            bool: True if buffer shows consistent detection
        """
        if len(self.state.mood_buffer) < 2:
            return True  # Not enough data, allow transition
        
        # Count how many recent detections match the new mood
        recent_moods = list(self.state.mood_buffer)
        matching_count = sum(1 for mood in recent_moods if mood == new_mood)
        
        # Require at least 2 out of 3 recent detections to match
        # But be more strict - require majority of recent detections
        required_consistency = max(2, (len(recent_moods) + 1) // 2)
        return matching_count >= required_consistency
    
    @staticmethod
    def _ease_in_out(t: float) -> float:
        """
        Apply ease-in-out easing function for smooth transitions.
        
        Args:
            t: Progress value from 0.0 to 1.0
            
        Returns:
            float: Eased progress value
        """
        if t < 0.5:
            return 2 * t * t
        else:
            return -1 + (4 - 2 * t) * t