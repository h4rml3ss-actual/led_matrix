#!/usr/bin/env python3
"""
Demo script showing the MoodTransitionSmoother in action.
Demonstrates anti-flickering logic, confidence thresholds, and smooth transitions.
"""

import time
from mood_transition_smoother import MoodTransitionSmoother
from mood_config import MoodConfig, SmoothingConfig


def demo_basic_transition():
    """Demonstrate basic mood transition with smoothing."""
    print("=== Basic Transition Demo ===")
    
    # Create smoother with fast settings for demo
    config = MoodConfig()
    config.smoothing = SmoothingConfig(
        transition_time=1.0,  # 1 second transitions
        minimum_duration=2.0,  # 2 second minimum hold
        confidence_threshold=0.7
    )
    
    smoother = MoodTransitionSmoother(config)
    
    print(f"Initial mood: {smoother.get_current_mood()}")
    
    # Try to transition immediately (should be blocked by minimum duration)
    result = smoother.smooth_mood_transition("energetic", 0.8)
    print(f"Immediate transition attempt: {result} (should stay neutral)")
    
    # Wait for minimum duration and try again
    time.sleep(2.1)
    result = smoother.smooth_mood_transition("energetic", 0.8)
    print(f"After minimum duration: {result}, transitioning: {smoother.is_transitioning()}")
    
    # Show transition progress
    while smoother.is_transitioning():
        progress = smoother.get_transition_progress()
        target = smoother.get_target_mood()
        print(f"Transitioning to {target}: {progress:.2f}")
        time.sleep(0.2)
        smoother.smooth_mood_transition("energetic", 0.8)  # Continue feeding same mood
    
    print(f"Final mood: {smoother.get_current_mood()}")
    print()


def demo_confidence_threshold():
    """Demonstrate confidence threshold blocking."""
    print("=== Confidence Threshold Demo ===")
    
    config = MoodConfig()
    config.smoothing = SmoothingConfig(
        transition_time=0.5,
        minimum_duration=0.1,  # Very short for demo
        confidence_threshold=0.7
    )
    
    smoother = MoodTransitionSmoother(config)
    time.sleep(0.2)  # Wait past minimum duration
    
    # Low confidence should be rejected
    result = smoother.smooth_mood_transition("energetic", 0.5)  # Below threshold
    print(f"Low confidence (0.5): {result}, transitioning: {smoother.is_transitioning()}")
    
    # High confidence should work
    result = smoother.smooth_mood_transition("energetic", 0.8)  # Above threshold
    print(f"High confidence (0.8): {result}, transitioning: {smoother.is_transitioning()}")
    
    stats = smoother.get_statistics()
    print(f"Rejected transitions: {stats['rejected_transitions']}")
    print()


def demo_anti_flickering():
    """Demonstrate anti-flickering logic."""
    print("=== Anti-Flickering Demo ===")
    
    config = MoodConfig()
    config.smoothing = SmoothingConfig(
        transition_time=0.5,
        minimum_duration=0.1,
        confidence_threshold=0.7
    )
    
    smoother = MoodTransitionSmoother(config)
    time.sleep(0.2)
    
    # Rapid alternating detections
    moods = ["energetic", "calm", "energetic", "calm", "energetic"]
    for mood in moods:
        result = smoother.smooth_mood_transition(mood, 0.8)
        print(f"Detected {mood}, current: {result}, transitioning: {smoother.is_transitioning()}")
    
    # Show that consistent detections work
    print("\nNow with consistent detections:")
    for _ in range(3):
        result = smoother.smooth_mood_transition("excited", 0.8)
        print(f"Detected excited, current: {result}, transitioning: {smoother.is_transitioning()}")
    
    print()


def demo_statistics():
    """Demonstrate statistics and monitoring."""
    print("=== Statistics Demo ===")
    
    smoother = MoodTransitionSmoother()
    
    # Generate some activity
    time.sleep(5.1)  # Wait past minimum duration
    smoother.smooth_mood_transition("energetic", 0.8)
    smoother.smooth_mood_transition("energetic", 0.8)
    
    # Wait for transition to complete
    time.sleep(2.1)
    smoother.smooth_mood_transition("energetic", 0.8)
    
    # Try some rejected transitions
    smoother.smooth_mood_transition("calm", 0.5)  # Low confidence
    smoother.smooth_mood_transition("excited", 0.8)  # Too soon
    
    stats = smoother.get_statistics()
    print("Current Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    print()


if __name__ == "__main__":
    print("MoodTransitionSmoother Demo")
    print("=" * 40)
    
    demo_basic_transition()
    demo_confidence_threshold()
    demo_anti_flickering()
    demo_statistics()
    
    print("Demo complete!")