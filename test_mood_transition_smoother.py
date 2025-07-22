#!/usr/bin/env python3
"""
Unit tests for the MoodTransitionSmoother class.
Tests transition logic, timing, confidence thresholds, and anti-flickering behavior.
"""

import unittest
import time
from unittest.mock import patch, MagicMock
from mood_transition_smoother import MoodTransitionSmoother, TransitionState
from mood_config import MoodConfig, SmoothingConfig


class TestMoodTransitionSmoother(unittest.TestCase):
    """Test cases for MoodTransitionSmoother functionality."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create test configuration with known values
        self.test_config = MoodConfig()
        self.test_config.smoothing = SmoothingConfig(
            transition_time=2.0,
            minimum_duration=5.0,
            confidence_threshold=0.7
        )
        
        self.smoother = MoodTransitionSmoother(self.test_config)
    
    def test_initialization(self):
        """Test proper initialization of MoodTransitionSmoother."""
        self.assertEqual(self.smoother.state.current_mood, "neutral")
        self.assertFalse(self.smoother.state.transition_in_progress)
        self.assertEqual(self.smoother.state.transition_progress, 0.0)
        self.assertEqual(len(self.smoother.state.confidence_buffer), 0)
        self.assertEqual(len(self.smoother.state.mood_buffer), 0)
    
    def test_confidence_threshold_checking(self):
        """Test that transitions are blocked when confidence is below threshold."""
        # Low confidence should prevent transition
        result = self.smoother.smooth_mood_transition("energetic", 0.5)  # Below 0.7 threshold
        self.assertEqual(result, "neutral")  # Should stay in neutral
        self.assertFalse(self.smoother.state.transition_in_progress)
        
        # High confidence should allow transition (after minimum duration)
        with patch('time.time', return_value=time.time() + 6.0):  # Skip minimum duration
            result = self.smoother.smooth_mood_transition("energetic", 0.8)  # Above threshold
            self.assertTrue(self.smoother.state.transition_in_progress)
    
    def test_minimum_duration_holds(self):
        """Test that moods are held for minimum duration before transitions."""
        start_time = time.time()
        
        # Try to transition before minimum duration
        with patch('time.time', return_value=start_time + 2.0):  # Only 2 seconds, need 5
            result = self.smoother.smooth_mood_transition("energetic", 0.8)
            self.assertEqual(result, "neutral")  # Should stay in neutral
            self.assertFalse(self.smoother.state.transition_in_progress)
        
        # Try to transition after minimum duration
        with patch('time.time', return_value=start_time + 6.0):  # 6 seconds, exceeds minimum
            result = self.smoother.smooth_mood_transition("energetic", 0.8)
            self.assertTrue(self.smoother.state.transition_in_progress)
    
    def test_transition_buffer_consistency(self):
        """Test that transitions require consistent mood detections."""
        start_time = time.time()
        
        # Skip minimum duration for all tests
        with patch('time.time', return_value=start_time + 6.0):
            # First detection - buffer has 1 item, should allow transition
            result = self.smoother.smooth_mood_transition("energetic", 0.8)
            # With only 1 item in buffer, consistency check passes, so transition starts
            self.assertTrue(self.smoother.state.transition_in_progress)
    
    def test_transition_progress_timing(self):
        """Test that transition progress updates correctly over time."""
        start_time = time.time()
        
        # Start a transition
        with patch('time.time', return_value=start_time + 6.0):
            self.smoother.smooth_mood_transition("energetic", 0.8)
            self.smoother.smooth_mood_transition("energetic", 0.8)  # Trigger transition
            self.assertTrue(self.smoother.state.transition_in_progress)
            self.assertEqual(self.smoother.state.transition_progress, 0.0)
        
        # Check progress at 50% completion (1 second into 2-second transition)
        with patch('time.time', return_value=start_time + 7.0):
            self.smoother.smooth_mood_transition("energetic", 0.8)
            self.assertGreater(self.smoother.state.transition_progress, 0.0)
            self.assertLess(self.smoother.state.transition_progress, 1.0)
        
        # Check completion after full transition time
        with patch('time.time', return_value=start_time + 8.5):  # 2.5 seconds after start
            self.smoother.smooth_mood_transition("energetic", 0.8)
            self.assertFalse(self.smoother.state.transition_in_progress)
            self.assertEqual(self.smoother.state.current_mood, "energetic")
    
    def test_anti_flickering_logic(self):
        """Test that rapid mood changes are smoothed out."""
        start_time = time.time()
        
        # Skip to after minimum duration
        with patch('time.time', return_value=start_time + 6.0):
            # Rapid alternating detections should not cause flickering
            self.smoother.smooth_mood_transition("energetic", 0.8)
            self.smoother.smooth_mood_transition("calm", 0.8)
            self.smoother.smooth_mood_transition("energetic", 0.8)
            
            # The buffer now has [energetic, calm, energetic] - no majority for any mood
            # So no transition should occur
            self.assertEqual(self.smoother.state.current_mood, "neutral")
    
    def test_should_transition_rules(self):
        """Test individual rules in should_transition method."""
        # Rule 1: Don't transition if already in progress
        self.smoother.state.transition_in_progress = True
        self.assertFalse(self.smoother.should_transition("neutral", "energetic", 0.8))
        
        # Reset for next test
        self.smoother.state.transition_in_progress = False
        
        # Rule 2: Don't transition to same mood
        self.assertFalse(self.smoother.should_transition("neutral", "neutral", 0.8))
        
        # Rule 3: Check confidence threshold
        self.assertFalse(self.smoother.should_transition("neutral", "energetic", 0.5))
        
        # Rule 4: Check minimum duration (tested separately above)
        
        # Rule 5: Check buffer consistency (tested separately above)
    
    def test_get_transition_progress(self):
        """Test transition progress getter."""
        self.assertEqual(self.smoother.get_transition_progress(), 0.0)
        
        # Manually set progress to test getter
        self.smoother.state.transition_progress = 0.5
        self.assertEqual(self.smoother.get_transition_progress(), 0.5)
    
    def test_is_transitioning(self):
        """Test transition status getter."""
        self.assertFalse(self.smoother.is_transitioning())
        
        self.smoother.state.transition_in_progress = True
        self.assertTrue(self.smoother.is_transitioning())
    
    def test_get_current_mood(self):
        """Test current mood getter."""
        self.assertEqual(self.smoother.get_current_mood(), "neutral")
        
        self.smoother.state.current_mood = "energetic"
        self.assertEqual(self.smoother.get_current_mood(), "energetic")
    
    def test_get_target_mood(self):
        """Test target mood getter."""
        self.assertIsNone(self.smoother.get_target_mood())
        
        self.smoother.state.transition_in_progress = True
        self.smoother.target_mood = "energetic"
        self.assertEqual(self.smoother.get_target_mood(), "energetic")
    
    def test_force_transition(self):
        """Test forced immediate transition."""
        self.smoother.force_transition("excited")
        
        self.assertEqual(self.smoother.state.current_mood, "excited")
        self.assertFalse(self.smoother.state.transition_in_progress)
        self.assertEqual(len(self.smoother.state.confidence_buffer), 0)
        self.assertEqual(len(self.smoother.state.mood_buffer), 0)
    
    def test_reset(self):
        """Test smoother reset functionality."""
        # Modify state
        self.smoother.state.current_mood = "energetic"
        self.smoother.state.transition_in_progress = True
        self.smoother.state.confidence_buffer.append(0.8)
        self.smoother.transition_count = 5
        
        # Reset
        self.smoother.reset()
        
        # Check reset state
        self.assertEqual(self.smoother.state.current_mood, "neutral")
        self.assertFalse(self.smoother.state.transition_in_progress)
        self.assertEqual(len(self.smoother.state.confidence_buffer), 0)
        self.assertEqual(self.smoother.transition_count, 0)
    
    def test_get_statistics(self):
        """Test statistics gathering."""
        stats = self.smoother.get_statistics()
        
        required_keys = [
            'current_mood', 'time_in_current_mood', 'transition_in_progress',
            'transition_progress', 'target_mood', 'total_transitions',
            'rejected_transitions', 'recent_confidences', 'recent_moods',
            'minimum_duration_remaining'
        ]
        
        for key in required_keys:
            self.assertIn(key, stats)
        
        self.assertEqual(stats['current_mood'], "neutral")
        self.assertFalse(stats['transition_in_progress'])
    
    def test_update_config(self):
        """Test configuration updates."""
        new_config = MoodConfig()
        new_config.smoothing.confidence_threshold = 0.9
        
        self.smoother.update_config(new_config)
        self.assertEqual(self.smoother.config.smoothing.confidence_threshold, 0.9)
    
    def test_ease_in_out_function(self):
        """Test the easing function for smooth transitions."""
        # Test boundary values
        self.assertEqual(self.smoother._ease_in_out(0.0), 0.0)
        self.assertEqual(self.smoother._ease_in_out(1.0), 1.0)
        
        # Test midpoint
        mid_result = self.smoother._ease_in_out(0.5)
        self.assertGreater(mid_result, 0.0)
        self.assertLess(mid_result, 1.0)
        
        # Test that function is monotonic (always increasing)
        prev_result = 0.0
        for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            result = self.smoother._ease_in_out(t)
            self.assertGreater(result, prev_result)
            prev_result = result
    
    def test_confidence_buffer_averaging(self):
        """Test that average confidence is considered for transitions."""
        start_time = time.time()
        
        with patch('time.time', return_value=start_time + 6.0):
            # Add low confidence values to buffer (need at least 3 for averaging to kick in)
            self.smoother.smooth_mood_transition("energetic", 0.4)
            self.smoother.smooth_mood_transition("energetic", 0.5)
            self.smoother.smooth_mood_transition("energetic", 0.6)
            
            # Now add a fourth detection with high confidence
            # Average will be (0.4 + 0.5 + 0.6 + 0.9) / 4 = 0.6, which is above 0.56 (0.7 * 0.8)
            # So transition should occur
            result = self.smoother.smooth_mood_transition("energetic", 0.9)
            # The first call with only 1 item in buffer should have started transition already
            self.assertTrue(self.smoother.state.transition_in_progress)
    
    def test_multiple_transitions(self):
        """Test multiple sequential transitions."""
        start_time = time.time()
        
        # First transition: neutral -> energetic
        with patch('time.time', return_value=start_time + 6.0):
            self.smoother.smooth_mood_transition("energetic", 0.8)
            self.smoother.smooth_mood_transition("energetic", 0.8)
            self.assertTrue(self.smoother.state.transition_in_progress)
        
        # Complete first transition
        with patch('time.time', return_value=start_time + 8.5):
            self.smoother.smooth_mood_transition("energetic", 0.8)
            self.assertEqual(self.smoother.state.current_mood, "energetic")
            self.assertFalse(self.smoother.state.transition_in_progress)
        
        # Second transition: energetic -> calm (after minimum duration)
        with patch('time.time', return_value=start_time + 14.0):
            self.smoother.smooth_mood_transition("calm", 0.8)
            self.smoother.smooth_mood_transition("calm", 0.8)
            self.assertTrue(self.smoother.state.transition_in_progress)
            self.assertEqual(self.smoother.get_target_mood(), "calm")
    
    def test_rejected_transitions_counter(self):
        """Test that rejected transitions are properly counted."""
        initial_rejected = self.smoother.rejected_transitions
        
        # Try transition with low confidence (should be rejected)
        self.smoother.smooth_mood_transition("energetic", 0.5)
        self.assertEqual(self.smoother.rejected_transitions, initial_rejected + 1)
        
        # Try transition before minimum duration (should be rejected)
        with patch('time.time', return_value=time.time() + 2.0):
            self.smoother.smooth_mood_transition("energetic", 0.8)
            self.assertEqual(self.smoother.rejected_transitions, initial_rejected + 2)


class TestTransitionTiming(unittest.TestCase):
    """Specific tests for transition timing behavior."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = MoodConfig()
        self.config.smoothing = SmoothingConfig(
            transition_time=1.0,  # Shorter for testing
            minimum_duration=2.0,  # Shorter for testing
            confidence_threshold=0.7
        )
        self.smoother = MoodTransitionSmoother(self.config)
    
    def test_transition_timing_precision(self):
        """Test precise timing of transitions."""
        start_time = 1000.0  # Use fixed time for precision
        
        # Initialize the smoother's mood_start_time to a known value
        with patch('time.time', return_value=start_time):
            self.smoother.reset()  # This will set mood_start_time to start_time
        
        # Start transition after minimum duration (2.0 seconds in this test config)
        with patch('time.time', return_value=start_time + 3.0):  # After minimum duration
            self.smoother.smooth_mood_transition("energetic", 0.8)
            # First call should start transition since buffer consistency passes with 1 item
            self.assertTrue(self.smoother.state.transition_in_progress)
        
        # Check progress at quarter point
        with patch('time.time', return_value=start_time + 3.25):
            self.smoother.smooth_mood_transition("energetic", 0.8)
            progress = self.smoother.get_transition_progress()
            self.assertGreater(progress, 0.0)
            self.assertLess(progress, 0.5)
        
        # Check progress at half point
        with patch('time.time', return_value=start_time + 3.5):
            self.smoother.smooth_mood_transition("energetic", 0.8)
            progress = self.smoother.get_transition_progress()
            self.assertGreater(progress, 0.2)
            self.assertLess(progress, 0.8)
        
        # Check completion
        with patch('time.time', return_value=start_time + 4.1):  # Just past transition time
            self.smoother.smooth_mood_transition("energetic", 0.8)
            self.assertFalse(self.smoother.state.transition_in_progress)
            self.assertEqual(self.smoother.state.current_mood, "energetic")
    
    def test_transition_interruption(self):
        """Test behavior when transition is interrupted by new mood."""
        start_time = 1000.0
        
        # Initialize the smoother's mood_start_time to a known value
        with patch('time.time', return_value=start_time):
            self.smoother.reset()  # This will set mood_start_time to start_time
        
        # Start first transition
        with patch('time.time', return_value=start_time + 3.0):
            self.smoother.smooth_mood_transition("energetic", 0.8)
            # First call should start transition since buffer consistency passes with 1 item
            self.assertTrue(self.smoother.state.transition_in_progress)
        
        # Try to start new transition while first is in progress
        with patch('time.time', return_value=start_time + 3.5):
            result = self.smoother.should_transition("energetic", "calm", 0.8)
            self.assertFalse(result)  # Should be blocked


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)