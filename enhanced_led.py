#!/usr/bin/env python3
"""
Enhanced audio-driven LED 'mouth' with advanced mood detection system.
Integrates all enhanced components: feature extraction, mood detection, 
transition smoothing, noise filtering, and user calibration.
"""

import sys
import time
import random
import os
import threading
from pathlib import Path
import numpy as np
import sounddevice as sd
from PIL import Image, ImageFont, ImageDraw, ImageSequence
from rgbmatrix import RGBMatrix, RGBMatrixOptions

# Import our enhanced components
from enhanced_audio_features import EnhancedFeatureExtractor
from advanced_mood_detector import AdvancedMoodDetector
from mood_transition_smoother import MoodTransitionSmoother
from noise_filter import NoiseFilter
from user_calibration import UserCalibrator, get_calibrated_detector
from mood_config import ConfigManager

# --------------------------------------------------------------------
# Enhanced System Configuration
# --------------------------------------------------------------------
class EnhancedSystemConfig:
    """Configuration for the enhanced LED system."""
    
    def __init__(self):
        # Audio settings
        self.samplerate = 44100
        self.blocksize = 1024
        self.channels = 1
        
        # Enhanced system settings
        self.enable_noise_filtering = True
        self.enable_user_calibration = False
        self.user_id = None
        self.enable_transition_smoothing = True
        
        # Display settings
        self.frame_rate = 20  # Hz
        self.brightness = 80
        
        # Animation settings
        self.gif_folder = "/home/operator/led_matrix/gifs"
        self.min_animation_interval = 30  # seconds
        self.max_animation_interval = 300  # seconds
        
        # Smile/silence detection
        self.silence_threshold = 10  # seconds
        self.smile_duration = 5.0  # seconds
        self.smile_cooldown = 10  # seconds


# --------------------------------------------------------------------
# Enhanced Audio Processing System
# --------------------------------------------------------------------
class EnhancedAudioProcessor:
    """
    Manages all audio processing with enhanced features.
    """
    
    def __init__(self, config: EnhancedSystemConfig):
        self.config = config
        
        # Initialize enhanced components
        self.feature_extractor = EnhancedFeatureExtractor(
            samplerate=config.samplerate,
            frame_size=config.blocksize,
            enable_noise_filtering=config.enable_noise_filtering
        )
        
        # Initialize mood detector (calibrated or default)
        if config.enable_user_calibration and config.user_id:
            try:
                self.mood_detector = get_calibrated_detector(config.user_id)
                print(f"Using calibrated mood detector for user: {config.user_id}")
            except Exception as e:
                print(f"Failed to load calibrated detector: {e}")
                self.mood_detector = AdvancedMoodDetector()
        else:
            self.mood_detector = AdvancedMoodDetector()
        
        # Initialize transition smoother
        if config.enable_transition_smoothing:
            self.transition_smoother = MoodTransitionSmoother()
        else:
            self.transition_smoother = None
        
        # Audio data storage
        self.latest_audio_block = None
        self.latest_volume = 0.0
        self.latest_mood = "neutral"
        self.latest_confidence = 0.0
        
        # Processing statistics
        self.processing_times = []
        self.feature_history = []
        
    def audio_callback(self, indata, frames, time_info, status):
        """Enhanced audio callback with comprehensive processing."""
        if status:
            print(f"Audio status: {status}", file=sys.stderr)
        
        start_time = time.time()
        
        try:
            # Store raw audio data
            self.latest_audio_block = indata.copy()
            
            # Calculate simple volume for backward compatibility
            self.latest_volume = np.sqrt(np.mean(indata**2))
            
            # Extract enhanced features
            features = self.feature_extractor.extract_features(
                indata.flatten(), 
                timestamp=time.time()
            )
            
            # Store feature history for analysis
            self.feature_history.append(features)
            if len(self.feature_history) > 100:  # Keep last 100 features
                self.feature_history.pop(0)
            
            # Detect mood using advanced detector
            mood_result = self.mood_detector.detect_mood(features)
            
            # Apply transition smoothing if enabled
            if self.transition_smoother:
                final_mood = self.transition_smoother.smooth_transition(
                    mood_result.mood, 
                    mood_result.confidence
                )
            else:
                final_mood = mood_result.mood
            
            # Update latest results
            self.latest_mood = final_mood
            self.latest_confidence = mood_result.confidence
            
            # Track processing time
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            if len(self.processing_times) > 50:
                self.processing_times.pop(0)
            
        except Exception as e:
            print(f"Error in audio processing: {e}")
            # Fallback to simple processing
            self.latest_volume = np.sqrt(np.mean(indata**2))
            self.latest_mood = "neutral"
            self.latest_confidence = 0.5
    
    def get_current_mood(self):
        """Get the current mood with confidence."""
        return self.latest_mood, self.latest_confidence
    
    def get_current_volume(self):
        """Get the current volume level."""
        return self.latest_volume
    
    def get_processing_stats(self):
        """Get processing performance statistics."""
        if not self.processing_times:
            return {"avg_time": 0, "max_time": 0, "min_time": 0}
        
        return {
            "avg_time": np.mean(self.processing_times),
            "max_time": np.max(self.processing_times),
            "min_time": np.min(self.processing_times),
            "sample_count": len(self.processing_times)
        }
    
    def get_noise_filter_info(self):
        """Get noise filter information if available."""
        if hasattr(self.feature_extractor, 'get_noise_filter_info'):
            return self.feature_extractor.get_noise_filter_info()
        return {"noise_filtering_enabled": False}


# --------------------------------------------------------------------
# Frame Management System (from original led.py)
# --------------------------------------------------------------------
def parse_ascii_frame(ascii_block, rows=32, cols=64):
    """Parse ASCII block into a 2D grid."""
    grid = []
    lines = ascii_block.strip('\n').split('\n')
    data_lines = lines[2:]  # Skip header lines
    
    if len(data_lines) < rows:
        raise ValueError(f"ASCII block has fewer than {rows} data lines.")

    for row_index in range(rows):
        line = data_lines[row_index]
        pipe_index = line.find('|')
        if pipe_index == -1:
            raise ValueError(f"No '|' found in line: '{line}'")
        row_data_str = line[pipe_index + 1:].strip()
        row_chars = row_data_str.split()
        if len(row_chars) < cols:
            raise ValueError(f"Row {row_index} doesn't have {cols} columns.")
        row_chars = row_chars[:cols]
        grid.append(row_chars)
    return grid


def load_mood_frames(folder_path):
    """Load frames with two-digit prefixes and mood-specific smile frame."""
    frames = []
    smile = None
    
    if not os.path.exists(folder_path):
        print(f"Warning: Frame folder {folder_path} not found")
        return [], None
    
    for fname in sorted(os.listdir(folder_path)):
        if not fname.endswith('.txt'):
            continue
        prefix = fname[:2]
        full_path = os.path.join(folder_path, fname)
        
        try:
            if prefix.upper() == 'XX':
                # Mood-specific smile
                with open(full_path, 'r') as f:
                    smile = parse_ascii_frame(f.read(), rows=32, cols=64)
            else:
                idx = int(prefix)
                with open(full_path, 'r') as f:
                    grid = parse_ascii_frame(f.read(), rows=32, cols=64)
                frames.append((idx, grid))
        except (ValueError, FileNotFoundError) as e:
            print(f"Warning: Could not load frame {fname}: {e}")
            continue
    
    # Sort frames by numeric prefix
    frames_sorted = [g for (i, g) in sorted(frames, key=lambda x: x[0])]
    return frames_sorted, smile


# --------------------------------------------------------------------
# Enhanced Frame Selection System
# --------------------------------------------------------------------
class EnhancedFrameSelector:
    """
    Enhanced frame selection based on mood and volume with smooth transitions.
    """
    
    def __init__(self):
        # Load mood-specific frame sets
        self.load_frame_sets()
        
        # Frame selection state
        self.current_mood = "neutral"
        self.current_frame_index = 0
        self.frame_transition_smoothing = 0.7  # Smoothing factor for frame changes
        
    def load_frame_sets(self):
        """Load all mood-specific frame sets."""
        # Default frames fallback
        try:
            self.default_frames, self.default_smile = load_mood_frames(
                os.path.join('ascii_frames', 'default')
            )
        except Exception as e:
            print(f"Warning: Could not load default frames: {e}")
            self.default_frames, self.default_smile = [], None
        
        # Mood-specific frame directories
        mood_dirs = {
            'calm': os.path.join('ascii_frames', 'calm'),
            'neutral': os.path.join('ascii_frames', 'neutral'),
            'energetic': os.path.join('ascii_frames', 'energetic'),
            'excited': os.path.join('ascii_frames', 'bright'),  # Map excited to bright
        }
        
        self.mood_frames = {}
        self.mood_smiles = {}
        
        for mood, path in mood_dirs.items():
            try:
                frames, smile = load_mood_frames(path)
                self.mood_frames[mood] = frames if frames else self.default_frames
                self.mood_smiles[mood] = smile if smile else self.default_smile
            except Exception as e:
                print(f"Warning: Loading mood '{mood}' failed: {e}")
                self.mood_frames[mood] = self.default_frames
                self.mood_smiles[mood] = self.default_smile
    
    def select_frame(self, mood: str, volume: float, confidence: float, show_smile: bool = False):
        """
        Select appropriate frame based on mood, volume, and confidence.
        
        Args:
            mood: Current mood ('calm', 'neutral', 'energetic', 'excited')
            volume: Audio volume level
            confidence: Mood detection confidence
            show_smile: Whether to show smile frame
            
        Returns:
            2D grid representing the frame to display
        """
        # Handle smile override
        if show_smile:
            smile_frame = self.mood_smiles.get(mood, self.default_smile)
            if smile_frame:
                return smile_frame
        
        # Get frames for current mood
        frames = self.mood_frames.get(mood, self.default_frames)
        if not frames:
            print(f"Warning: No frames available for mood {mood}")
            return None
        
        # Calculate frame index based on volume with enhanced logic
        frame_index = self.calculate_frame_index(volume, confidence, len(frames))
        
        # Apply smoothing to frame transitions
        if hasattr(self, 'previous_frame_index'):
            smoothed_index = (
                self.frame_transition_smoothing * self.previous_frame_index +
                (1 - self.frame_transition_smoothing) * frame_index
            )
            frame_index = int(round(smoothed_index))
        
        self.previous_frame_index = frame_index
        
        # Ensure frame index is within bounds
        frame_index = max(0, min(frame_index, len(frames) - 1))
        
        return frames[frame_index]
    
    def calculate_frame_index(self, volume: float, confidence: float, num_frames: int):
        """
        Calculate frame index based on volume and confidence.
        Enhanced version of the original pick_frame_index function.
        """
        if num_frames <= 1:
            return 0
        
        # Base frame selection on volume (similar to original)
        if volume < 0.02:
            base_index = 0
        elif volume < 0.04:
            base_index = 1
        elif volume < 0.06:
            base_index = 2
        elif volume < 0.08:
            base_index = 3
        elif volume < 0.10:
            base_index = 4
        elif volume < 0.12:
            base_index = 5
        else:
            base_index = min(6, num_frames - 1)
        
        # Adjust based on confidence
        confidence_factor = max(0.5, confidence)  # Minimum 50% confidence
        adjusted_index = int(base_index * confidence_factor)
        
        # Ensure within bounds
        return max(0, min(adjusted_index, num_frames - 1))


# --------------------------------------------------------------------
# Display System (from original led.py with enhancements)
# --------------------------------------------------------------------
class EnhancedDisplaySystem:
    """Enhanced display system with performance monitoring."""
    
    def __init__(self, config: EnhancedSystemConfig):
        self.config = config
        self.matrix = self.init_matrix()
        self.canvas = self.matrix.CreateFrameCanvas()
        
        # Color mapping (from original)
        self.char_to_color = {
            '.': (0,   0,   0),      # Off
            'R': (255, 0,   0),      # Red
            'G': (0,   255, 0),      # Green
            'B': (0,   0,   255),    # Blue
            'K': (255, 105, 180),    # Pink
            'O': (255, 140, 0),      # Orange
            'P': (255, 0,   255),    # Purple
            'Y': (255, 255, 0),      # Yellow
        }
        
        # Performance tracking
        self.frame_times = []
        self.frames_rendered = 0
        
    def init_matrix(self):
        """Initialize the RGB matrix with enhanced settings."""
        options = RGBMatrixOptions()
        options.hardware_mapping = 'adafruit-hat'
        options.led_rgb_sequence = 'RBG'
        options.gpio_slowdown = 2
        options.pixel_mapper_config = 'Rotate:0'
        options.rows = 32
        options.cols = 64
        options.chain_length = 2  # Two 64x32 panels chained horizontally
        options.parallel = 1
        options.brightness = self.config.brightness
        options.drop_privileges = False
        
        return RGBMatrix(options=options)
    
    def draw_frame(self, grid):
        """
        Draw a frame grid to the display with mirroring.
        Enhanced version of draw_left_and_flipped.
        """
        if not grid:
            return
        
        start_time = time.time()
        
        rows = len(grid)
        cols = len(grid[0]) if rows > 0 else 0
        
        self.canvas.Clear()
        
        for y in range(min(rows, 32)):  # Ensure we don't exceed display height
            for x in range(min(cols, 64)):  # Ensure we don't exceed half width
                char = grid[y][x]
                color = self.char_to_color.get(char, (0, 0, 0))
                r, g, b = color
                
                # Left half
                self.canvas.SetPixel(x, y, r, g, b)
                
                # Mirror horizontally for right half
                flipped_x = (63 - x) + 64
                self.canvas.SetPixel(flipped_x, y, r, g, b)
        
        self.canvas = self.matrix.SwapOnVSync(self.canvas)
        
        # Track performance
        frame_time = time.time() - start_time
        self.frame_times.append(frame_time)
        if len(self.frame_times) > 100:
            self.frame_times.pop(0)
        
        self.frames_rendered += 1
    
    def draw_text_frame(self, text, font_size=10, scroll=False, delay=0.05):
        """Draw text frame with scrolling support."""
        image = Image.new("RGB", (128, 64), (0, 0, 0))
        draw = ImageDraw.Draw(image)
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        draw.text((4, 32), text, font=font, fill=(0, 255, 0))
        
        if not scroll:
            cropped = image.crop((0, 32, 128, 64))
            self.draw_pil_image(cropped)
        else:
            for y in range(32, -1, -2):
                cropped = image.crop((0, y, 128, y + 32))
                self.draw_pil_image(cropped)
                time.sleep(delay)
    
    def draw_pil_image(self, pil_image):
        """Draw a PIL image directly to the display."""
        for y in range(min(32, pil_image.height)):
            for x in range(min(128, pil_image.width)):
                color = pil_image.getpixel((x, y))
                if len(color) >= 3:
                    r, g, b = color[:3]
                    self.canvas.SetPixel(x, y, r, g, b)
        
        self.canvas = self.matrix.SwapOnVSync(self.canvas)
    
    def get_performance_stats(self):
        """Get display performance statistics."""
        if not self.frame_times:
            return {"avg_frame_time": 0, "fps": 0, "frames_rendered": 0}
        
        avg_frame_time = np.mean(self.frame_times)
        fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
        
        return {
            "avg_frame_time": avg_frame_time,
            "fps": fps,
            "frames_rendered": self.frames_rendered,
            "max_frame_time": np.max(self.frame_times),
            "min_frame_time": np.min(self.frame_times)
        }
    
    def clear(self):
        """Clear the display."""
        self.matrix.Clear()


# --------------------------------------------------------------------
# Enhanced Main System
# --------------------------------------------------------------------
class EnhancedLEDSystem:
    """
    Main enhanced LED system that coordinates all components.
    """
    
    def __init__(self, config: EnhancedSystemConfig):
        self.config = config
        
        # Initialize subsystems
        self.audio_processor = EnhancedAudioProcessor(config)
        self.frame_selector = EnhancedFrameSelector()
        self.display = EnhancedDisplaySystem(config)
        
        # Smile/animation state
        self.smile_showing = False
        self.smile_timer = None
        self.last_keyword_time = time.time()
        self.next_animation_time = time.time() + random.randint(
            config.min_animation_interval, 
            config.max_animation_interval
        )
        
        # Performance monitoring
        self.start_time = time.time()
        self.loop_count = 0
        
        # Silence checker thread
        self.silence_checker_thread = None
        self.running = False
    
    def show_boot_sequence(self):
        """Display enhanced boot sequence."""
        boot_lines = [
            "ENHANCED MOOD DETECTION SYSTEM v2.0",
            "[OK] ENHANCED FEATURE EXTRACTOR LOADED",
            "[OK] ADVANCED MOOD DETECTOR INITIALIZED",
            "[OK] TRANSITION SMOOTHER ACTIVE",
            "[OK] NOISE FILTERING ENABLED" if self.config.enable_noise_filtering else "[--] NOISE FILTERING DISABLED",
            "[OK] USER CALIBRATION READY" if self.config.enable_user_calibration else "[--] USER CALIBRATION DISABLED",
            "[OK] PERFORMANCE MONITORING ACTIVE",
            "[OK] MULTI-DIMENSIONAL MOOD ANALYSIS",
            "[OK] CONFIDENCE-BASED TRANSITIONS",
            "[OK] ADAPTIVE THRESHOLD SYSTEM",
            ":: ENHANCED SYSTEM ONLINE ::",
            ":: READY FOR AUDIO INPUT ::",
        ]
        
        # Add user-specific message if calibrated
        if self.config.enable_user_calibration and self.config.user_id:
            boot_lines.insert(-2, f"[OK] CALIBRATED FOR USER: {self.config.user_id}")
        
        for line in boot_lines:
            self.display.draw_text_frame(line, scroll=True, delay=0.03)
            time.sleep(0.1)
    
    def start_silence_checker(self):
        """Start the silence checker thread."""
        self.silence_checker_thread = threading.Thread(target=self.silence_checker, daemon=True)
        self.silence_checker_thread.start()
    
    def silence_checker(self):
        """Check for silence and show smile when appropriate."""
        while self.running:
            time.sleep(1)  # Check every second
            
            if not self.smile_showing:
                silence_duration = time.time() - self.last_keyword_time
                if silence_duration > self.config.silence_threshold:
                    self.show_smile_frame()
    
    def show_smile_frame(self, duration=None):
        """Show smile frame for specified duration."""
        if duration is None:
            duration = self.config.smile_duration
        
        if self.smile_timer:
            self.smile_timer.cancel()
        
        self.smile_showing = True
        self.smile_timer = threading.Timer(duration, self.clear_smile_frame)
        self.smile_timer.start()
    
    def clear_smile_frame(self):
        """Clear smile frame and return to normal display."""
        self.smile_timer = None
        self.smile_showing = False
        self.last_keyword_time = time.time()
    
    def play_random_gif(self):
        """Play a random GIF animation using the configured GIF folder."""
        gif_dir = Path(self.config.gif_folder)

        if not gif_dir.exists():
            print(f"GIF folder not found: {gif_dir}")
            return

        gif_files = [p for p in gif_dir.iterdir() if p.suffix.lower() == ".gif"]
        if not gif_files:
            print(f"No GIF files found in {gif_dir}")
            return

        gif_path = random.choice(gif_files)
        print(f"Starting GIF playback: {gif_path}")

        try:
            with Image.open(gif_path) as gif:
                frames = []
                durations = []

                for frame in ImageSequence.Iterator(gif):
                    frames.append(frame.convert("RGB").resize((128, 32)))
                    durations.append(max(frame.info.get("duration", 100) / 1000.0, 0.01))

                if not frames:
                    raise ValueError("GIF contained no frames")

                for frame, duration in zip(frames, durations):
                    self.display.draw_pil_image(frame)
                    time.sleep(duration)

            print(f"Finished GIF playback: {gif_path}")

        except FileNotFoundError:
            print(f"GIF file not found: {gif_path}")
        except Exception as e:
            print(f"Error playing GIF {gif_path}: {e}")
    
    def print_status(self):
        """Print system status information."""
        mood, confidence = self.audio_processor.get_current_mood()
        volume = self.audio_processor.get_current_volume()
        
        audio_stats = self.audio_processor.get_processing_stats()
        display_stats = self.display.get_performance_stats()
        noise_info = self.audio_processor.get_noise_filter_info()
        
        uptime = time.time() - self.start_time
        
        print(f"\n=== Enhanced LED System Status ===")
        print(f"Uptime: {uptime:.1f}s | Loops: {self.loop_count}")
        print(f"Current Mood: {mood} (confidence: {confidence:.3f})")
        print(f"Volume: {volume:.4f} | Smile: {self.smile_showing}")
        print(f"Audio Processing: {audio_stats['avg_time']*1000:.1f}ms avg")
        print(f"Display FPS: {display_stats['fps']:.1f}")
        
        if noise_info.get('noise_filtering_enabled'):
            print(f"Noise Filter: Gain={noise_info.get('current_gain', 0):.2f}")
        
        print("=" * 35)
    
    def run(self):
        """Main system loop."""
        print("Starting Enhanced LED Mood Detection System...")
        
        # Show boot sequence
        self.show_boot_sequence()
        
        # Start silence checker
        self.running = True
        self.start_silence_checker()
        
        try:
            # Start audio stream
            with sd.InputStream(
                channels=self.config.channels,
                samplerate=self.config.samplerate,
                blocksize=self.config.blocksize,
                callback=self.audio_processor.audio_callback
            ):
                print("Enhanced audio stream started. Press Ctrl+C to stop.")
                
                status_interval = 50  # Print status every 50 loops
                
                while True:
                    # Check for random animation
                    if time.time() >= self.next_animation_time:
                        self.play_random_gif()
                        self.next_animation_time = time.time() + random.randint(
                            self.config.min_animation_interval,
                            self.config.max_animation_interval
                        )
                    
                    # Get current mood and volume
                    mood, confidence = self.audio_processor.get_current_mood()
                    volume = self.audio_processor.get_current_volume()
                    
                    # Update last keyword time if there's significant audio
                    if volume > 0.01:  # Threshold for "activity"
                        self.last_keyword_time = time.time()
                    
                    # Select and display frame
                    frame = self.frame_selector.select_frame(
                        mood=mood,
                        volume=volume,
                        confidence=confidence,
                        show_smile=self.smile_showing
                    )
                    
                    if frame:
                        self.display.draw_frame(frame)
                    
                    # Print status periodically
                    if self.loop_count % status_interval == 0:
                        self.print_status()
                    
                    self.loop_count += 1
                    time.sleep(1.0 / self.config.frame_rate)
        
        except KeyboardInterrupt:
            print("\nShutting down Enhanced LED System...")
        
        finally:
            self.running = False
            if self.smile_timer:
                self.smile_timer.cancel()
            self.display.clear()


# --------------------------------------------------------------------
# Configuration and Startup
# --------------------------------------------------------------------
def load_system_config():
    """Load system configuration from command line args or config file."""
    config = EnhancedSystemConfig()
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        if "--calibrated" in sys.argv:
            config.enable_user_calibration = True
            # Look for user ID
            try:
                user_idx = sys.argv.index("--user") + 1
                if user_idx < len(sys.argv):
                    config.user_id = sys.argv[user_idx]
            except ValueError:
                config.user_id = "default_user"
        
        if "--no-noise-filter" in sys.argv:
            config.enable_noise_filtering = False
        
        if "--no-smoothing" in sys.argv:
            config.enable_transition_smoothing = False
        
        if "--brightness" in sys.argv:
            try:
                brightness_idx = sys.argv.index("--brightness") + 1
                if brightness_idx < len(sys.argv):
                    config.brightness = int(sys.argv[brightness_idx])
            except (ValueError, IndexError):
                pass
    
    return config


def main():
    """Main entry point."""
    # Load configuration
    config = load_system_config()
    
    print("Enhanced LED Mood Detection System")
    print("=" * 40)
    print(f"Noise Filtering: {'Enabled' if config.enable_noise_filtering else 'Disabled'}")
    print(f"User Calibration: {'Enabled' if config.enable_user_calibration else 'Disabled'}")
    if config.user_id:
        print(f"User ID: {config.user_id}")
    print(f"Transition Smoothing: {'Enabled' if config.enable_transition_smoothing else 'Disabled'}")
    print(f"Frame Rate: {config.frame_rate} Hz")
    print(f"Brightness: {config.brightness}")
    print("=" * 40)
    
    # Create and run the enhanced system
    system = EnhancedLEDSystem(config)
    system.run()


if __name__ == "__main__":
    main()