#!/usr/bin/env python3
"""
Enhanced integration for existing led.py - maintains backward compatibility
while adding enhanced mood detection capabilities.

This version updates the original led.py to use enhanced components while
preserving the existing frame structure and animation system.
"""

import sys
import time
import random
import os
import numpy as np
import sounddevice as sd
from PIL import Image, ImageFont, ImageDraw
from rgbmatrix import RGBMatrix, RGBMatrixOptions
import threading

# Enhanced components - graceful fallback if not available
try:
    from enhanced_audio_features import EnhancedFeatureExtractor
    from advanced_mood_detector import AdvancedMoodDetector
    from mood_transition_smoother import MoodTransitionSmoother
    from noise_filter import NoiseFilter
    from user_calibration import get_calibrated_detector
    from mood_config import ConfigManager
    from performance_monitor import get_global_monitor, get_global_scaler
    from error_handling import get_global_error_manager, SafeAudioProcessor, MicrophoneErrorHandler, ErrorSeverity
    ENHANCED_AVAILABLE = True
    print("Enhanced mood detection components loaded successfully")
except ImportError as e:
    print(f"Enhanced components not available: {e}")
    print("Falling back to original mood detection")
    ENHANCED_AVAILABLE = False

# --------------------------------------------------------------------
# Enhanced Audio Processing (with fallback)
# --------------------------------------------------------------------
class AudioProcessor:
    """Audio processor with enhanced capabilities and fallback support."""
    
    def __init__(self, samplerate=44100, blocksize=1024, enable_enhanced=True, user_id=None):
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.enable_enhanced = enable_enhanced and ENHANCED_AVAILABLE
        
        # Storage for audio data and results
        self.latest_volume = [0.0]
        self.latest_audio_block = [None]
        self.latest_mood = "neutral"
        self.latest_confidence = 0.5
        
        # Error handling
        self.error_manager = None
        self.safe_processor = None
        self.microphone_handler = None
        self.consecutive_audio_errors = 0
        self.max_consecutive_audio_errors = 10
        
        # Performance monitoring
        self.performance_monitor = None
        self.performance_scaler = None
        
        if self.enable_enhanced:
            try:
                # Initialize error handling first
                self.error_manager = get_global_error_manager()
                self.safe_processor = SafeAudioProcessor(self.error_manager)
                self.microphone_handler = MicrophoneErrorHandler(self.error_manager)
                
                self._init_enhanced_components(user_id)
                # Initialize performance monitoring
                self.performance_monitor = get_global_monitor()
                self.performance_scaler = get_global_scaler()
                print("Enhanced audio processing with error handling enabled")
            except Exception as e:
                print(f"Failed to initialize enhanced components: {e}")
                self.enable_enhanced = False
                print("Falling back to original audio processing")
        else:
            print("Using original audio processing")
    
    def _init_enhanced_components(self, user_id=None):
        """Initialize enhanced audio processing components."""
        try:
            # Feature extractor with noise filtering
            self.feature_extractor = EnhancedFeatureExtractor(
                samplerate=self.samplerate,
                frame_size=self.blocksize,
                enable_noise_filtering=True
            )
            
            # Mood detector (calibrated if user_id provided)
            if user_id:
                try:
                    self.mood_detector = get_calibrated_detector(user_id)
                    print(f"Using calibrated mood detector for user: {user_id}")
                except Exception as e:
                    print(f"Failed to load calibrated detector: {e}")
                    self.mood_detector = AdvancedMoodDetector()
            else:
                self.mood_detector = AdvancedMoodDetector()
            
            # Transition smoother
            self.transition_smoother = MoodTransitionSmoother()
            
            print("Enhanced audio processing initialized")
            
        except Exception as e:
            print(f"Failed to initialize enhanced components: {e}")
            self.enable_enhanced = False
    
    def audio_callback(self, indata, frames, time_info, status):
        """Enhanced audio callback with comprehensive error handling."""
        try:
            # Handle audio system status
            if status:
                print(f"Sounddevice status: {status}", file=sys.stderr)
                
                # Handle different types of audio errors
                if 'underflow' in str(status).lower():
                    if self.performance_monitor:
                        self.performance_monitor.report_audio_underrun()
                    if self.error_manager:
                        self.error_manager.handle_error(
                            'audio_system', 'buffer_underrun', 
                            Exception(f"Audio underrun: {status}"), ErrorSeverity.LOW
                        )
                elif 'overflow' in str(status).lower():
                    if self.error_manager:
                        self.error_manager.handle_error(
                            'audio_system', 'buffer_overflow',
                            Exception(f"Audio overflow: {status}"), ErrorSeverity.MEDIUM
                        )
                elif 'input' in str(status).lower() and 'underflow' in str(status).lower():
                    # Potential microphone disconnection
                    if self.microphone_handler:
                        self.microphone_handler.handle_microphone_disconnection()
            
            # Validate input data
            if indata is None or len(indata) == 0:
                if self.error_manager:
                    self.error_manager.handle_error(
                        'audio_input', 'no_data',
                        Exception("No audio data received"), ErrorSeverity.MEDIUM
                    )
                # Use silence as fallback
                indata = np.zeros((frames, 1), dtype=np.float32)
            
            # Always compute basic volume for compatibility
            try:
                rms = np.sqrt(np.mean(indata**2))
                # Handle NaN or inf values
                if not np.isfinite(rms):
                    rms = 0.0
                self.latest_volume[0] = rms
                self.latest_audio_block[0] = indata.copy()
            except Exception as e:
                if self.error_manager:
                    self.error_manager.handle_error(
                        'volume_calculation', 'calculation_error', e, ErrorSeverity.LOW
                    )
                self.latest_volume[0] = 0.0
                self.latest_audio_block[0] = np.zeros_like(indata)
            
            # Process audio with error handling
            if self.enable_enhanced:
                try:
                    self._enhanced_processing(indata.flatten())
                    # Reset consecutive error count on success
                    self.consecutive_audio_errors = 0
                except Exception as e:
                    self.consecutive_audio_errors += 1
                    if self.error_manager:
                        severity = ErrorSeverity.HIGH if self.consecutive_audio_errors > 5 else ErrorSeverity.MEDIUM
                        self.error_manager.handle_error(
                            'enhanced_processing', 'processing_error', e, severity
                        )
                    
                    print(f"Enhanced processing failed (attempt {self.consecutive_audio_errors}): {e}")
                    
                    # Disable enhanced processing if too many consecutive errors
                    if self.consecutive_audio_errors >= self.max_consecutive_audio_errors:
                        print("Too many consecutive audio errors, disabling enhanced processing")
                        self.enable_enhanced = False
                    
                    self._fallback_processing(indata.flatten())
            else:
                self._fallback_processing(indata.flatten())
                
        except Exception as e:
            # Critical error in audio callback
            if self.error_manager:
                self.error_manager.handle_error(
                    'audio_callback', 'critical_error', e, ErrorSeverity.CRITICAL
                )
            print(f"Critical error in audio callback: {e}")
            
            # Set safe defaults
            self.latest_volume[0] = 0.0
            self.latest_mood = "neutral"
            self.latest_confidence = 0.3
    
    def _enhanced_processing(self, audio_data):
        """Enhanced audio processing using new components with error handling."""
        if self.performance_monitor:
            self.performance_monitor.start_cycle()
        
        try:
            # Extract enhanced features with error handling
            if self.safe_processor:
                features = self.safe_processor.safe_extract_features(
                    audio_data, self.feature_extractor, timestamp=time.time()
                )
            else:
                features = self.feature_extractor.extract_features(audio_data, timestamp=time.time())
            
            # Detect mood with error handling
            if self.performance_monitor:
                with self.performance_monitor.measure_stage('mood_detection'):
                    if self.safe_processor:
                        mood_result = self.safe_processor.safe_detect_mood(features, self.mood_detector)
                    else:
                        mood_result = self.mood_detector.detect_mood(features)
            else:
                if self.safe_processor:
                    mood_result = self.safe_processor.safe_detect_mood(features, self.mood_detector)
                else:
                    mood_result = self.mood_detector.detect_mood(features)
            
            # Apply transition smoothing with error handling
            try:
                if self.performance_monitor:
                    with self.performance_monitor.measure_stage('transition_smoothing'):
                        smoothed_mood = self.transition_smoother.smooth_transition(
                            mood_result.mood, 
                            mood_result.confidence
                        )
                else:
                    smoothed_mood = self.transition_smoother.smooth_transition(
                        mood_result.mood, 
                        mood_result.confidence
                    )
            except Exception as e:
                if self.error_manager:
                    self.error_manager.handle_error(
                        'transition_smoothing', 'smoothing_error', e, ErrorSeverity.LOW
                    )
                # Use unsmoothed mood as fallback
                smoothed_mood = mood_result.mood
            
            # Update results
            self.latest_mood = smoothed_mood
            self.latest_confidence = mood_result.confidence
            
        except Exception as e:
            # If enhanced processing fails completely, fall back to original
            if self.error_manager:
                self.error_manager.handle_error(
                    'enhanced_processing', 'complete_failure', e, ErrorSeverity.HIGH
                )
            raise  # Re-raise to trigger fallback processing
            
        finally:
            if self.performance_monitor:
                try:
                    metrics = self.performance_monitor.end_cycle()
                    # Check for performance issues and log if needed
                    if metrics.performance_level == 'low':
                        print(f"Performance warning: {metrics.total_duration*1000:.1f}ms processing time")
                    elif metrics.total_cycles % 100 == 0:  # Log every 100 cycles
                        print(f"Performance: {metrics.total_duration*1000:.1f}ms avg, level: {metrics.performance_level}")
                except Exception as e:
                    # Don't let performance monitoring errors crash the system
                    print(f"Performance monitoring error: {e}")
    
    def _fallback_processing(self, audio_data):
        """Fallback to original mood detection."""
        # Use original feature extraction
        rms, zcr, centroid = self.extract_features_original(audio_data)
        
        # Use original mood detection
        mood = self.detect_mood_original(rms, zcr, centroid)
        
        # Update results
        self.latest_mood = mood
        self.latest_confidence = 0.7  # Default confidence for original method
    
    def extract_features_original(self, audio_data):
        """Original feature extraction method."""
        x = audio_data.astype(np.float32)
        
        # RMS energy
        rms = np.sqrt(np.mean(x**2))
        
        # Zero-crossing rate
        zcr = np.sum(np.abs(np.diff(np.sign(x)))) / (2 * len(x))
        
        # Spectral centroid
        freqs = np.fft.rfftfreq(len(x), d=1/self.samplerate)
        mags = np.abs(np.fft.rfft(x))
        centroid = np.sum(freqs * mags) / (np.sum(mags) + 1e-6)
        
        return rms, zcr, centroid
    
    def detect_mood_original(self, rms, zcr, centroid):
        """Original mood detection method."""
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
    
    def get_current_mood(self):
        """Get current mood and confidence."""
        return self.latest_mood, self.latest_confidence
    
    def get_current_volume(self):
        """Get current volume level."""
        return self.latest_volume[0]
    
    def get_performance_summary(self):
        """Get performance monitoring summary."""
        if self.performance_monitor:
            return self.performance_monitor.get_performance_summary()
        return {'status': 'disabled', 'message': 'Performance monitoring not available'}
    
    def save_performance_log(self, filepath='performance_log.json'):
        """Save performance log to file."""
        if self.performance_monitor:
            self.performance_monitor.save_performance_log(filepath)
            return True
        return False
    
    def get_current_performance_level(self):
        """Get current performance level and scale factor."""
        if self.performance_monitor:
            return (self.performance_monitor.current_performance_level, 
                   self.performance_monitor.performance_scale_factor)
        return ('unknown', 1.0)


# --------------------------------------------------------------------
# Original LED System Components (preserved from led.py)
# --------------------------------------------------------------------

# Globals for smile timer logic
smile_timer = None
smile_showing = False
last_smile_time = 0
last_keyword_time = time.time()
silence_check_interval = 1
silence_threshold = 10
silence_checker_thread = None
smile_cooldown = 10

def play_glitch_noise(canvas, matrix, duration=0.6, font_size=10):
    """Original glitch noise function."""
    glitches = [
        "!!##$%@ AUTH B1T5 N07 C0MPL13NT",
        ">>>_GR1D_ERR::R37RY",
        "404: L00P P0IN73R M1551N6",
        "//LOOP[NOISE]::v0.0.1a//",
        ">>> D4T4: 0x7F00B1E5",
        ":: SYNC[FAIL] ::",
        "!!! UPL1NK_0V3R7HR0TTL3D"
    ]
    glitch = random.choice(glitches)
    draw_text_frame(canvas, matrix, glitch, font_size=font_size, scroll=True, delay=0.02)
    time.sleep(duration)

def draw_text_frame(canvas, matrix, text, font_size=10, scroll=False, delay=0.05):
    """Original text drawing function."""
    image = Image.new("RGB", (128, 64), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", font_size)
    except:
        font = ImageFont.load_default()
    draw.text((4, 32), text, font=font, fill=(0, 255, 0))

    if not scroll:
        cropped = image.crop((0, 32, 128, 64))
        draw_128x32(canvas, cropped)
        matrix.SwapOnVSync(canvas)
    else:
        for y in range(32, -1, -2):
            cropped = image.crop((0, y, 128, y + 32))
            draw_128x32(canvas, cropped)
            matrix.SwapOnVSync(canvas)
            time.sleep(delay)

def parse_ascii_frame(ascii_block, rows=32, cols=64):
    """Original ASCII frame parsing."""
    grid = []
    lines = ascii_block.strip('\n').split('\n')
    data_lines = lines[2:]
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

def rgb_to_grb(color):
    """Convert (R,G,B) to (G,R,B) tuple."""
    return (color[1], color[0], color[2])

USE_GRB_ORDER = False

def convert_color(color):
    if USE_GRB_ORDER:
        return rgb_to_grb(color)
    return color

CHAR_TO_COLOR = {
    '.': (0,   0,   0),
    'R': (255, 0,   0),
    'G': (0,   255, 0),
    'B': (0,   0,   255),
    'K': (255, 105, 180),
    'O': (255, 140, 0),
    'P': (255, 0,   255),
    'Y': (255, 255, 0),
}

def load_mood_frames(folder_path):
    """Original mood frame loading."""
    frames = []
    smile = None
    
    if not os.path.exists(folder_path):
        return frames, smile
    
    for fname in sorted(os.listdir(folder_path)):
        if not fname.endswith('.txt'):
            continue
        prefix = fname[:2]
        full_path = os.path.join(folder_path, fname)
        if prefix.upper() == 'XX':
            with open(full_path, 'r') as f:
                smile = parse_ascii_frame(f.read(), rows=32, cols=64)
        else:
            try:
                idx = int(prefix)
            except ValueError:
                continue
            with open(full_path, 'r') as f:
                grid = parse_ascii_frame(f.read(), rows=32, cols=64)
            frames.append((idx, grid))
    
    frames_sorted = [g for (i, g) in sorted(frames, key=lambda x: x[0])]
    return frames_sorted, smile

def init_matrix():
    """Original matrix initialization."""
    options = RGBMatrixOptions()
    options.hardware_mapping = 'adafruit-hat'
    options.led_rgb_sequence = 'RBG'
    options.gpio_slowdown = 2
    options.pixel_mapper_config = 'Rotate:0'
    options.rows = 32
    options.cols = 64
    options.chain_length = 2
    options.parallel = 1
    options.brightness = 80
    options.drop_privileges = False
    
    matrix = RGBMatrix(options=options)
    return matrix

def draw_left_and_flipped(canvas, grid):
    """Original drawing function."""
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    
    left_width = 64
    
    for y in range(rows):
        for x in range(cols):
            char = grid[y][x]
            color = CHAR_TO_COLOR.get(char, (0,0,0))
            r, g, b = convert_color(color)
            
            canvas.SetPixel(x, y, r, g, b)
            
            flipped_x = (left_width - 1 - x) + left_width
            canvas.SetPixel(flipped_x, y, r, g, b)

def draw_128x32(canvas, pil_image):
    """Original PIL image drawing."""
    for y in range(32):
        for x in range(128):
            color = pil_image.getpixel((x, y))
            r, g, b = convert_color(color)
            canvas.SetPixel(x, y, r, g, b)

def pick_frame_index(volume):
    """Original frame index selection."""
    if volume < 0.02:
        return 0
    elif volume < 0.04:
        return 1
    elif volume < 0.06:
        return 2
    elif volume < 0.08:
        return 3
    elif volume < 0.10:
        return 4
    elif volume < 0.12:
        return 5
    else:
        return 6

# Load frame sets
try:
    DEFAULT_FRAMES, DEFAULT_SMILE = load_mood_frames(os.path.join('ascii_frames', 'default'))
except:
    DEFAULT_FRAMES, DEFAULT_SMILE = [], None

MOOD_FRAME_DIRS = {
    'calm': os.path.join('ascii_frames', 'calm'),
    'neutral': os.path.join('ascii_frames', 'neutral'),
    'energetic': os.path.join('ascii_frames', 'energetic'),
    'bright': os.path.join('ascii_frames', 'bright'),
}

MOOD_FRAMES = {}
MOOD_SMILES = {}
for mood, path in MOOD_FRAME_DIRS.items():
    try:
        frames, smile = load_mood_frames(path)
        MOOD_FRAMES[mood] = frames if frames else DEFAULT_FRAMES
        MOOD_SMILES[mood] = smile if smile else DEFAULT_SMILE
    except Exception as e:
        print(f"Warning: loading mood '{mood}' from '{path}' failed: {e}")
        MOOD_FRAMES[mood] = DEFAULT_FRAMES
        MOOD_SMILES[mood] = DEFAULT_SMILE

# GIF playback functions (preserved from original)
def play_gif(matrix, canvas, gif_path):
    """Original GIF playback."""
    try:
        gif = Image.open(gif_path)
    except Exception as e:
        print(f"Error opening GIF {gif_path}: {e}")
        return

    frame_index = 0
    while True:
        frame = gif.convert("RGB")
        # Simplified drawing for compatibility
        draw_128x32(canvas, frame.resize((128, 32)))
        canvas = matrix.SwapOnVSync(canvas)

        duration_ms = gif.info.get('duration', 100)
        time.sleep(duration_ms / 1000.0)

        frame_index += 1
        try:
            gif.seek(frame_index)
        except EOFError:
            break

    gif.close()

def play_random_gif(matrix, canvas, gif_folder):
    """Original random GIF playback."""
    try:
        files = [f for f in os.listdir(gif_folder) if f.lower().endswith('.gif')]
        if not files:
            print("No GIF files found. Skipping.")
            return
        chosen = random.choice(files)
        gif_path = os.path.join(gif_folder, chosen)
        print(f"GLITCHGLITCH!: {gif_path}")
        play_gif(matrix, canvas, gif_path)
    except Exception as e:
        print(f"Error in play_random_gif: {e}")

# Smile frame management
def show_smile_frame(duration=5.0):
    """Original smile frame management."""
    global smile_timer, smile_showing, last_smile_time

    if smile_timer is not None:
        smile_timer.cancel()

    smile_showing = True
    last_smile_time = time.time()

    smile_timer = threading.Timer(duration, clear_smile_frame)
    smile_timer.start()

def clear_smile_frame():
    """Original smile frame clearing."""
    global smile_timer, smile_showing, last_keyword_time
    smile_timer = None
    smile_showing = False
    last_keyword_time = time.time()

def update_last_keyword_time():
    """Original keyword time update."""
    global last_keyword_time
    last_keyword_time = time.time()

def silence_checker():
    """Original silence checker."""
    global smile_showing
    while True:
        time.sleep(silence_check_interval)
        now = time.time()
        if not smile_showing and (now - last_keyword_time) > silence_threshold:
            show_smile_frame()

# --------------------------------------------------------------------
# Enhanced Main Function
# --------------------------------------------------------------------
def main():
    """Enhanced main function with backward compatibility."""
    global silence_checker_thread, smile_showing, last_smile_time
    
    # Check for enhanced mode flag
    enable_enhanced = "--enhanced" in sys.argv or ENHANCED_AVAILABLE
    user_id = None
    
    # Check for user calibration
    if "--user" in sys.argv:
        try:
            user_idx = sys.argv.index("--user") + 1
            if user_idx < len(sys.argv):
                user_id = sys.argv[user_idx]
        except (ValueError, IndexError):
            pass
    
    print("Enhanced LED Mood Detection System")
    print("=" * 40)
    print(f"Enhanced Mode: {'Enabled' if enable_enhanced else 'Disabled'}")
    if user_id:
        print(f"User Calibration: {user_id}")
    print("=" * 40)
    
    # Initialize components
    matrix = init_matrix()
    canvas = matrix.CreateFrameCanvas()
    
    # Initialize audio processor
    audio_processor = AudioProcessor(
        samplerate=44100,
        blocksize=1024,
        enable_enhanced=enable_enhanced,
        user_id=user_id
    )
    
    # Enhanced boot sequence
    if enable_enhanced:
        boot_lines = [
            "ENHANCED MOOD DETECTION v2.0",
            "[OK] ADVANCED FEATURES LOADED",
            "[OK] MULTI-DIMENSIONAL ANALYSIS",
            "[OK] CONFIDENCE-BASED TRANSITIONS",
            "[OK] NOISE FILTERING ACTIVE",
            ":: ENHANCED SYSTEM READY ::"
        ]
    else:
        boot_lines = [
            "STANDARD MOOD DETECTION v1.0",
            "[OK] BASIC FEATURES LOADED",
            "[OK] ORIGINAL ALGORITHM ACTIVE",
            ":: SYSTEM READY ::"
        ]
    
    for line in boot_lines:
        draw_text_frame(canvas, matrix, line, scroll=True, delay=0.03)
        time.sleep(0.1)
    
    # Audio settings
    samplerate = 44100
    blocksize = 1024
    channels = 1

    next_animation_time = time.time() + random.randint(10, 150)
    GIF_FOLDER = "/home/operator/led_matrix/gifs"

    # Start silence checker
    silence_checker_thread = threading.Thread(target=silence_checker, daemon=True)
    silence_checker_thread.start()

    try:
        with sd.InputStream(channels=channels, samplerate=samplerate,
                            blocksize=blocksize, callback=audio_processor.audio_callback):
            print("Enhanced audio stream started. Press Ctrl+C to stop.")
            
            loop_count = 0
            
            while True:
                # Check for random GIF
                if time.time() >= next_animation_time:
                    print("GLITCHGLITCH!")
                    play_random_gif(matrix, canvas, GIF_FOLDER)
                    next_animation_time = time.time() + random.randint(30, 300)

                # Get current mood and volume
                current_mood, confidence = audio_processor.get_current_mood()
                current_volume = audio_processor.get_current_volume()
                
                # Update activity tracking
                if current_volume > 0.01:
                    update_last_keyword_time()

                # Select frame
                if smile_showing:
                    current_frame = MOOD_SMILES.get(current_mood, DEFAULT_SMILE)
                else:
                    idx = pick_frame_index(current_volume)
                    frames_list = MOOD_FRAMES.get(current_mood, DEFAULT_FRAMES)
                    if frames_list:
                        current_frame = frames_list[idx % len(frames_list)]
                    else:
                        current_frame = None

                # Draw frame
                if current_frame:
                    canvas.Clear()
                    draw_left_and_flipped(canvas, current_frame)
                    canvas = matrix.SwapOnVSync(canvas)
                
                # Print status periodically
                if enable_enhanced and loop_count % 100 == 0:
                    print(f"Mood: {current_mood} (conf: {confidence:.3f}) | Vol: {current_volume:.4f}")
                
                loop_count += 1
                time.sleep(0.05)

    except KeyboardInterrupt:
        print("Exiting on Ctrl+C")

    finally:
        matrix.Clear()


if __name__ == "__main__":
    main()