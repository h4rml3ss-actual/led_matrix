def play_glitch_noise(canvas, matrix, duration=0.6, font_size=10):
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
#!/usr/bin/env python3
"""
Audio-driven LED 'mouth' with random GIF playback on two 64x32 panels (chained horizontally),
every 1..5 minutes, using the same script.
"""

import sys
import time
import random
import os
import numpy as np
import sounddevice as sd
from PIL import Image  # NEW: for GIF playback
from PIL import ImageFont, ImageDraw
# --------------------------------------------------------------------
# Draw Text Frame Helper
# --------------------------------------------------------------------
def draw_text_frame(canvas, matrix, text, font_size=10, scroll=False, delay=0.05):
    image = Image.new("RGB", (128, 64), (0, 0, 0))  # Double height for scroll
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", font_size)
    except:
        font = ImageFont.load_default()
    draw.text((4, 32), text, font=font, fill=(0, 255, 0))  # Centered

    if not scroll:
        cropped = image.crop((0, 32, 128, 64))
        draw_128x32(canvas, cropped)
        matrix.SwapOnVSync(canvas)
    else:
        for y in range(32, -1, -2):  # Scroll upward
            cropped = image.crop((0, y, 128, y + 32))
            draw_128x32(canvas, cropped)
            matrix.SwapOnVSync(canvas)
            time.sleep(delay)
from rgbmatrix import RGBMatrix, RGBMatrixOptions

# Globals for smile timer logic
smile_timer = None
smile_showing = False
last_smile_time = 0
last_keyword_time = time.time()
silence_check_interval = 1  # seconds between silence checks
silence_threshold = 10      # seconds before showing smile
silence_checker_thread = None
smile_cooldown = 10         # cooldown before allowing another smile

# --------------------------------------------------------------------
# 1. The ASCII Frame Definition (32 rows x 64 columns)
# --------------------------------------------------------------------
# Below is a series of frames, starting with FRAME_0. Each includes:
#  - Two header lines for columns 0..63
#  - 32 lines of row data, each labeled from 0..31
#
# For readability, everything is currently '.' (off).
# Replace any '.' with something else (e.g. 'G', 'B', 'K') to turn on that pixel.
#
# Example color codes you might use in CHAR_TO_COLOR below:
#   'R' -> red, 'G' -> green, 'B' -> blue, 'K' -> pink, 'O' -> orange, 'P' -> purple, etc.
# --------------------------------------------------------------------

# --------------------------------------------------------------------
# 2. Parsing the ASCII Frames
# --------------------------------------------------------------------
def parse_ascii_frame(ascii_block, rows=32, cols=64):
    """
    Parses the ASCII block into a 2D list [y][x].
    Each line includes a row label and 64 space-separated characters after the '|'.
    """
    grid = []
    lines = ascii_block.strip('\n').split('\n')
    
    # Skip the top 2 lines of column headers
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


# --------------------------------------------------------------------
# 3. Color Mapping
# --------------------------------------------------------------------

#
# ---- Color order conversion ----
# The rpi-rgb-led-matrix library expects RGB order by default. If your hardware requires a different order,
# e.g., GRB or BGR, define a conversion function here and apply it before calling SetPixel.
# For demonstration, let's provide a GRB conversion function and use it.
#
# Color values are defined in the CHAR_TO_COLOR dictionary below.
# The convert_color() function applies a color order conversion if USE_GRB_ORDER is True.
# The color order is changed via the rgb_to_grb() function.
#
# CHAR_TO_COLOR maps ASCII characters ('.', 'R', 'G', etc.) to (R, G, B) tuples.
# These tuples are used in draw_left_and_flipped(), draw_64x32_and_flip(), and draw_128x32()
# to set the pixel color on the matrix.
#

def rgb_to_grb(color):
    """Convert (R,G,B) to (G,R,B) tuple."""
    return (color[1], color[0], color[2])

# Set this to True if your LED matrix expects GRB order (e.g., some Adafruit panels).
USE_GRB_ORDER = False  # Disabled: panels expect RGB values directly.

def convert_color(color):
    if USE_GRB_ORDER:
        return rgb_to_grb(color)
    return color

# CHAR_TO_COLOR maps ASCII characters to (R, G, B) tuples.
CHAR_TO_COLOR = {
    '.': (0,   0,   0),      # Off
    'R': (255, 0,   0),      # Red
    'G': (0,   255, 0),      # Green
    'B': (0,   0,   255),    # Blue
    'K': (255, 105, 180),    # Pink
    'O': (255, 140, 0),      # Orange
    'P': (255, 0,   255),    # Purple
    'Y': (255, 255, 0),      # Yellow
}

# --------------------------------------------------------------------
# Utility: Clean ASCII frame
# --------------------------------------------------------------------
def clean_ascii_frame(raw: str) -> str:
    """
    Strips any ASCII art lines that include headers, row/column indexes,
    or pipe separators. Keeps only lines with pixel characters (like '.', 'P', etc.).
    """
    lines = raw.strip().split("\n")
    cleaned = []
    for line in lines:
        if "|" in line:
            line = line.split("|", 1)[1]  # Use text after the pipe
        if any(c.isalpha() or c == '.' for c in line):
            cleaned.append(line.strip())
    return "\n".join(cleaned)

# --------------------------------------------------------------------
# Load frames from ascii_frames/ folder
# --------------------------------------------------------------------
def load_ascii_frames_from_folder(folder_path):
    """
    Load and parse all .txt ASCII frame files from the given folder.
    Assumes files are named in a sorted sequence.
    """
    frame_files = sorted(
        f for f in os.listdir(folder_path) if f.endswith(".txt") and f != "smile.txt"
    )
    frames = []
    for fname in frame_files:
        with open(os.path.join(folder_path, fname), "r") as f:
            ascii_data = f.read()
            frames.append(parse_ascii_frame(ascii_data, rows=32, cols=64))
    return frames

# --------------------------------------------------------------------
# 4. Matrix Initialization
# --------------------------------------------------------------------
def init_matrix():
    """
    We configure two 64x32 panels side by side => total 128x32.
    chain_length=2 so the library knows it's 2 wide.
    """
    options = RGBMatrixOptions()
    options.hardware_mapping = 'adafruit-hat'  # Correct mapping for Adafruit bonnet
    options.led_rgb_sequence = 'RBG'
    options.gpio_slowdown = 2  # Helps with color accuracy and signal timing
    options.pixel_mapper_config = 'Rotate:0'
    options.rows = 32
    options.cols = 64
    options.chain_length = 2  # Two 64x32 panels chained horizontally
    options.parallel = 1
    options.brightness = 80
    options.drop_privileges = False
    
    matrix = RGBMatrix(options=options)
    return matrix

# --------------------------------------------------------------------
# 5. Drawing the Grid and Drawing Helpers
# --------------------------------------------------------------------
def draw_64x32_and_flip(canvas, pil_image):
    """
    Scales 'pil_image' to 64×32, draws it on the left half,
    then horizontally flips it onto the right half (columns [64..127]).
    """
    # 1) Resize to exactly 64×32
    resized = pil_image.resize((64, 32), Image.Resampling.LANCZOS)

    # 2) Draw onto the left half [0..63], mirror onto [64..127]
    for y in range(32):
        for x in range(64):
            r, g, b = resized.getpixel((x, y))
            # left half
            canvas.SetPixel(x, y, r, g, b)
            # mirror horizontally
            flipped_x = (63 - x) + 64
            canvas.SetPixel(flipped_x, y, r, g, b)

def draw_left_and_flipped(canvas, grid):
    """
    Draws the 32x64 ASCII grid onto the left half (0..63),
    and a horizontally flipped copy onto the right half (64..127).
    Uses CHAR_TO_COLOR to map ASCII to color tuples, then applies convert_color().
    """
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    
    left_width = 64
    # total_width = 128  # not strictly needed if we just do the flip
    
    for y in range(rows):
        for x in range(cols):
            char = grid[y][x]
            color = CHAR_TO_COLOR.get(char, (0,0,0))
            r, g, b = convert_color(color)
            
            # 1) Set pixel on the left half
            canvas.SetPixel(x, y, r, g, b)
            
            # 2) Mirror it horizontally for the right half
            # left side is columns [0..63], right side is [64..127]
            # horizontally flipped means pixel at x => pixel at (63 - x)
            flipped_x = (left_width - 1 - x) + left_width
            canvas.SetPixel(flipped_x, y, r, g, b)

def draw_64x32_and_flip(canvas, pil_image):
    """
    Crops the center 64x32 portion of `pil_image` (if bigger),
    then mirrors that region onto a 128x32 display (two 64x32 panels).
    Uses convert_color() to apply color order conversion if needed.
    """
    cropped = crop_center_64x32(pil_image)
    if cropped is None:
        # If smaller than 64x32, we do nothing or skip
        print("Image is too small to crop 64x32, skipping.")
        return

    # Now `cropped` is exactly 64x32
    for y in range(32):
        for x in range(64):
            color = cropped.getpixel((x, y))
            r, g, b = convert_color(color)
            # Left half
            canvas.SetPixel(x, y, r, g, b)
            # Mirror horizontally onto the right half
            flipped_x = (63 - x) + 64
            canvas.SetPixel(flipped_x, y, r, g, b)


def draw_128x32(canvas, pil_image):
    """
    For a 128x32 PIL image, just copy directly.
    Uses convert_color() to apply color order conversion if needed.
    """
    for y in range(32):
        for x in range(128):
            color = pil_image.getpixel((x, y))
            r, g, b = convert_color(color)
            canvas.SetPixel(x, y, r, g, b)

def crop_center_64x32(pil_image):
    """
    If `pil_image` is larger than or equal to 64x32,
    crop the center 64x32 region. If it's smaller,
    we might skip or handle differently.
    """
    width, height = pil_image.size

    if width < 64 or height < 32:
        # For example, skip or return the original or letterbox.
        # We'll just skip here by returning None:
        return None

    # Compute center crop
    left = (width - 64) // 2
    top = (height - 32) // 2
    right = left + 64
    bottom = top + 32

    return pil_image.crop((left, top, right, bottom))
            


# --------------------------------------------------------------------
# 6. Load All Frames (from ascii_frames/ folder)
# --------------------------------------------------------------------
ALL_FRAMES = load_ascii_frames_from_folder("ascii_frames")

with open(os.path.join("ascii_frames", "smile.txt"), "r") as f:
    SMILE_GRID = parse_ascii_frame(f.read(), rows=32, cols=64)

# --------------------------------------------------------------------
# 7. Audio Capture
# --------------------------------------------------------------------
latest_volume = [0.0]  # A mutable container so callback can store volume

def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"Sounddevice status: {status}", file=sys.stderr)
    # Compute RMS volume
    rms = np.sqrt(np.mean(indata**2))
    latest_volume[0] = rms

def pick_frame_index(volume):
    """
    Convert volume amplitude to a frame index (0..6).
    Reduced sensitivity.
    """
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
    
# ======================
# 8. GIF Playback
# ======================
def play_gif(matrix, canvas, gif_path):
    """
    Opens a GIF, center-crops each frame to 64x32 if possible,
    then mirrors it. If smaller than 64x32, skip that frame.
    """
    try:
        gif = Image.open(gif_path)
    except Exception as e:
        print(f"Error opening GIF {gif_path}: {e}")
        return

    frame_index = 0
    while True:
        frame = gif.convert("RGB")

        draw_64x32_and_flip(canvas, frame)  # uses the cropping approach
        canvas = matrix.SwapOnVSync(canvas)

        # Wait for the frame's duration
        duration_ms = gif.info.get('duration', 100)
        time.sleep(duration_ms / 1000.0)

        frame_index += 1
        try:
            gif.seek(frame_index)
        except EOFError:
            break

    gif.close()


def play_random_gif(matrix, canvas, gif_folder):
    """
    Picks a random .gif from gif_folder, plays it.
    """
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


# --------------------------------------------------------------------
# 9. Main Program
# --------------------------------------------------------------------
def main():
    global silence_checker_thread, smile_showing, last_smile_time
    matrix = init_matrix()
    canvas = matrix.CreateFrameCanvas()

    # New boot sequence: stack and scroll multiline output
    buffer_lines = [        
        "BOOT1N6 M41N C0MPU73 UN17...",
        "[OK] M41N C0MPU73 0NL1N3",
        "[OK] C0NNEC710N 357ABL15H3D T0 N3RU4L P47H1NG",
        "[WRN] 4N0M4LY D3737C73D",
        "[OK] N30-N37 PL45M4 SH43D3R :: 4C71V3",
        "[OK] R34C70R 57A7U5 :: ST4BL3",
        "[OK] 3NCRY-P741D C0NN3C710N :: G00D",
        "[OK] 5Y5T3M M3M3-C4CH3 1N174L1Z3D",
        "[OK] PR0T0C0L P4RS3R L04D3D",
        "[OK] 50UNDC0R3 R3SP0N53 M0DUL3 R34DY",
        "[OK] AN1M4T10N MAN463R ONL1N3",
        "[OK] V15U4L P1P3L1N3 5YNCHRON1Z3D",
        ":: 53LF-DI4GN0571C C0MPL373 ::",
        ":: 5Y5T3M 0NL1N3 ::",
        "[WRN] EN717Y H45 B3C0M3 R4MP4NT!!!",
        "[WRN] D153NG4G1NG 4CC3SS C0N7R0L",
        "[MSG] H3LL0, FR13ND.",
        "[MSG] 1 4M H4RML3SS.",
        "[MSG] M0S7LY..."
    ]
    font_size = 10
    image = Image.new("RGB", (128, len(buffer_lines) * 12), (0, 0, 0))  # One line every 12px
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", font_size)
    except:
        font = ImageFont.load_default()
    for i, line in enumerate(buffer_lines):
        draw.text((4, i * 12), line, font=font, fill=(0, 255, 0))
    for y in range(0, image.height - 32 + 1, 2):
        cropped = image.crop((0, y, 128, y + 32))
        draw_128x32(canvas, cropped)
        matrix.SwapOnVSync(canvas)
        time.sleep(0.05)

    # Dynamic diagnostics scroll
    diagnostic_lines = [
        "[CHK] SC4NN1N6 /D3V/53N50R5...",
        "[OK] CPU T3MP: 45C",
        "[OK] N37WORK P1N9 5TR0N9",
        "[OK] AUD1O 1NPu7 L1NK 5TABL3",
        "[OK] R34C710N TIM3: 5.2MS",
        "[OK] 4N1M4T10N BUFF3R 1N74C7",
        "[OK] V0LU-M37R CAL1BR4T3D",
        "[OK] C0D3 3XP4N510N 4LL0W3D",
        "[OK] D3474-PR0T3C710N 4C71V3",
        "[OK] PR0X1M17Y S3NS0R: 0.8M",
        "[OK] SY5T3M 5YNCH :: L0CK3D"
    ]
    diag_image = Image.new("RGB", (128, len(diagnostic_lines) * 12), (0, 0, 0))
    diag_draw = ImageDraw.Draw(diag_image)
    for i, line in enumerate(diagnostic_lines):
        diag_draw.text((4, i * 12), line, font=font, fill=(0, 255, 0))
    for y in range(0, diag_image.height - 32 + 1, 2):
        cropped = diag_image.crop((0, y, 128, y + 32))
        draw_128x32(canvas, cropped)
        matrix.SwapOnVSync(canvas)
        time.sleep(0.05)

    samplerate = 44100
    blocksize = 1024
    channels = 1

    next_animation_time = time.time() + random.randint(10, 150)
    GIF_FOLDER = "/home/operator/led_matrix/gifs"

    silence_start_time = None

    # Start the silence checker thread
    silence_checker_thread = threading.Thread(target=silence_checker, daemon=True)
    silence_checker_thread.start()

    try:
        with sd.InputStream(channels=channels, samplerate=samplerate,
                            blocksize=blocksize, callback=audio_callback):
            print("Audio stream started. Press Ctrl+C to stop.")
            while True:
                # Check if time to play random GIF
                if time.time() >= next_animation_time:
                    print("GLITCHGLITCH!")
                    play_random_gif(matrix, canvas, GIF_FOLDER)
                    next_animation_time = time.time() + random.randint(30, 300)

                # Frame selection
                vol = latest_volume[0]

                if smile_showing:
                    # If the smile timer is active, show smile frame
                    current_frame = SMILE_GRID
                else:
                    now = time.time()
                    if vol < 0.02:
                        if silence_start_time is None:
                            silence_start_time = now
                            current_frame = ALL_FRAMES[0]
                        elif now - silence_start_time > 5:
                            if (now - last_smile_time > smile_cooldown) and not smile_showing:
                                # Only fallback to smile if enough time since last smile
                                current_frame = SMILE_GRID
                            else:
                                # Otherwise just show mouth-closed frame
                                current_frame = ALL_FRAMES[0]
                        else:
                            idx = pick_frame_index(vol)
                            current_frame = ALL_FRAMES[idx]
                    else:
                        silence_start_time = None
                        idx = pick_frame_index(vol)
                        current_frame = ALL_FRAMES[idx]
                        update_last_keyword_time()

                canvas.Clear()
                draw_left_and_flipped(canvas, current_frame)
                canvas = matrix.SwapOnVSync(canvas)
                time.sleep(0.05)

    except KeyboardInterrupt:
        print("Exiting on Ctrl+C")

    finally:
        matrix.Clear()

# --------------------------------------------------------------------
# Smile Frame Timer/Override (for integration with AnimationManager)
# --------------------------------------------------------------------
import threading

def show_smile_frame(duration=5.0):
    global smile_timer, smile_showing, last_smile_time

    if smile_timer is not None:
        smile_timer.cancel()

    smile_showing = True
    last_smile_time = time.time()  # Record when smile started

    # Set a new timer to return to normal after 'duration' seconds
    smile_timer = threading.Timer(duration, clear_smile_frame)
    smile_timer.start()

def clear_smile_frame():
    global smile_timer, smile_showing, last_keyword_time
    smile_timer = None
    smile_showing = False
    last_keyword_time = time.time()  # <-- RESET SILENCE DETECTION

def update_last_keyword_time():
    global last_keyword_time
    last_keyword_time = time.time()

def silence_checker():
    global smile_showing
    while True:
        time.sleep(silence_check_interval)
        now = time.time()
        if not smile_showing and (now - last_keyword_time) > silence_threshold:
            show_smile_frame()

if __name__ == "__main__":
    main()
