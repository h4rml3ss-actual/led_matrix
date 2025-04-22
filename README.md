# led_matrix
Pi 02 W and Adafruit RGB Matrix Bonnet code for a Protogen Visor

# LED Matrix Display System for Protogen Visor

This project runs on a Raspberry Pi Zero 2 W and uses the Adafruit RGB Matrix Bonnet to drive a 64x32 LED panel. It powers a Protogen visor display that responds to microphone input, shows animated facial expressions, and supports dynamic frame editing.

---

## 🚀 Features

- 🎙️ **Audio-Reactive Mouth Animation**  
  Visualizes mouth movement by reacting to microphone volume.

- 🎭 **Idle Behavior**  
  Automatically displays a smile or idle animation when no speech is detected.

- 🖼️ **GUI Frame Editor**  
  Easily design 64x32 ASCII frames with a drag-and-draw color grid.

- 🔁 **Frame Import/Export**  
  - Import from PNG images or ASCII text
  - Export to `.txt` for use in code
  - Auto-load all frames from `ascii_frames/` directory

- 🎨 **Color Palette Support**  
  Supports neon-style limited palette: Red, Green, Blue, Pink, Purple, Orange, White, Black.

---

## 🧰 Folder Structure

```
ascii_frames/        # Contains all frame .txt files (one per animation frame)
gifs/                # Optional folder for idle animation gifs
frame_editor.py      # GUI tool to create and edit frames
led.py               # Main animation display engine
cleanup_animations.py# GIF filter/cleanup tool
speech_recognizer.py # Vosk-based recognizer for word-triggered animations
README.md            # You are here
```

---

## 📦 Dependencies

- Python 3.7+
- OpenCV
- Pillow
- sounddevice
- vosk (speech recognition)
- tkinter (GUI)

Install with:

```bash
pip install -r requirements.txt
```

---

## 🧪 Quick Start

1. Connect the RGB matrix panel to the Pi with the Adafruit bonnet.
2. Clone this repo and run:

```bash
python3 led.py
```

3. Use the editor to build new frames:

```bash
python3 frame_editor.py
```

4. Drop new `.txt` files into `ascii_frames/` to make them live.

---

## 🎨 Frame Editor Keyboard Legend

- `R` - Red
- `G` - Green
- `B` - Blue
- `K` - Pink
- `P` - Purple
- `O` - Orange
- `W` - White
- `.` - Off (Black)

---

## ❤️ Maintained by

[@h4rml3ss-actual](https://github.com/h4rml3ss-actual)