# LED Matrix System — Feature Overview

## 🧠 Hardware

- **Controller**: Raspberry Pi Zero 2 W  
- **Display**: Adafruit RGB LED Matrix Bonnet  
- **Panels**: 2x 64x32 LED Matrix Panels  
- **Input**: USB Microphone  

---

## ⚙️ System Considerations

- Runs on **Raspberry Pi OS Lite**
- **Autostarts** when powered on (no manual intervention)

---

## 🎭 Desired Behavior

### 🎙️ Audio-Driven Animation

- Continuously listens via the USB microphone
- Maps **audio volume levels** to **animation frames** to simulate mouth movement
  - Louder audio → higher frame number → wider mouth
  - Quieter audio → lower frame number → smaller mouth
- Gives the illusion of **talking** based on real-time audio input

### 😐 Idle Behavior

- **After 5–10 seconds** of silence (randomized):
  - Displays a **static "smile" frame**
- **After 30–60 seconds** of continuous silence:
  - Plays a **random .gif** from a local `gifs/` directory

---

## ✨ Features (To Implement)

- [ ] Move animation frame data into **separate files** (externalized, modular)
- [ ] Include a **fallback 'smile' frame**
- [ ] Idle mode triggers gif playback from a folder

---

## 🛠️ Eventual Feature State

- GUI-based animation editor:
  - Allows frame design visually
  - Add new animations and assign trigger logic **without code changes**