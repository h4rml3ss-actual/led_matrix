# Calibration Data Directory

This directory contains user calibration data for the enhanced mood detection system.

## Files

- `template_calibration.json` - Template showing the structure of calibration files
- `*_calibration.json` - User-specific calibration files (excluded from git)
- `samples/` - Audio samples used for calibration (excluded from git)
- `*.log` - Calibration process logs (excluded from git)

## Usage

User calibration files are automatically generated when running:

```bash
python demo_user_calibration.py --interactive --user "your_name"
```

These files contain personalized voice characteristics and detection thresholds that improve mood detection accuracy for specific users.

## Privacy Note

Personal calibration files are excluded from version control to protect user privacy. Only the template file is included to show the expected data structure.