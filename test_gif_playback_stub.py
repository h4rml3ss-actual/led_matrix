"""Test stub to validate EnhancedLEDSystem GIF playback loop logic."""
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("PIL")
from PIL import Image

from enhanced_led import EnhancedLEDSystem, EnhancedSystemConfig


class DummyDisplay:
    """Minimal display stand-in that records rendered frames."""

    def __init__(self, *_args, **_kwargs):
        self.rendered_frames = []

    def draw_pil_image(self, pil_image):
        # Store a copy so content is retained even if the source image changes
        self.rendered_frames.append(pil_image.copy())

    def clear(self):
        self.rendered_frames.clear()


class GifPlaybackTest(unittest.TestCase):
    """Stub test ensuring play_random_gif iterates frames."""

    def test_play_random_gif_uses_all_frames(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gif_path = Path(tmpdir) / "stub.gif"

            # Create a simple 2-frame GIF with explicit durations
            frame_one = Image.new("RGB", (16, 16), (255, 0, 0))
            frame_two = Image.new("RGB", (16, 16), (0, 255, 0))
            frame_one.save(
                gif_path,
                save_all=True,
                append_images=[frame_two],
                duration=[30, 70],
                loop=0,
                format="GIF",
            )

            config = EnhancedSystemConfig()
            config.gif_folder = tmpdir

            with patch("enhanced_led.EnhancedDisplaySystem", DummyDisplay):
                with patch("random.choice", return_value=gif_path):
                    # Avoid real delays during the test stub
                    with patch.object(time, "sleep", return_value=None):
                        system = EnhancedLEDSystem(config)
                        system.play_random_gif()

            self.assertEqual(
                len(system.display.rendered_frames),
                2,
                "All GIF frames should be rendered by play_random_gif",
            )


if __name__ == "__main__":
    unittest.main()
