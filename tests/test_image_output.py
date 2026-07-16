import tempfile
import unittest
import random
from pathlib import Path

from PIL import Image

from image_output import DEFAULT_TARGET_MAX_BYTES, optimize_image_file


class ImageOutputTests(unittest.TestCase):
    def test_large_noise_image_is_compacted_for_delivery(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "render.png"
            pixels = random.Random(42).randbytes(1200 * 1600 * 3)
            Image.frombytes("RGB", (1200, 1600), pixels).save(source, format="PNG")

            result = optimize_image_file(source)

            self.assertEqual(result.path.suffix, ".jpg")
            self.assertTrue(result.path.exists())
            self.assertLessEqual(result.output_bytes, DEFAULT_TARGET_MAX_BYTES)
            self.assertLess(result.output_bytes, result.original_bytes)
            self.assertLessEqual(max(result.width, result.height), 1600)

    def test_small_jpeg_is_not_made_larger(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "small.jpg"
            Image.new("RGB", (320, 320), (30, 80, 120)).save(source, format="JPEG", quality=82)
            original_size = source.stat().st_size

            result = optimize_image_file(source)

            self.assertEqual(result.output_bytes, min(original_size, result.output_bytes))
            self.assertLessEqual(result.output_bytes, DEFAULT_TARGET_MAX_BYTES)


if __name__ == "__main__":
    unittest.main()
