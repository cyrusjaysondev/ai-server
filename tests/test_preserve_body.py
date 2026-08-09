import tempfile
import unittest
from pathlib import Path

from PIL import Image

from workflows import (
    PRESERVE_BODY_NECK_EXTENSION_RATIO,
    preserve_body_head,
)


class PreserveBodyHeadTests(unittest.TestCase):
    BBOX = (80, 70, 160, 150)

    def _run_composite(self, directory: str):
        root = Path(directory)
        template_path = root / "template.png"
        generated_path = root / "generated.png"
        template = Image.new("RGB", (240, 320), (10, 40, 220))
        generated = Image.new("RGB", (240, 320), (230, 30, 20))
        template.save(template_path)
        generated.save(generated_path)

        preserved, message = preserve_body_head(
            generated_path,
            template_path,
            detect_face_bbox=lambda _: self.BBOX,
        )
        return preserved, message, template_path, generated_path

    def test_pixels_below_neck_cutoff_remain_template_identical(self):
        with tempfile.TemporaryDirectory() as directory:
            preserved, message, template_path, generated_path = self._run_composite(directory)

            self.assertTrue(preserved, message)
            _, y1, _, y2 = self.BBOX
            face_height = y2 - y1
            cutoff = round(y2 + face_height * PRESERVE_BODY_NECK_EXTENSION_RATIO)
            template = Image.open(template_path).convert("RGB")
            result = Image.open(generated_path).convert("RGB")
            protected_box = (0, cutoff, result.width, result.height)
            self.assertEqual(
                result.crop(protected_box).tobytes(),
                template.crop(protected_box).tobytes(),
            )

    def test_head_is_composited_while_background_stays_template(self):
        with tempfile.TemporaryDirectory() as directory:
            preserved, message, template_path, generated_path = self._run_composite(directory)

            self.assertTrue(preserved, message)
            template = Image.open(template_path).convert("RGB")
            result = Image.open(generated_path).convert("RGB")
            self.assertEqual(result.getpixel((120, 110)), (230, 30, 20))
            self.assertEqual(result.getpixel((10, 10)), template.getpixel((10, 10)))


if __name__ == "__main__":
    unittest.main()
