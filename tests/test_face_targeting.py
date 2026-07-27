import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from face_targeting import (
    normalize_target_face_indices,
    preserve_selected_faces,
)


class FaceTargetingTests(unittest.TestCase):
    def test_normalizes_multipart_and_serverless_values(self):
        self.assertEqual(normalize_target_face_indices("1", 1), [1])
        self.assertEqual(normalize_target_face_indices("0,1", 2), [0, 1])
        self.assertEqual(normalize_target_face_indices([1, 0], 2), [1, 0])
        self.assertEqual(normalize_target_face_indices("", 1), [0])

    def test_rejects_missing_duplicate_or_out_of_range_slots(self):
        for value, count in [("0", 2), ("1,1", 2), ("2", 1)]:
            with self.subTest(value=value, count=count):
                with self.assertRaises(ValueError):
                    normalize_target_face_indices(value, count)

    def test_preserves_unselected_person_and_background_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template_path = root / "template.png"
            generated_path = root / "generated.png"

            template = Image.new("RGB", (240, 120), "white")
            template_draw = ImageDraw.Draw(template)
            template_draw.rectangle((35, 35, 75, 75), fill="red")
            template_draw.rectangle((165, 35, 205, 75), fill="green")
            template.save(template_path)

            generated = Image.new("RGB", (240, 120), "black")
            generated_draw = ImageDraw.Draw(generated)
            generated_draw.rectangle((35, 35, 75, 75), fill="blue")
            generated_draw.rectangle((165, 35, 205, 75), fill="yellow")
            generated.save(generated_path)

            preserved, _ = preserve_selected_faces(
                generated_path,
                template_path,
                face_order="left-to-right",
                target_face_indices=[1],
                detect_face_bboxes=lambda _: [
                    (35, 35, 75, 75),
                    (165, 35, 205, 75),
                ],
            )

            self.assertTrue(preserved)
            result = Image.open(generated_path).convert("RGB")
            self.assertEqual(result.getpixel((55, 55)), (255, 0, 0))
            self.assertEqual(result.getpixel((185, 55)), (255, 255, 0))
            self.assertEqual(result.getpixel((120, 110)), (255, 255, 255))


if __name__ == "__main__":
    unittest.main()
