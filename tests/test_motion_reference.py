import unittest

from motion_reference import (
    MotionReferenceValidationError,
    motion_reference_extension,
    validate_motion_reference_url,
)


VALID_REFERENCE = (
    "https://syjkzcagphbfzeufqdxf.supabase.co/storage/v1/object/public/"
    "template-assets/motion-templates/viral-dance/opalite-dance-challenge.mp4"
)


class MotionReferenceTests(unittest.TestCase):
    def test_allows_public_supabase_template_asset(self):
        self.assertEqual(validate_motion_reference_url(VALID_REFERENCE), VALID_REFERENCE)
        self.assertEqual(motion_reference_extension(VALID_REFERENCE), ".mp4")

    def test_rejects_non_https_and_non_supabase_hosts(self):
        invalid = [
            VALID_REFERENCE.replace("https://", "http://"),
            VALID_REFERENCE.replace("syjkzcagphbfzeufqdxf.supabase.co", "example.com"),
            VALID_REFERENCE.replace("supabase.co", "supabase.co.example.com"),
        ]
        for value in invalid:
            with self.subTest(value=value):
                with self.assertRaises(MotionReferenceValidationError):
                    validate_motion_reference_url(value)

    def test_rejects_other_buckets_queries_credentials_and_traversal(self):
        invalid = [
            VALID_REFERENCE.replace("template-assets", "uploads"),
            f"{VALID_REFERENCE}?download=1",
            VALID_REFERENCE.replace("https://", "https://user:pass@"),
            VALID_REFERENCE.replace("viral-dance/", "viral-dance/../"),
        ]
        for value in invalid:
            with self.subTest(value=value):
                with self.assertRaises(MotionReferenceValidationError):
                    validate_motion_reference_url(value)

    def test_unknown_extension_falls_back_to_mp4(self):
        url = VALID_REFERENCE.rsplit(".", 1)[0] + ".bin"
        self.assertEqual(motion_reference_extension(url), ".mp4")


if __name__ == "__main__":
    unittest.main()
