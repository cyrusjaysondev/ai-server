import unittest

import safety


class FaceMatchPolicyTests(unittest.TestCase):
    def test_reported_lookalike_score_is_allowed(self):
        self.assertFalse(safety.is_confident_face_match(
            0.5834, safety.DEFAULT_FACE_FILTER_THRESHOLD
        ))

    def test_high_confidence_blocked_identity_is_rejected(self):
        self.assertTrue(safety.is_confident_face_match(
            0.87, safety.DEFAULT_FACE_FILTER_THRESHOLD
        ))

    def test_threshold_is_strict_at_boundary(self):
        self.assertFalse(safety.is_confident_face_match(
            safety.DEFAULT_FACE_FILTER_THRESHOLD,
            safety.DEFAULT_FACE_FILTER_THRESHOLD,
        ))


if __name__ == "__main__":
    unittest.main()
