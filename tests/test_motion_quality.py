import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import motion_reference
from motion_reference import (
    MOTION_IDENTITY_MAX_SAMPLED_FRAMES,
    MOTION_IDENTITY_SAMPLE_FPS,
    MOTION_QUALITY_HEIGHT,
    MOTION_QUALITY_SAMPLE_FPS,
    MOTION_QUALITY_WIDTH,
    assess_generated_motion,
    assess_motion_identity,
    summarize_motion_identity,
)


class FakeFaceDetector:
    @staticmethod
    def get_largest_face_embedding(_image_bytes):
        return [1.0, 0.0, 0.0]


class MotionQualityTests(unittest.TestCase):
    def test_identity_sampling_uses_four_fps_and_sixty_frame_cap(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.jpg"
            video = root / "result.mp4"
            source.write_bytes(b"source")
            video.write_bytes(b"video")

            def fake_extract(command, **_kwargs):
                pattern = command[-1]
                for index in range(8):
                    Path(pattern.replace("%03d", f"{index:03d}")).write_bytes(b"frame")
                return subprocess.CompletedProcess(command, 0, b"", b"")

            with mock.patch.object(
                motion_reference.subprocess,
                "run",
                side_effect=fake_extract,
            ) as run:
                result = assess_motion_identity(
                    source,
                    video,
                    face_detector=FakeFaceDetector(),
                    output_dir=root,
                )

            command = run.call_args.args[0]

        self.assertTrue(result["passed"])
        self.assertEqual(result["sample_rate_fps"], MOTION_IDENTITY_SAMPLE_FPS)
        self.assertEqual(result["max_sampled_frames"], MOTION_IDENTITY_MAX_SAMPLED_FRAMES)
        self.assertEqual(result["sampled_frames"], 8)
        self.assertIn(f"fps={MOTION_IDENTITY_SAMPLE_FPS}", command)
        frames_argument = command.index("-frames:v")
        self.assertEqual(
            command[frames_argument + 1],
            str(MOTION_IDENTITY_MAX_SAMPLED_FRAMES),
        )

    def test_identity_threshold_boundaries_remain_unchanged(self):
        exact_boundary = summarize_motion_identity(
            [0.30] * 7 + [0.21] * 3,
            sampled_frames=10,
        )
        below_median = summarize_motion_identity(
            [0.299] * 10,
            sampled_frames=10,
        )
        below_stable_fraction = summarize_motion_identity(
            [0.30] * 6 + [0.21] * 4,
            sampled_frames=10,
        )

        self.assertTrue(exact_boundary["passed"])
        self.assertEqual(exact_boundary["median_similarity"], 0.30)
        self.assertEqual(exact_boundary["stable_frame_fraction"], 0.70)
        self.assertFalse(below_median["passed"])
        self.assertFalse(below_stable_fraction["passed"])

    def test_full_video_motion_assessment_accepts_sustained_motion(self):
        frame_size = MOTION_QUALITY_WIDTH * MOTION_QUALITY_HEIGHT
        raw = bytes([0]) * frame_size + bytes([2]) * frame_size + bytes([4]) * frame_size
        completed = subprocess.CompletedProcess(["ffmpeg"], 0, raw, b"")

        with mock.patch.object(motion_reference.subprocess, "run", return_value=completed) as run:
            result = assess_generated_motion(Path("result.mp4"))

        command = run.call_args.args[0]
        self.assertTrue(result["passed"])
        self.assertEqual(result["sample_rate_fps"], MOTION_QUALITY_SAMPLE_FPS)
        self.assertEqual(result["mean_frame_delta"], 2.0)
        self.assertEqual(result["moving_frame_fraction"], 1.0)
        self.assertTrue(
            any(f"fps={MOTION_QUALITY_SAMPLE_FPS}" in argument for argument in command)
        )

    def test_full_video_motion_assessment_fails_closed_for_static_output(self):
        frame_size = MOTION_QUALITY_WIDTH * MOTION_QUALITY_HEIGHT
        raw = bytes([0]) * frame_size * 4
        completed = subprocess.CompletedProcess(["ffmpeg"], 0, raw, b"")

        with mock.patch.object(motion_reference.subprocess, "run", return_value=completed):
            result = assess_generated_motion(Path("result.mp4"))

        self.assertFalse(result["passed"])
        self.assertEqual(result["mean_frame_delta"], 0.0)
        self.assertEqual(result["moving_frame_fraction"], 0.0)
        self.assertIn("nearly static", result["reason"])

    def test_full_video_motion_assessment_requires_sustained_motion(self):
        frame_size = MOTION_QUALITY_WIDTH * MOTION_QUALITY_HEIGHT
        raw = bytes([0]) * frame_size + bytes([255]) * frame_size * 3
        completed = subprocess.CompletedProcess(["ffmpeg"], 0, raw, b"")

        with mock.patch.object(motion_reference.subprocess, "run", return_value=completed):
            result = assess_generated_motion(Path("result.mp4"))

        self.assertFalse(result["passed"])
        self.assertGreater(result["mean_frame_delta"], 0.75)
        self.assertLess(result["moving_frame_fraction"], 0.70)
        self.assertIn("not sustained", result["reason"])


if __name__ == "__main__":
    unittest.main()
