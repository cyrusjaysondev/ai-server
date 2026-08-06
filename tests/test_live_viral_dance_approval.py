import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


HARNESS_PATH = Path(__file__).with_name("live_viral_dance_approval.py")
SPEC = importlib.util.spec_from_file_location("live_viral_dance_approval", HARNESS_PATH)
assert SPEC and SPEC.loader
HARNESS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = HARNESS
SPEC.loader.exec_module(HARNESS)


class LiveViralDanceApprovalTests(unittest.TestCase):
    PROFILE = {
        "name": "Opalite Dance Challenge",
        "durationSeconds": 3,
        "motionStrength": 0.6,
        "seed": 20260807,
        "referenceStartSeconds": 11,
        "autoSelectMotionWindow": False,
        "motionPrompt": "Preserve identity while following the reference dance.",
    }

    def test_profile_values_are_used_when_cli_has_no_overrides(self):
        settings = HARNESS.resolve_profile_settings(self.PROFILE)

        self.assertEqual(
            settings,
            {
                "duration_seconds": 3.0,
                "motion_strength": 0.6,
                "seed": 20260807,
                "reference_start_seconds": 11.0,
                "auto_select_motion_window": False,
            },
        )

    def test_explicit_cli_values_override_profile_values(self):
        settings = HARNESS.resolve_profile_settings(
            self.PROFILE,
            duration_seconds=5,
            motion_strength=0.9,
            seed=42,
            reference_start_seconds=2.5,
            auto_select_motion_window=True,
        )

        self.assertEqual(
            settings,
            {
                "duration_seconds": 5.0,
                "motion_strength": 0.9,
                "seed": 42,
                "reference_start_seconds": 2.5,
                "auto_select_motion_window": True,
            },
        )

    def test_parser_keeps_motion_window_override_tristate(self):
        parser = HARNESS.build_argument_parser()
        required = [
            "--base-url", "http://pod.test",
            "--source-image", "source.jpg",
            "--templates", "templates.json",
            "--profiles", "profiles.json",
            "--output-dir", "output",
        ]

        self.assertIsNone(parser.parse_args(required).auto_select_motion_window)
        self.assertTrue(
            parser.parse_args(required + ["--auto-select-motion-window"]).auto_select_motion_window
        )
        self.assertFalse(
            parser.parse_args(required + ["--no-auto-select-motion-window"]).auto_select_motion_window
        )

    def test_approval_uses_effective_profile_duration(self):
        final = {
            "identity_quality": {"passed": True},
            "media_duration_seconds": 2.8,
        }
        motion = {"mean_frame_delta": 1.2, "moving_frame_fraction": 0.9}

        approved_for_profile, _ = HARNESS.approval_result(final, motion, 3.0)
        approved_for_wrong_global_default, reasons = HARNESS.approval_result(final, motion, 4.0)

        self.assertTrue(approved_for_profile)
        self.assertFalse(approved_for_wrong_global_default)
        self.assertIn("output is shorter than the approved action window", reasons)

    def test_selected_failure_causes_nonzero_main_result(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source, templates, profiles, output = self._write_inputs(root)
            output.mkdir()
            (output / "report.json").write_text(
                json.dumps(
                    {
                        "motion-dance-opalite": {
                            "automatic_approval": False,
                            "status": "completed",
                        }
                    }
                )
            )

            with mock.patch.object(
                sys,
                "argv",
                self._argv(source, templates, profiles, output),
            ):
                result = HARNESS.main()

        self.assertEqual(result, 1)

    def test_main_submits_profile_defaults_and_records_effective_settings(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source, templates, profiles, output = self._write_inputs(root)
            submission = {
                "job_id": "job-1",
                "poll_url": "/ltx/motion/jobs/job-1",
                "reference_start_seconds": 11,
                "target_frames": 89,
            }
            final = {
                "status": "completed",
                "url": "https://storage.test/result.mp4",
                "media_duration_seconds": 2.8,
                "identity_quality": {"passed": True, "median_similarity": 0.7},
            }

            def fake_download(_url, destination):
                destination.write_bytes(b"test")

            with (
                mock.patch.object(sys, "argv", self._argv(source, templates, profiles, output)),
                mock.patch.object(HARNESS, "download", side_effect=fake_download),
                mock.patch.object(HARNESS, "wait_for_video_capacity"),
                mock.patch.object(HARNESS, "submit_job", return_value=submission) as submit_job,
                mock.patch.object(HARNESS, "poll_job", return_value=final),
                mock.patch.object(
                    HARNESS,
                    "measure_motion",
                    return_value={"mean_frame_delta": 1.2, "moving_frame_fraction": 0.9},
                ),
                mock.patch.object(HARNESS, "create_contact_sheet"),
            ):
                result = HARNESS.main()

            report = json.loads((output / "report.json").read_text())

        self.assertEqual(result, 0)
        submit_job.assert_called_once()
        self.assertEqual(
            submit_job.call_args.kwargs,
            {
                "duration_seconds": 3.0,
                "motion_strength": 0.6,
                "seed": 20260807,
                "reference_start_seconds": 11.0,
                "auto_select_motion_window": False,
            },
        )
        self.assertEqual(
            report["motion-dance-opalite"]["effective_settings"],
            {
                "duration_seconds": 3.0,
                "motion_strength": 0.6,
                "seed": 20260807,
                "reference_start_seconds": 11.0,
                "auto_select_motion_window": False,
            },
        )
        self.assertTrue(report["motion-dance-opalite"]["automatic_approval"])

    def _write_inputs(self, root):
        source = root / "source.jpg"
        templates = root / "templates.json"
        profiles = root / "profiles.json"
        output = root / "output"
        source.write_bytes(b"source")
        templates.write_text(
            json.dumps(
                [
                    {
                        "template_key": "motion-dance-opalite",
                        "sample_video_url": "https://storage.test/reference.mp4",
                    }
                ]
            )
        )
        profiles.write_text(json.dumps({"motion-dance-opalite": self.PROFILE}))
        return source, templates, profiles, output

    @staticmethod
    def _argv(source, templates, profiles, output):
        return [
            str(HARNESS_PATH),
            "--base-url", "http://persistent-pod.test",
            "--source-image", str(source),
            "--templates", str(templates),
            "--profiles", str(profiles),
            "--output-dir", str(output),
        ]


if __name__ == "__main__":
    unittest.main()
