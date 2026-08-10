import unittest
from pathlib import Path

from workflows import build_ltx_flf2v_workflow, ltx_end_hold_frame_index


REPO_ROOT = Path(__file__).resolve().parents[1]


class FirstLastFrameWorkflowTests(unittest.TestCase):
    def test_fast_chains_start_hold_and_final_guides_into_sampler(self):
        workflow = build_ltx_flf2v_workflow(
            "closed-shirt.png", "open-shirt.png",
            prompt="the person opens the shirt once", negative_prompt="morphing",
            width=544, height=960, length=73, fps=24, seed=42, preset="fast",
        )

        self.assertEqual(workflow["352"]["inputs"]["resize_type.width"], 544)
        self.assertEqual(workflow["352"]["inputs"]["resize_type.height"], 960)
        self.assertEqual(workflow["360"]["inputs"]["image"], ["354", 0])
        self.assertEqual(workflow["360"]["inputs"]["frame_idx"], 0)
        self.assertEqual(workflow["365"]["inputs"]["image"], ["355", 0])
        self.assertEqual(workflow["365"]["inputs"]["frame_idx"], 48)
        self.assertEqual(workflow["365"]["inputs"]["latent"], ["360", 2])
        self.assertEqual(workflow["365"]["inputs"]["positive"], ["360", 0])
        self.assertEqual(workflow["365"]["inputs"]["negative"], ["360", 1])
        self.assertEqual(workflow["361"]["inputs"]["image"], ["355", 0])
        self.assertEqual(workflow["361"]["inputs"]["frame_idx"], -1)
        self.assertEqual(workflow["361"]["inputs"]["latent"], ["365", 2])
        self.assertEqual(workflow["361"]["inputs"]["positive"], ["365", 0])
        self.assertEqual(workflow["361"]["inputs"]["negative"], ["365", 1])
        self.assertEqual(workflow["231"]["inputs"]["positive"], ["361", 0])
        self.assertEqual(workflow["231"]["inputs"]["negative"], ["361", 1])
        self.assertEqual(workflow["215"]["inputs"]["latent_image"], ["361", 2])
        self.assertEqual(workflow["364"]["inputs"]["positive"], ["361", 0])
        self.assertEqual(workflow["364"]["inputs"]["negative"], ["361", 1])
        self.assertEqual(workflow["364"]["inputs"]["latent"], ["215", 0])
        self.assertEqual(workflow["251"]["inputs"]["samples"], ["364", 2])
        self.assertNotIn("362", workflow)

    def test_realtime_uses_the_same_one_second_end_hold_chain(self):
        workflow = build_ltx_flf2v_workflow(
            "closed-shirt.png", "open-shirt.png", prompt="controlled reveal",
            negative_prompt="morphing", width=544, height=960, length=73,
            fps=24, seed=47, preset="realtime",
        )

        self.assertEqual(workflow["365"]["inputs"]["frame_idx"], 48)
        self.assertEqual(workflow["361"]["inputs"]["frame_idx"], -1)
        self.assertEqual(workflow["361"]["inputs"]["latent"], ["365", 2])
        self.assertEqual(workflow["215"]["inputs"]["latent_image"], ["361", 2])
        self.assertEqual(workflow["364"]["inputs"]["latent"], ["215", 0])

    def test_fast_zero_hold_omits_duplicate_and_chains_final_to_start(self):
        workflow = build_ltx_flf2v_workflow(
            "closed-shirt.png", "open-shirt.png", prompt="continuous reveal",
            negative_prompt="freeze", width=544, height=960, length=97,
            fps=24, seed=48, preset="fast", end_hold_seconds=0,
        )

        self.assertNotIn("365", workflow)
        self.assertEqual(workflow["361"]["inputs"]["frame_idx"], -1)
        self.assertEqual(workflow["361"]["inputs"]["latent"], ["360", 2])
        self.assertEqual(workflow["361"]["inputs"]["positive"], ["360", 0])
        self.assertEqual(workflow["361"]["inputs"]["negative"], ["360", 1])
        self.assertEqual(workflow["231"]["inputs"]["positive"], ["361", 0])
        self.assertEqual(workflow["231"]["inputs"]["negative"], ["361", 1])
        self.assertEqual(workflow["215"]["inputs"]["latent_image"], ["361", 2])
        self.assertEqual(workflow["364"]["inputs"]["positive"], ["361", 0])
        self.assertEqual(workflow["364"]["inputs"]["negative"], ["361", 1])

    def test_realtime_zero_hold_uses_the_same_two_guide_chain(self):
        workflow = build_ltx_flf2v_workflow(
            "closed-shirt.png", "open-shirt.png", prompt="continuous reveal",
            negative_prompt="freeze", width=544, height=960, length=97,
            fps=24, seed=49, preset="realtime", end_hold_seconds=0.0,
        )

        self.assertNotIn("365", workflow)
        self.assertEqual(workflow["361"]["inputs"]["latent"], ["360", 2])
        self.assertEqual(workflow["215"]["inputs"]["latent_image"], ["361", 2])
        self.assertEqual(workflow["364"]["inputs"]["latent"], ["215", 0])

    def test_quality_crops_upscales_and_reapplies_all_three_guides(self):
        workflow = build_ltx_flf2v_workflow(
            "closed-shirt.png", "open-shirt.png",
            prompt="controlled reveal", negative_prompt="morphing",
            width=720, height=1280, length=121, fps=24, seed=43, preset="quality",
        )

        self.assertEqual(workflow["352"]["inputs"]["resize_type.width"], 352)
        self.assertEqual(workflow["352"]["inputs"]["resize_type.height"], 640)
        self.assertEqual(workflow["356"]["inputs"]["resize_type.width"], 720)
        self.assertEqual(workflow["356"]["inputs"]["resize_type.height"], 1280)
        self.assertEqual(workflow["365"]["inputs"]["frame_idx"], 96)
        self.assertEqual(workflow["361"]["inputs"]["latent"], ["365", 2])
        self.assertEqual(workflow["212"]["inputs"]["positive"], ["361", 0])
        self.assertEqual(workflow["212"]["inputs"]["negative"], ["361", 1])
        self.assertEqual(workflow["253"]["inputs"]["samples"], ["212", 2])
        self.assertEqual(workflow["362"]["inputs"]["latent"], ["253", 0])
        self.assertEqual(workflow["362"]["inputs"]["image"], ["358", 0])
        self.assertEqual(workflow["366"]["inputs"]["image"], ["359", 0])
        self.assertEqual(workflow["366"]["inputs"]["frame_idx"], 96)
        self.assertEqual(workflow["366"]["inputs"]["latent"], ["362", 2])
        self.assertEqual(workflow["366"]["inputs"]["positive"], ["362", 0])
        self.assertEqual(workflow["366"]["inputs"]["negative"], ["362", 1])
        self.assertEqual(workflow["363"]["inputs"]["image"], ["359", 0])
        self.assertEqual(workflow["363"]["inputs"]["frame_idx"], -1)
        self.assertEqual(workflow["363"]["inputs"]["latent"], ["366", 2])
        self.assertEqual(workflow["363"]["inputs"]["positive"], ["366", 0])
        self.assertEqual(workflow["363"]["inputs"]["negative"], ["366", 1])
        self.assertEqual(workflow["213"]["inputs"]["positive"], ["363", 0])
        self.assertEqual(workflow["213"]["inputs"]["negative"], ["363", 1])
        self.assertEqual(workflow["219"]["inputs"]["latent_image"], ["363", 2])
        self.assertEqual(workflow["364"]["inputs"]["positive"], ["363", 0])
        self.assertEqual(workflow["364"]["inputs"]["negative"], ["363", 1])
        self.assertEqual(workflow["364"]["inputs"]["latent"], ["219", 0])
        self.assertEqual(workflow["251"]["inputs"]["samples"], ["364", 2])

    def test_quality_zero_hold_omits_both_duplicate_guides(self):
        workflow = build_ltx_flf2v_workflow(
            "closed-shirt.png", "open-shirt.png", prompt="continuous reveal",
            negative_prompt="freeze", width=720, height=1280, length=97,
            fps=24, seed=50, preset="quality", end_hold_seconds=0,
        )

        self.assertNotIn("365", workflow)
        self.assertNotIn("366", workflow)
        self.assertEqual(workflow["361"]["inputs"]["latent"], ["360", 2])
        self.assertEqual(workflow["361"]["inputs"]["positive"], ["360", 0])
        self.assertEqual(workflow["361"]["inputs"]["negative"], ["360", 1])
        self.assertEqual(workflow["212"]["inputs"]["positive"], ["361", 0])
        self.assertEqual(workflow["212"]["inputs"]["negative"], ["361", 1])
        self.assertEqual(workflow["253"]["inputs"]["samples"], ["212", 2])
        self.assertEqual(workflow["363"]["inputs"]["latent"], ["362", 2])
        self.assertEqual(workflow["363"]["inputs"]["positive"], ["362", 0])
        self.assertEqual(workflow["363"]["inputs"]["negative"], ["362", 1])
        self.assertEqual(workflow["213"]["inputs"]["positive"], ["363", 0])
        self.assertEqual(workflow["213"]["inputs"]["negative"], ["363", 1])
        self.assertEqual(workflow["219"]["inputs"]["latent_image"], ["363", 2])
        self.assertEqual(workflow["364"]["inputs"]["positive"], ["363", 0])
        self.assertEqual(workflow["364"]["inputs"]["negative"], ["363", 1])

    def test_audio_crops_only_the_separated_video_latent(self):
        fast = build_ltx_flf2v_workflow(
            "closed-shirt.png", "open-shirt.png", prompt="controlled reveal",
            negative_prompt="morphing", width=544, height=960, length=73,
            fps=24, seed=44, preset="fast", audio=True,
        )
        quality = build_ltx_flf2v_workflow(
            "closed-shirt.png", "open-shirt.png", prompt="controlled reveal",
            negative_prompt="morphing", width=720, height=1280, length=121,
            fps=24, seed=45, preset="quality", audio=True,
        )

        self.assertEqual(fast["364"]["inputs"]["latent"], ["217", 0])
        self.assertEqual(fast["220"]["inputs"]["samples"], ["217", 1])
        self.assertEqual(quality["364"]["inputs"]["latent"], ["218", 0])
        self.assertEqual(quality["220"]["inputs"]["samples"], ["218", 1])

    def test_rejects_invalid_frame_count(self):
        for invalid_length in (1, 72):
            with self.assertRaisesRegex(ValueError, "8n\\+1"):
                build_ltx_flf2v_workflow(
                    "first.png", "last.png", prompt="", negative_prompt="",
                    width=544, height=960, length=invalid_length, fps=24, seed=1,
                )

    def test_custom_end_hold_seconds_and_short_clip_clamping(self):
        self.assertEqual(ltx_end_hold_frame_index(121, 24, 0.5), 108)
        self.assertEqual(ltx_end_hold_frame_index(9, 24, 1.0), 1)
        self.assertIsNone(ltx_end_hold_frame_index(97, 24, 0))

        workflow = build_ltx_flf2v_workflow(
            "first.png", "last.png", prompt="hold", negative_prompt="",
            width=544, height=960, length=121, fps=24, seed=46,
            end_hold_seconds=0.5,
        )
        self.assertEqual(workflow["365"]["inputs"]["frame_idx"], 108)

    def test_negative_end_hold_seconds_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least 0"):
            ltx_end_hold_frame_index(97, 24, -0.01)

    def test_route_exposes_and_forwards_end_hold_seconds(self):
        main_source = (REPO_ROOT / "main.py").read_text()
        route_start = main_source.index('@app.post("/ltx/flf2v")')
        route_end = main_source.index('@app.post("/ltx/t2v")')
        route_source = main_source[route_start:route_end]

        self.assertIn("end_hold_seconds: float = Form(", route_source)
        self.assertIn("ge=0.0", route_source)
        self.assertIn("end_hold_seconds=end_hold_seconds", route_source)
        self.assertIn("keyframes = [0, -1] if end_hold_frame is None", route_source)
        self.assertIn('"end_hold_seconds": end_hold_seconds', route_source)


if __name__ == "__main__":
    unittest.main()
