import unittest
from pathlib import Path

from workflows import (
    MOTION_IDENTITY_MIN_STRENGTH,
    build_ltx_motion_workflow,
    build_ltx_motion_workflow_no_vhs,
    duration_to_ltx_frames,
    motion_pose_is_full_body,
    select_motion_start_seconds,
    split_ltx_frame_count,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class MotionControlWorkflowTests(unittest.TestCase):
    @staticmethod
    def _pose_payload(visible_indices):
        points = [0.0] * (18 * 3)
        for index in visible_indices:
            points[index * 3:index * 3 + 3] = [100.0 + index, 200.0 + index, 1.0]
        return [{"people": [{"pose_keypoints_2d": points}]}]

    def test_full_body_preflight_accepts_both_legs_and_feet(self):
        payload = self._pose_payload(range(18))

        self.assertTrue(motion_pose_is_full_body(payload))

    def test_full_body_preflight_rejects_cropped_or_seated_photo(self):
        payload = self._pose_payload([0, 1, 2, 3, 4, 5, 6, 7, 8])

        self.assertFalse(motion_pose_is_full_body(payload))

    def test_motion_window_skips_quiet_intro_with_short_lead_in(self):
        samples = [(index / 5, 1.5) for index in range(20)]
        samples.extend((4 + index / 5, 6.0) for index in range(20))

        self.assertEqual(
            select_motion_start_seconds(
                samples,
                duration_seconds=12,
                window_seconds=4,
            ),
            3.2,
        )

    def test_continuity_frame_keeps_original_identity_reference(self):
        workflow = build_ltx_motion_workflow(
            reference_video_filename="dance.mp4",
            character_image_filename="continuity.png",
            identity_image_filename="original.png",
            prompt="dance",
            negative_prompt="",
            width=544,
            height=960,
            length=121,
            fps=30,
            seed=42,
        )

        self.assertEqual(workflow["269"]["inputs"]["image"], "continuity.png")
        self.assertEqual(workflow["270"]["inputs"]["image"], "original.png")
        self.assertEqual(workflow["274"]["inputs"]["image"], ["270", 0])

    def test_workflow_contains_required_motion_nodes_and_model(self):
        workflow = build_ltx_motion_workflow(
            reference_video_filename="dance.mp4",
            character_image_filename="character.png",
            prompt="the character dances",
            negative_prompt="deformed anatomy",
            width=544,
            height=960,
            length=121,
            fps=24,
            seed=42,
            inplace_strength=0.5,
            motion_strength=1.0,
        )

        self.assertEqual(workflow["262"]["class_type"], "LTXICLoRALoaderModelOnly")
        self.assertEqual(
            workflow["262"]["inputs"]["lora_name"],
            "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors",
        )
        self.assertEqual(workflow["310"]["class_type"], "VHS_LoadVideo")
        self.assertEqual(workflow["310"]["inputs"]["video"], "dance.mp4")
        self.assertEqual(workflow["320"]["class_type"], "DWPreprocessor")
        self.assertEqual(workflow["320"]["inputs"]["detect_face"], "disable")
        self.assertEqual(workflow["320"]["inputs"]["resolution"], 768)
        self.assertEqual(workflow["330"]["class_type"], "LTXAddVideoICLoRAGuide")
        self.assertEqual(workflow["331"]["class_type"], "LTXVCropGuides")
        self.assertEqual(workflow["331"]["inputs"]["latent"], ["215", 0])
        self.assertEqual(workflow["251"]["inputs"]["samples"], ["331", 2])
        self.assertEqual(workflow["269"]["inputs"]["image"], "character.png")
        self.assertEqual(workflow["324"]["class_type"], "LTXVPreprocess")
        self.assertEqual(workflow["324"]["inputs"]["image"], ["238", 0])
        self.assertEqual(workflow["325"]["inputs"]["image"], ["324", 0])
        self.assertEqual(
            workflow["325"]["inputs"]["strength"],
            MOTION_IDENTITY_MIN_STRENGTH,
        )
        self.assertEqual(workflow["330"]["inputs"]["strength"], 1.0)
        self.assertEqual(workflow["272"]["class_type"], "LoraLoader")
        self.assertEqual(workflow["274"]["class_type"], "TextGenerateLTX2Prompt")
        self.assertEqual(workflow["274"]["inputs"]["image"], ["269", 0])
        self.assertEqual(workflow["240"]["inputs"]["text"], ["274", 0])
        self.assertIn("IDENTITY LOCK", workflow["274"]["inputs"]["prompt"])
        self.assertIn(
            "reference video supplies pose, timing, and motion only",
            workflow["274"]["inputs"]["prompt"],
        )
        self.assertIn("gender change", workflow["247"]["inputs"]["text"])
        self.assertIn("reference performer appearance", workflow["247"]["inputs"]["text"])
        self.assertIn("asymmetrical eyes", workflow["247"]["inputs"]["text"])
        self.assertIn("extra fingers", workflow["247"]["inputs"]["text"])

    def test_quality_motion_uses_two_stage_upscale_and_refine(self):
        workflow = build_ltx_motion_workflow(
            reference_video_filename="dance.mp4",
            character_image_filename="character.png",
            prompt="the character dances",
            negative_prompt="deformed anatomy",
            width=544,
            height=960,
            length=121,
            fps=30,
            seed=42,
            preset="quality",
        )

        self.assertEqual(workflow["228"]["inputs"]["width"], 320)
        self.assertEqual(workflow["228"]["inputs"]["height"], 576)
        self.assertEqual(workflow["322"]["inputs"]["resize_type.width"], 640)
        self.assertEqual(workflow["322"]["inputs"]["resize_type.height"], 1152)
        self.assertEqual(workflow["350"]["class_type"], "LatentUpscaleModelLoader")
        self.assertEqual(workflow["351"]["class_type"], "LTXVLatentUpsampler")
        self.assertNotIn("352", workflow)
        self.assertEqual(workflow["360"]["inputs"]["latent"], ["351", 0])
        self.assertEqual(workflow["360"]["class_type"], "LTXAddVideoICLoRAGuide")
        self.assertEqual(workflow["366"]["class_type"], "LTXVCropGuides")
        self.assertEqual(workflow["251"]["inputs"]["samples"], ["366", 2])
        self.assertEqual(workflow["242"]["inputs"]["crf"], 17)
        self.assertEqual(workflow["274"]["inputs"]["max_length"], 192)

    def test_workflow_uses_safe_canvas_and_fixed_motion_timeline(self):
        workflow = build_ltx_motion_workflow(
            reference_video_filename="dance.mp4",
            character_image_filename="character.png",
            prompt="dance",
            negative_prompt="",
            width=544,
            height=960,
            length=121,
            fps=24,
            seed=7,
        )

        self.assertEqual(workflow["228"]["inputs"]["width"], 576)
        self.assertEqual(workflow["228"]["inputs"]["height"], 960)
        self.assertEqual(workflow["239"]["inputs"]["frame_rate"], 30.0)
        self.assertEqual(workflow["310"]["inputs"]["force_rate"], 30.0)

    def test_frame_loader_fallback_uses_the_same_pose_and_crop_graph(self):
        workflow = build_ltx_motion_workflow_no_vhs(
            reference_frame_filenames=["frame-1.png", "frame-2.png", "frame-3.png"],
            character_image_filename="character.png",
            prompt="dance",
            negative_prompt="",
            width=544,
            height=960,
            length=121,
            fps=30,
            seed=7,
        )

        self.assertNotIn("310", workflow)
        self.assertEqual(workflow["320"]["class_type"], "DWPreprocessor")
        self.assertEqual(workflow["330"]["class_type"], "LTXAddVideoICLoRAGuide")
        self.assertEqual(workflow["331"]["class_type"], "LTXVCropGuides")
        self.assertEqual(workflow["311"]["inputs"]["input"], ["2001", 0])

    def test_fifteen_second_reference_is_split_without_losing_frames(self):
        total_frames = duration_to_ltx_frames(15.0)
        chunks = split_ltx_frame_count(total_frames)

        self.assertEqual(total_frames, 449)
        self.assertEqual(chunks, [121, 121, 121, 89])
        self.assertEqual(sum(chunk - 1 for chunk in chunks) + 1, total_frames)
        self.assertTrue(all((chunk - 1) % 8 == 0 for chunk in chunks))

    def test_motion_route_no_longer_contains_destructive_tail_trim(self):
        main_source = (REPO_ROOT / "main.py").read_text()

        self.assertNotIn("trim_first_half", main_source)
        self.assertNotIn("_MOTION_CLEAN_FRACTION", main_source)
        self.assertIn("match_reference_duration", main_source)
        self.assertIn("MOTION_SEGMENT_ESTIMATE_SECONDS", main_source)
        self.assertIn("Segment {segment} of {total_segments}", main_source)
        self.assertIn("overall_fraction", main_source)

    def test_motion_route_uses_identity_safe_defaults(self):
        main_source = (REPO_ROOT / "main.py").read_text()

        self.assertIn("match_reference_duration: bool = Form(False", main_source)
        self.assertIn("max_duration_seconds: float = Form(4.0", main_source)
        self.assertIn("auto_select_motion_window: bool = Form(True", main_source)

    def test_comfy_history_poll_tolerates_decoder_timeouts(self):
        main_source = (REPO_ROOT / "main.py").read_text()

        self.assertIn("except httpx.RequestError as exc", main_source)
        self.assertIn("transient ComfyUI history poll", main_source)
        self.assertIn("consecutive_poll_errors = 0", main_source)
        self.assertIn("continue", main_source)

    def test_setup_provisions_every_live_dependency(self):
        setup = (REPO_ROOT / "setup.sh").read_text()

        self.assertIn("Lightricks/ComfyUI-LTXVideo", setup)
        self.assertIn("Fannovel16/comfyui_controlnet_aux", setup)
        self.assertIn('kornia==0.6.12', setup)
        self.assertIn(
            "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors",
            setup,
        )
        self.assertIn('?cb=${cache_bust}', setup)

    def test_admin_api_refresh_bypasses_stale_raw_github_cache(self):
        main_source = (REPO_ROOT / "main.py").read_text()

        self.assertIn('?cb={time.time_ns()}', main_source)


if __name__ == "__main__":
    unittest.main()
