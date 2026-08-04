import unittest
from pathlib import Path

from workflows import (
    MOTION_IDENTITY_MIN_STRENGTH,
    build_ltx_motion_workflow,
    build_ltx_motion_workflow_no_vhs,
    duration_to_ltx_frames,
    split_ltx_frame_count,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class MotionControlWorkflowTests(unittest.TestCase):
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
        self.assertEqual(workflow["330"]["class_type"], "LTXAddVideoICLoRAGuide")
        self.assertEqual(workflow["331"]["class_type"], "LTXVCropGuides")
        self.assertEqual(workflow["331"]["inputs"]["latent"], ["215", 0])
        self.assertEqual(workflow["251"]["inputs"]["samples"], ["331", 2])
        self.assertEqual(workflow["269"]["inputs"]["image"], "character.png")
        self.assertEqual(
            workflow["325"]["inputs"]["strength"],
            MOTION_IDENTITY_MIN_STRENGTH,
        )
        self.assertEqual(workflow["330"]["inputs"]["strength"], 1.0)
        self.assertIn("IDENTITY LOCK", workflow["240"]["inputs"]["text"])
        self.assertIn(
            "reference video supplies pose, timing, and motion only",
            workflow["240"]["inputs"]["text"],
        )
        self.assertIn("gender change", workflow["247"]["inputs"]["text"])
        self.assertIn("reference performer appearance", workflow["247"]["inputs"]["text"])

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

    def test_setup_provisions_every_live_dependency(self):
        setup = (REPO_ROOT / "setup.sh").read_text()

        self.assertIn("Lightricks/ComfyUI-LTXVideo", setup)
        self.assertIn("Fannovel16/comfyui_controlnet_aux", setup)
        self.assertIn('kornia==0.6.12', setup)
        self.assertIn(
            "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors",
            setup,
        )


if __name__ == "__main__":
    unittest.main()
