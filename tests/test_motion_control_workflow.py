import ast
import unittest
from pathlib import Path

from workflows import build_ltx_motion_workflow


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
        self.assertEqual(workflow["269"]["inputs"]["image"], "character.png")
        self.assertEqual(workflow["325"]["inputs"]["strength"], 0.5)
        self.assertEqual(workflow["330"]["inputs"]["strength"], 1.0)

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

    def test_clean_tail_fraction_stays_at_empirical_boundary(self):
        tree = ast.parse((REPO_ROOT / "main.py").read_text())
        assignments = {
            target.id: ast.literal_eval(node.value)
            for node in tree.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
            and target.id == "_MOTION_CLEAN_FRACTION"
        }

        self.assertEqual(assignments["_MOTION_CLEAN_FRACTION"], 0.40)

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
