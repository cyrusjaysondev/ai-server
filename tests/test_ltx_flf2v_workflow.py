import unittest

from workflows import build_ltx_flf2v_workflow


class FirstLastFrameWorkflowTests(unittest.TestCase):
    def test_fast_chains_both_endpoint_guides_into_sampler(self):
        workflow = build_ltx_flf2v_workflow(
            "closed-shirt.png", "open-shirt.png",
            prompt="the person opens the shirt once", negative_prompt="morphing",
            width=544, height=960, length=73, fps=24, seed=42, preset="fast",
        )

        self.assertEqual(workflow["352"]["inputs"]["resize_type.width"], 544)
        self.assertEqual(workflow["352"]["inputs"]["resize_type.height"], 960)
        self.assertEqual(workflow["360"]["inputs"]["image"], ["354", 0])
        self.assertEqual(workflow["361"]["inputs"]["image"], ["355", 0])
        self.assertEqual(workflow["360"]["inputs"]["frame_idx"], 0)
        self.assertEqual(workflow["361"]["inputs"]["frame_idx"], -1)
        self.assertEqual(workflow["361"]["inputs"]["latent"], ["360", 2])
        self.assertEqual(workflow["231"]["inputs"]["positive"], ["361", 0])
        self.assertEqual(workflow["215"]["inputs"]["latent_image"], ["361", 2])
        self.assertEqual(workflow["364"]["inputs"]["latent"], ["215", 0])
        self.assertEqual(workflow["251"]["inputs"]["samples"], ["364", 2])
        self.assertNotIn("362", workflow)

    def test_quality_crops_upscales_and_reapplies_both_guides(self):
        workflow = build_ltx_flf2v_workflow(
            "closed-shirt.png", "open-shirt.png",
            prompt="controlled reveal", negative_prompt="morphing",
            width=720, height=1280, length=121, fps=24, seed=43, preset="quality",
        )

        self.assertEqual(workflow["352"]["inputs"]["resize_type.width"], 352)
        self.assertEqual(workflow["352"]["inputs"]["resize_type.height"], 640)
        self.assertEqual(workflow["356"]["inputs"]["resize_type.width"], 720)
        self.assertEqual(workflow["356"]["inputs"]["resize_type.height"], 1280)
        self.assertEqual(workflow["212"]["inputs"]["positive"], ["361", 0])
        self.assertEqual(workflow["253"]["inputs"]["samples"], ["212", 2])
        self.assertEqual(workflow["362"]["inputs"]["latent"], ["253", 0])
        self.assertEqual(workflow["363"]["inputs"]["latent"], ["362", 2])
        self.assertEqual(workflow["362"]["inputs"]["image"], ["358", 0])
        self.assertEqual(workflow["363"]["inputs"]["image"], ["359", 0])
        self.assertEqual(workflow["213"]["inputs"]["positive"], ["363", 0])
        self.assertEqual(workflow["219"]["inputs"]["latent_image"], ["363", 2])
        self.assertEqual(workflow["364"]["inputs"]["latent"], ["219", 0])
        self.assertEqual(workflow["251"]["inputs"]["samples"], ["364", 2])

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


if __name__ == "__main__":
    unittest.main()
