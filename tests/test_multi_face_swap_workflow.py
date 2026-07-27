import unittest

from workflows import (
    build_flux_multi_face_swap_workflow,
    build_multi_face_swap_prompt,
)


class MultiFaceSwapWorkflowTests(unittest.TestCase):
    def test_one_face_maps_only_first_person(self):
        prompt = build_multi_face_swap_prompt(1, "right-to-left")

        self.assertIn("right to left", prompt)
        self.assertIn("Replace only the first person's head and face", prompt)
        self.assertIn("Do not change the identity, face, or hair of any other person", prompt)
        self.assertNotIn("image 3", prompt)

    def test_one_face_can_target_second_person(self):
        prompt = build_multi_face_swap_prompt(
            1,
            "left-to-right",
            target_face_indices=[1],
        )

        self.assertIn("Replace only the second person's head and face", prompt)
        self.assertIn("identity from image 2", prompt)
        self.assertIn("Do not change the identity, face, or hair of any other person", prompt)
        self.assertIn("preserve that angle exactly", prompt)
        self.assertIn("Do not frontalize, mirror", prompt)

    def test_two_faces_have_separate_ordered_identity_mapping(self):
        prompt = build_multi_face_swap_prompt(
            2,
            "top-to-bottom",
            "Keep the wedding veil exactly unchanged.",
        )

        self.assertIn("top to bottom", prompt)
        self.assertIn("first person's head and face", prompt)
        self.assertIn("identity from image 2", prompt)
        self.assertIn("second person's head and face", prompt)
        self.assertIn("identity from image 3", prompt)
        self.assertIn("never blend, average, merge, or swap them", prompt)
        self.assertIn("Keep the wedding veil exactly unchanged.", prompt)

    def test_rejects_invalid_face_count_and_order(self):
        with self.assertRaisesRegex(ValueError, "requires 1 or 2"):
            build_multi_face_swap_prompt(0)
        with self.assertRaisesRegex(ValueError, "requires 1 or 2"):
            build_multi_face_swap_prompt(3)
        with self.assertRaisesRegex(ValueError, "invalid face_order"):
            build_multi_face_swap_prompt(1, "random")

    def test_workflow_chains_template_and_two_face_references(self):
        workflow = build_flux_multi_face_swap_workflow(
            "couple-template.png",
            ["face-a.png", "face-b.png"],
            seed=42,
            face_order="left-to-right",
            megapixels=1.0,
            target_face_indices=[0, 1],
        )

        self.assertEqual(workflow["200"]["inputs"]["image"], "couple-template.png")
        self.assertEqual(workflow["201"]["inputs"]["image"], "face-a.png")
        self.assertEqual(workflow["202"]["inputs"]["image"], "face-b.png")
        self.assertEqual(workflow["230"]["inputs"]["conditioning"], ["10", 0])
        self.assertEqual(workflow["231"]["inputs"]["conditioning"], ["230", 0])
        self.assertEqual(workflow["232"]["inputs"]["conditioning"], ["231", 0])
        self.assertEqual(workflow["30"]["inputs"]["conditioning"], ["232", 0])
        self.assertEqual(workflow["4"]["class_type"], "LoraLoaderModelOnly")
        self.assertEqual(
            workflow["70"]["inputs"]["filename_prefix"],
            "images/flux_multi_face_swap_42",
        )


if __name__ == "__main__":
    unittest.main()
