import ast
import unittest
from pathlib import Path


MAIN_PATH = Path(__file__).resolve().parents[1] / "main.py"


class BlocklistEnforcementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse(MAIN_PATH.read_text())

    def test_server_enforcement_is_enabled(self):
        assignment = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "ENFORCE_FACE_BLOCKLIST"
                for target in node.targets
            )
        )
        self.assertIsInstance(assignment.value, ast.Constant)
        self.assertIs(assignment.value.value, True)

    def test_input_filter_overrides_caller_flag(self):
        function = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_apply_face_filter"
        )
        override = next(
            node
            for node in function.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "face_filter"
                for target in node.targets
            )
        )
        self.assertIsInstance(override.value, ast.Name)
        self.assertEqual(override.value.id, "ENFORCE_FACE_BLOCKLIST")

    def test_all_generated_output_checks_use_server_policy(self):
        output_filter_values = [
            keyword.value
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Call)
            for keyword in node.keywords
            if keyword.arg == "output_face_filter"
        ]
        self.assertGreaterEqual(len(output_filter_values), 3)
        self.assertTrue(all(
            isinstance(value, ast.Name)
            and value.id == "ENFORCE_FACE_BLOCKLIST"
            for value in output_filter_values
        ))

    def test_admin_area_ratio_is_json_serializable(self):
        area_ratio_values = [
            value
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Dict)
            for key, value in zip(node.keys, node.values)
            if isinstance(key, ast.Constant) and key.value == "area_ratio"
        ]
        self.assertEqual(len(area_ratio_values), 1)
        round_call = area_ratio_values[0]
        self.assertIsInstance(round_call, ast.Call)
        self.assertIsInstance(round_call.func, ast.Name)
        self.assertEqual(round_call.func.id, "round")
        float_call = round_call.args[0]
        self.assertIsInstance(float_call, ast.Call)
        self.assertIsInstance(float_call.func, ast.Name)
        self.assertEqual(float_call.func.id, "float")


if __name__ == "__main__":
    unittest.main()
