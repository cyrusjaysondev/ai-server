import ast
import unittest
from pathlib import Path


MAIN_PATH = Path(__file__).resolve().parents[1] / "main.py"


class BlocklistEnforcementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse(MAIN_PATH.read_text())

    def test_input_filter_honors_disabled_flag(self):
        function = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_apply_face_filter"
        )
        disabled_branch = next(
            node
            for node in function.body
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.UnaryOp)
            and isinstance(node.test.op, ast.Not)
            and isinstance(node.test.operand, ast.Name)
            and node.test.operand.id == "face_filter"
        )
        self.assertTrue(any(isinstance(node, ast.Return) for node in disabled_branch.body))
        self.assertTrue(any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "log_bypass"
            for node in ast.walk(disabled_branch)
        ))

    def test_input_filter_does_not_override_caller_flag(self):
        function = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_apply_face_filter"
        )
        overrides = [
            node
            for node in function.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "face_filter"
                for target in node.targets
            )
        ]
        self.assertEqual(overrides, [])

    def test_generated_output_checks_honor_request_flag(self):
        output_filter_values = [
            keyword.value
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Call)
            for keyword in node.keywords
            if keyword.arg == "output_face_filter"
        ]
        self.assertGreaterEqual(len(output_filter_values), 3)
        self.assertTrue(any(
            isinstance(value, ast.Attribute)
            and isinstance(value.value, ast.Name)
            and value.value.id == "req"
            and value.attr == "face_filter"
            for value in output_filter_values
        ))
        self.assertTrue(all(
            (
                isinstance(value, ast.Name)
                and value.id == "face_filter"
            )
            or (
                isinstance(value, ast.Attribute)
                and isinstance(value.value, ast.Name)
                and value.value.id == "req"
                and value.attr == "face_filter"
            )
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
