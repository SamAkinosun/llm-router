import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from llm_router.data import load_jsonl, split_prompts_labels

REPO_ROOT = Path(__file__).resolve().parent.parent


class LoadJsonlTests(unittest.TestCase):
    def test_loads_training_data(self):
        rows = load_jsonl(REPO_ROOT / "data" / "training.jsonl")
        self.assertGreater(len(rows), 30)
        for r in rows:
            self.assertIn("prompt", r)
            self.assertIn("label", r)

    def test_loads_eval_data(self):
        rows = load_jsonl(REPO_ROOT / "data" / "eval.jsonl")
        self.assertGreater(len(rows), 20)

    def test_blank_lines_skipped(self):
        with TemporaryDirectory() as d:
            p = Path(d) / "x.jsonl"
            p.write_text(
                '{"prompt": "a", "label": "simple"}\n'
                "\n"
                '{"prompt": "b", "label": "complex"}\n'
            )
            rows = load_jsonl(p)
            self.assertEqual(len(rows), 2)

    def test_invalid_json_raises_with_line_number(self):
        with TemporaryDirectory() as d:
            p = Path(d) / "x.jsonl"
            p.write_text(
                '{"prompt": "a", "label": "simple"}\n'
                "{not json\n"
            )
            with self.assertRaises(ValueError) as ctx:
                load_jsonl(p)
            self.assertIn(":2", str(ctx.exception))

    def test_missing_required_fields_raises(self):
        with TemporaryDirectory() as d:
            p = Path(d) / "x.jsonl"
            p.write_text('{"only_prompt": "a"}\n')
            with self.assertRaises(ValueError):
                load_jsonl(p)


if __name__ == "__main__":
    unittest.main()
