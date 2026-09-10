import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from run_experiment import already_done, main, save  # noqa: E402


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.outdir = Path(self.temporary_directory.name)
        self.scenario = {
            "question_id": 0,
            "question": "Question?",
            "answer": "Answer",
            "length": 1,
            "position": "first",
            "prompt": "Prompt",
            "doc_texts": ["Document"],
            "gold_doc": "Document",
            "gold_index": 0,
        }
        self.metadata = {
            "model_key": "test:model",
            "model_id": "test/model",
            "base_seed": 42,
            "resolved_model_revision": "revision",
            "generation_config": {"do_sample": True},
        }

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_malformed_result_is_not_skipped(self):
        (self.outdir / "result_0000.json").write_text("")
        self.assertFalse(already_done(self.outdir, 0))

    def test_complete_result_is_skipped(self):
        save(self.outdir, 0, self.scenario, "Answer", 0.1, 1, self.metadata)
        self.assertTrue(already_done(self.outdir, 0))
        self.assertFalse((self.outdir / "result_0000.json.tmp").exists())
        with (self.outdir / "result_0000.json").open() as f:
            self.assertEqual(json.load(f)["response"], "Answer")

    def test_empty_scenario_list_is_rejected_before_model_loading(self):
        scenarios_path = self.outdir / "contexts.json"
        scenarios_path.write_text("[]")
        argv = ["run_experiment.py", "--scenarios", str(scenarios_path)]
        with patch.object(sys, "argv", argv):
            with self.assertRaisesRegex(ValueError, "non-empty list"):
                main()


if __name__ == "__main__":
    unittest.main()
