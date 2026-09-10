import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from classify_response import classify, is_abstention, is_correct, normalize  # noqa: E402


def result(response):
    return {
        "response": response,
        "answer": "London",
        "doc_texts": [
            "The answer to the test question is London.",
            "Paris capital France",
        ],
        "gold_doc": "The answer to the test question is London.",
    }


class ClassificationTests(unittest.TestCase):
    def test_normalize_handles_case_punctuation_and_hyphens(self):
        self.assertEqual(normalize("North-East!"), "north east")

    def test_abstention_variants(self):
        self.assertTrue(is_abstention("It is not mentioned in the documents."))

    def test_correct_accepts_partial_multiword_answer(self):
        self.assertTrue(is_correct("The Caucasus region.", "Caucasus Mountains"))

    def test_abstention_has_priority_over_answer_match(self):
        self.assertEqual(classify(result("I don't know, but perhaps London.")), "abstained")

    def test_correct_answer(self):
        self.assertEqual(classify(result("London")), "correct")

    def test_distractor_match(self):
        self.assertEqual(classify(result("Paris")), "wrong_document")

    def test_residual_response_is_hallucinated(self):
        self.assertEqual(classify(result("Tokyo")), "hallucinated")


if __name__ == "__main__":
    unittest.main()
