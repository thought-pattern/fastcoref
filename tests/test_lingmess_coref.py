"""Tests for test lingmess coref."""

from unittest import TestCase as unittest_TestCase

from fastcoref import CorefResult, LingMessCoref


class TestLingMessCoref(unittest_TestCase):
    def setUp(self) -> bool:
        self.test_text = [
            "We are so happy to see you using our coref package. This package is very fast!",
            "The man tried to put the boot on his foot but it was too small.",
        ]
        self.model = LingMessCoref()

        self.expected_clusters = [
            [[(0, 2), (33, 36)], [(33, 50), (52, 64)]],
            [[(0, 7), (33, 36)], [(21, 29), (46, 48)]],
        ]
        self.expected_clusters_strings = [
            [["We", "our"], ["our coref package", "This package"]],
            [["The man", "his"], ["the boot", "it"]],
        ]
        return False

    def test_predict_with_unexpected_object(self):
        texts = {"text1": "sss"}
        with self.assertRaises(ValueError) as exc:
            self.model.predict(texts=texts)
        self.assertEqual(
            str(exc.exception),
            f"texts argument expected to be a list of strings, or one single text string. provided {type(texts)}",
        )
        return False

    def test_predict_with_single_string(self):
        preds = self.model.predict(texts=self.test_text[0])

        self.assertIsInstance(preds, CorefResult)
        return False

    def test_predict_with_list(self):
        preds = self.model.predict(texts=self.test_text)

        self.assertIsInstance(preds, list)
        for res_obj in preds:
            self.assertIsInstance(res_obj, CorefResult)
        return False

    def test_get_clusters(self):
        preds = self.model.predict(texts=self.test_text)

        self.assertIsInstance(preds, list)
        for i, res_obj in enumerate(preds):
            self.assertIsInstance(res_obj, CorefResult)
            self.assertListEqual(
                res_obj.get_clusters(as_strings=True), self.expected_clusters_strings[i]
            )
        return False

    def test_get_clusters_indices(self):
        preds = self.model.predict(texts=self.test_text)

        self.assertIsInstance(preds, list)
        for i, res_obj in enumerate(preds):
            self.assertIsInstance(res_obj, CorefResult)
            self.assertListEqual(
                res_obj.get_clusters(as_strings=False), self.expected_clusters[i]
            )
        return False

    def test_get_logits(self):
        preds = self.model.predict(texts=self.test_text)
        self.assertIsInstance(preds, list)

        self.assertGreater(preds[0].get_logit(span_i=(33, 50), span_j=(52, 64)), 0)
        self.assertGreater(preds[1].get_logit(span_i=(21, 29), span_j=(46, 48)), 0)
        return False
