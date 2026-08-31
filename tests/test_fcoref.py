"""Tests for test fcoref."""

from pathlib import Path
from unittest import TestCase as unittest_TestCase

from fastcoref import CorefResult, FCoref
from spacy import load as spacy_load


class TestFCoref(unittest_TestCase):
    @classmethod
    def setUpClass(cls):
        model_path = Path(__file__).parents[2] / "weights" / "biu-nlp" / "f-coref"
        nlp = spacy_load("en_core_web_sm", exclude=["tagger", "parser", "lemmatizer", "ner", "textcat"])
        cls.model = FCoref(
            model_name_or_path=str(model_path),
            device="cpu",
            nlp=nlp,
            enable_progress_bar=False,
        )
        cls.test_text = [
            "We are so happy to see you using our coref package. This package is very fast!",
            "The man tried to put the boot on his foot but it was too small.",
        ]
        cls.predictions = cls.model.predict(texts=cls.test_text)

    def test_predict_with_unexpected_object(self):
        texts = {"text1": "sss"}
        with self.assertRaises(ValueError):
            self.model.predict(texts=texts)

    def test_predict_with_single_string(self):
        preds = self.model.predict(texts=self.test_text[0])

        self.assertIsInstance(preds, CorefResult)

    def test_predict_with_list(self):
        self.assertIsInstance(self.predictions, list)
        self.assertEqual(len(self.predictions), len(self.test_text))
        for res_obj in self.predictions:
            self.assertIsInstance(res_obj, CorefResult)

    def test_clusters_preserve_text_spans(self):
        for text, result in zip(self.test_text, self.predictions, strict=True):
            index_clusters = result.get_clusters(as_strings=False)
            string_clusters = result.get_clusters(as_strings=True)
            self.assertTrue(index_clusters)
            self.assertEqual(len(index_clusters), len(string_clusters))
            for index_cluster, string_cluster in zip(index_clusters, string_clusters, strict=True):
                self.assertGreaterEqual(len(index_cluster), 2)
                self.assertEqual(
                    [text[start:end] for start, end in index_cluster],
                    string_cluster,
                )

    def test_clusters_resolve_known_references(self):
        first_clusters = [set(cluster) for cluster in self.predictions[0].get_clusters(as_strings=True)]
        second_clusters = [set(cluster) for cluster in self.predictions[1].get_clusters(as_strings=True)]
        self.assertIn({"We", "our"}, first_clusters)
        self.assertIn({"The man", "his"}, second_clusters)

    def test_get_logits(self):
        first_cluster = self.predictions[0].get_clusters(as_strings=False)[0]
        self.assertGreater(self.predictions[0].get_logit(*first_cluster), 0)
        with self.assertRaises(ValueError):
            self.predictions[1].get_logit(span_i=(21, 29), span_j=(46, 48))
