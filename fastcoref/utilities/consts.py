"""Utilities for consts."""

SUPPORTED_MODELS = ["bert", "roberta", "distilbert", "distilroberta", "longformer"]
SPEAKER_START = "#####"
SPEAKER_END = "###"
NULL_ID_FOR_COREF = -1

# Hugging Face Hub commits the default models download from. A local weights path ignores them.
FCOREF_REVISION = "e91dfff12879495d882ba9460d0c5d5dd44ade59"
LINGMESS_REVISION = "fa5d8a827a09388d03adbe9e800c7d8c509c3935"


PRONOUNS_GROUPS = {
    "i": 0,
    "me": 0,
    "my": 0,
    "mine": 0,
    "myself": 0,
    "you": 1,
    "your": 1,
    "yours": 1,
    "yourself": 1,
    "yourselves": 1,
    "he": 2,
    "him": 2,
    "his": 2,
    "himself": 2,
    "she": 3,
    "her": 3,
    "hers": 3,
    "herself": 3,
    "it": 4,
    "its": 4,
    "itself": 4,
    "we": 5,
    "us": 5,
    "our": 5,
    "ours": 5,
    "ourselves": 5,
    "they": 6,
    "them": 6,
    "their": 6,
    "themselves": 6,
    "that": 7,
    "this": 7,
}

STOPWORDS = {
    "'s",
    "a",
    "all",
    "an",
    "and",
    "at",
    "for",
    "from",
    "in",
    "into",
    "more",
    "of",
    "on",
    "or",
    "some",
    "the",
    "these",
    "those",
}

CATEGORIES = {
    "pron-pron-comp": 0,
    "pron-pron-no-comp": 1,
    "pron-ent": 2,
    "match": 3,
    "contain": 4,
    "other": 5,
}
