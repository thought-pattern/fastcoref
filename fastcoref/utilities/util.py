"""Utilities for util."""

from logging import getLogger as logging_getLogger
from os import makedirs as os_makedirs
from os import path as os_path
from random import seed as random_seed

from numpy import nonzero as np_nonzero
from numpy import random as np_random
from numpy import stack as np_stack
from torch import clamp as torch_clamp
from torch import cuda as torch_cuda
from torch import manual_seed as torch_manual_seed

from .consts import CATEGORIES, NULL_ID_FOR_COREF, PRONOUNS_GROUPS
from .metrics import CorefEvaluator, MentionEvaluator

logger = logging_getLogger(__name__)


def flatten(local_element):
    _return_value = [item for sublist in local_element for item in sublist]
    return _return_value


def save_all(model, tokenizer, output_dir):
    logger.info(f"Saving model to {output_dir}")
    if not os_path.exists(output_dir):
        os_makedirs(output_dir)

    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return False


def pad_clusters_inside(clusters, max_cluster_size):
    _return_value = [
        cluster
        + [(NULL_ID_FOR_COREF, NULL_ID_FOR_COREF)] * (max_cluster_size - len(cluster))
        for cluster in clusters
    ]
    return _return_value


def pad_clusters_outside(clusters, max_num_clusters):
    _return_value = clusters + [[]] * (max_num_clusters - len(clusters))
    return _return_value


def pad_clusters(clusters, max_num_clusters, max_cluster_size):
    clusters = pad_clusters_outside(clusters, max_num_clusters)
    clusters = pad_clusters_inside(clusters, max_cluster_size)
    return clusters


def output_evaluation_metrics(metrics_dict, prefix):
    loss = metrics_dict.get("loss", [])
    post_pruning_mention_pr, post_pruning_mentions_r, post_pruning_mention_f1 = (
        metrics_dict.get("post_pruning", MentionEvaluator()).get_prf()
    )
    mention_p, mentions_r, mention_f1 = metrics_dict.get(
        "mentions", MentionEvaluator()
    ).get_prf()
    p, r, f1 = metrics_dict.get("coref", CorefEvaluator()).get_prf()
    results = {
        "eval_loss": loss,
        "post pruning mention precision": post_pruning_mention_pr,
        "post pruning mention recall": post_pruning_mentions_r,
        "post pruning mention f1": post_pruning_mention_f1,
        "mention precision": mention_p,
        "mention recall": mentions_r,
        "mention f1": mention_f1,
        "precision": p,
        "recall": r,
        "f1": f1,
    }

    logger.info("***** Eval results {} *****".format(prefix))
    for key, value in results.items():
        if isinstance(value, float):
            logger.info(f"  {key: <30} = {value:.3f}")
        elif isinstance(value, dict):
            logger.info(f"  {key: <30} = {value}")

    return results


def update_metrics(metrics, span_starts, span_ends, gold_clusters, predicted_clusters):
    gold_clusters = extract_clusters(gold_clusters)
    candidate_mentions = list(zip(span_starts, span_ends, strict=False))

    mention_to_gold_clusters = extract_mentions_to_clusters(gold_clusters)
    mention_to_predicted_clusters = extract_mentions_to_clusters(predicted_clusters)

    gold_mentions = list(mention_to_gold_clusters.keys())
    predicted_mentions = list(mention_to_predicted_clusters.keys())

    metrics.get("post_pruning", {}).update(candidate_mentions, gold_mentions)
    metrics.get("mentions", {}).update(predicted_mentions, gold_mentions)
    metrics.get("coref", {}).update(
        predicted_clusters,
        gold_clusters,
        mention_to_predicted_clusters,
        mention_to_gold_clusters,
    )
    return False


def encode(batch, tokenizer, nlp):
    if nlp is not None:
        tokenized_texts = tokenize_with_spacy(batch.get("text", ""), nlp)
    else:
        tokenized_texts = batch
        tokenized_texts["offset_mapping"] = [
            (list(zip(range(len(tokens)), range(1, 1 + len(tokens)), strict=False)))
            for tokens in tokenized_texts.get("tokens", 0)
        ]
    encoded_batch = tokenizer(
        tokenized_texts.get("tokens", 0),
        add_special_tokens=True,
        is_split_into_words=True,
        return_length=True,
        return_attention_mask=False,
    )
    _return_value = {
        "tokens": tokenized_texts.get("tokens", 0),
        "input_ids": encoded_batch.get("input_ids", []),
        "length": encoded_batch.get("length", 0),
        # bpe token -> spacy tokens
        "subtoken_map": [enc.word_ids for enc in encoded_batch.encodings],
        # this is a can use for speaker info TODO: better name!
        "new_token_map": [
            list(range(len(tokens))) for tokens in tokenized_texts.get("tokens", 0)
        ],
        # spacy tokens -> text char
        "offset_mapping": tokenized_texts.get("offset_mapping", {}),
    }
    return _return_value


def tokenize_with_spacy(texts, nlp):
    def handle_doc(doc):
        tokens = []
        offset_mapping = []
        for tok in doc:
            tokens.append(tok.text)
            offset_mapping.append((tok.idx, tok.idx + len(tok.text)))
        return tokens, offset_mapping

    tokenized_texts = {"tokens": [], "offset_mapping": []}

    # Edge case - Also disable other custom components
    all_pipe_names = nlp.pipe_names
    tokenizer_pipe_names = ["tok2vec"]

    disabled_pipe_names = [
        pipe_name
        for pipe_name in all_pipe_names
        if pipe_name not in tokenizer_pipe_names
    ]
    docs = nlp.pipe(texts, disable=disabled_pipe_names)
    for doc in docs:
        tokens, offset_mapping = handle_doc(doc)
        tokenized_texts.get("tokens", []).append(tokens)
        tokenized_texts.get("offset_mapping", []).append(offset_mapping)

    return tokenized_texts


def align_to_char_level(
    span_starts, span_ends, token_to_char, subtoken_map=False, new_token_map=False
):
    if new_token_map is None:
        new_token_map = False
    if subtoken_map is None:
        subtoken_map = False
    char_map = {}
    reverse_char_map = {}
    for idx, (start, end) in enumerate(zip(span_starts, span_ends, strict=False)):
        new_start, new_end = start.copy(), end.copy()

        try:
            if subtoken_map is not False:
                new_start, new_end = subtoken_map[new_start], subtoken_map[new_end]
                if new_start is None or new_end is None:
                    # this is a special token index
                    char_map[(start, end)] = False, False
                    continue
            if new_token_map is not False:
                new_start, new_end = new_token_map[new_start], new_token_map[new_end]
            new_start, new_end = token_to_char[new_start][0], token_to_char[new_end][1]
            char_map[(start, end)] = idx, (new_start, new_end)
            reverse_char_map[(new_start, new_end)] = idx, (start, end)
        except IndexError:
            # this is padding index
            char_map[(start, end)] = False, False
            continue

    return char_map, reverse_char_map


def set_seed(args):
    random_seed(args.seed)
    np_random.seed(args.seed)
    torch_manual_seed(args.seed)
    if args.n_gpu > 0:
        torch_cuda.manual_seed_all(args.seed)
    return False


def extract_clusters(gold_clusters):
    gold_clusters = [
        tuple(tuple(m) for m in cluster if NULL_ID_FOR_COREF not in m)
        for cluster in gold_clusters
    ]
    gold_clusters = [cluster for cluster in gold_clusters if len(cluster) > 0]
    return gold_clusters


def extract_mentions_to_clusters(gold_clusters):
    mention_to_gold = {}
    for gc in gold_clusters:
        for mention in gc:
            mention_to_gold[mention] = gc
    return mention_to_gold


def create_clusters(mention_to_antecedent):
    # Note: mention_to_antecedent is a numpy array

    clusters, mention_to_cluster = [], {}
    for mention, antecedent in mention_to_antecedent:
        mention, antecedent = tuple(mention), tuple(antecedent)
        if antecedent in mention_to_cluster:
            cluster_idx = mention_to_cluster[antecedent]
            if mention not in clusters[cluster_idx]:
                clusters[cluster_idx].append(mention)
                mention_to_cluster[mention] = cluster_idx
        elif mention in mention_to_cluster:
            cluster_idx = mention_to_cluster[mention]
            if antecedent not in clusters[cluster_idx]:
                clusters[cluster_idx].append(antecedent)
                mention_to_cluster[antecedent] = cluster_idx
        else:
            cluster_idx = len(clusters)
            mention_to_cluster[mention] = cluster_idx
            mention_to_cluster[antecedent] = cluster_idx
            clusters.append([antecedent, mention])

    clusters = [tuple(cluster) for cluster in clusters]
    return clusters


def create_mention_to_antecedent(span_starts, span_ends, coref_logits):
    batch_size, n_spans, _ = coref_logits.shape

    max_antecedents = coref_logits.argmax(axis=-1)
    doc_indices, mention_indices = np_nonzero(
        max_antecedents < n_spans
    )  # indices where antecedent is not null.
    antecedent_indices = max_antecedents[max_antecedents < n_spans]
    span_indices = np_stack([span_starts, span_ends], axis=-1)

    mentions = span_indices[doc_indices, mention_indices]
    antecedents = span_indices[doc_indices, antecedent_indices]
    mention_to_antecedent = np_stack([mentions, antecedents], axis=1)

    return doc_indices, mention_to_antecedent


def mask_tensor(t, mask):
    t = t + ((1.0 - mask.float()) * -10000.0)
    t = torch_clamp(t, min=-10000.0, max=10000.0)
    return t


def get_pronoun_id(span):
    if len(span) == 1:
        span = list(span)
        if span[0] in PRONOUNS_GROUPS:
            _return_value = PRONOUNS_GROUPS.get(span[0], "")
            return _return_value
    _return_value = -1
    return _return_value


def get_category_id(mention, antecedent):
    mention, mention_pronoun_id = mention
    antecedent, antecedent_pronoun_id = antecedent

    if mention_pronoun_id > -1 and antecedent_pronoun_id > -1:
        if mention_pronoun_id == antecedent_pronoun_id:
            _return_value = CATEGORIES.get("pron-pron-comp", "")
            return _return_value
        else:
            _return_value = CATEGORIES.get("pron-pron-no-comp", "")
            return _return_value

    if mention_pronoun_id > -1 or antecedent_pronoun_id > -1:
        _return_value = CATEGORIES.get("pron-ent", "")
        return _return_value

    if mention == antecedent:
        _return_value = CATEGORIES.get("match", "")
        return _return_value

    union = mention.union(antecedent)
    if len(union) == max(len(mention), len(antecedent)):
        _return_value = CATEGORIES.get("contain", "")
        return _return_value

    _return_value = CATEGORIES.get("other", "")
    return _return_value
