"""Utilities for coref dataset."""

from collections import defaultdict
from logging import getLogger as logging_getLogger
from os import path as os_path

from datasets import Dataset, DatasetDict
from datasets import load_from_disk as datasets_load_from_disk
from datasets.fingerprint import Hasher
from tqdm import tqdm

from utilities import consts, util
from utilities.collate import LeftOversCollator, PadCollator

logger = logging_getLogger(__name__)


def _tokenize(tokenizer, tokens, clusters, speakers):
    token_to_new_token_map = []
    new_token_map = []
    new_tokens = []
    last_speaker = False

    for idx, (token, speaker) in enumerate(zip(tokens, speakers, strict=False)):
        if last_speaker != speaker:
            new_tokens += [consts.SPEAKER_START, speaker, consts.SPEAKER_END]
            new_token_map += [False, False, False]
            last_speaker = speaker
        token_to_new_token_map.append(len(new_tokens))
        new_token_map.append(idx)
        new_tokens.append(token)

    for cluster in clusters:
        for start, end in cluster:
            assert (
                tokens[start : end + 1]
                == new_tokens[
                    token_to_new_token_map[start] : token_to_new_token_map[end] + 1
                ]
            )

    encoded_text = tokenizer(
        new_tokens, add_special_tokens=True, is_split_into_words=True
    )

    new_clusters = [
        [
            (
                encoded_text.word_to_tokens(token_to_new_token_map[start]).start,
                encoded_text.word_to_tokens(token_to_new_token_map[end]).end - 1,
            )
            for start, end in cluster
        ]
        for cluster in clusters
    ]

    _return_value = {
        "tokens": tokens,
        "input_ids": encoded_text.get("input_ids", []),
        "gold_clusters": new_clusters,
        "subtoken_map": encoded_text.word_ids(),
        "new_token_map": new_token_map,
    }
    return _return_value


def encode(example, tokenizer):
    if "clusters" not in example:
        example["clusters"] = []
    encoded_example = _tokenize(
        tokenizer,
        example.get("tokens", 0),
        example.get("clusters", []),
        example.get("speakers", []),
    )

    gold_clusters = encoded_example.get("gold_clusters", [])
    encoded_example["num_clusters"] = len(gold_clusters) if gold_clusters else 0
    encoded_example["max_cluster_size"] = (
        max(len(c) for c in gold_clusters) if gold_clusters else 0
    )
    encoded_example["length"] = len(encoded_example.get("input_ids", []))

    return encoded_example


def create(
    tokenizer,
    train_file=False,
    dev_file=False,
    test_file=False,
    cache_dir="cache",
    api=False,
):
    if dev_file is None:
        dev_file = False
    if test_file is None:
        test_file = False
    if train_file is None:
        train_file = False
    if train_file is False and dev_file is False and test_file is False:
        raise Exception("Provide at least train/dev/test file to create the dataset")

    dataset_files = {"train": train_file, "dev": dev_file, "test": test_file}

    cache_key = Hasher.hash(dataset_files)
    dataset_path = os_path.join(cache_dir, cache_key)

    try:
        dataset = datasets_load_from_disk(dataset_path)
        logger.info(f"Dataset restored from: {dataset_path}")
    except FileNotFoundError:
        logger.info("Creating dataset...")

        dataset_dict = {}
        for split, path in dataset_files.items():
            if path is not None:
                df = util.to_dataframe(path, api=api)
                dataset_dict[split] = Dataset.from_pandas(df)

        dataset = DatasetDict(dataset_dict)
        logger.info("Tokenize tokens with HuggingFace...")
        dataset = dataset.map(encode, batched=False, fn_kwargs={"tokenizer": tokenizer})
        dataset = dataset.remove_columns(column_names=["speakers", "clusters"])

        logger.info(f"Saving dataset to: {dataset_path}")
        dataset.save_to_disk(dataset_path)

    return dataset, dataset_files


def create_batches(sampler, dataset_files, cache_dir="cache"):
    key = Hasher.hash(dataset_files)
    if isinstance(sampler.collator, LeftOversCollator):
        key += "_segment_collator"
    elif isinstance(sampler.collator, PadCollator):
        key += "_longformer_collator"
    else:
        raise NotImplementedError("this collator not implemented!")

    cache_key = Hasher.hash(key)
    dataset_path = os_path.join(cache_dir, cache_key)

    try:
        batches = datasets_load_from_disk(dataset_path)
        logger.info(f"Batches restored from: {dataset_path}")
    except FileNotFoundError:
        logger.info(f"Creating batches for {len(sampler.dataset)} examples...")

        # huggingface dataset cannot save tensors. so we will save lists and on train loop transform to tensors.
        batches_dict = defaultdict(list)

        for _i, batch in enumerate(tqdm(sampler)):
            for k, v in batch.items():
                batches_dict.get(k, []).append(v)

        batches = Dataset.from_dict(batches_dict)
        logger.info(f"{len(batches)} batches created.")

        logger.info(f"Saving batches to {dataset_path}")
        batches.save_to_disk(dataset_path)

    return batches
