"""Utilities for batches analysis."""

from torch import device as torch_device
from tqdm import tqdm
from transformers import AutoTokenizer

from utilities import coref_dataset
from utilities.collate import DynamicBatchSampler, LeftOversCollator

tokenizer = AutoTokenizer.from_pretrained(
    "distilroberta-base", cache_dir="cache", use_fast=True, add_prefix_space=True
)
dataset, dataset_files = coref_dataset.create(
    tokenizer=tokenizer,
    train_file="/home/nlp/shon711/lingmess-coref/prepare_ontonotes/train.english.jsonlines",
)
device = torch_device("cpu")

collator = LeftOversCollator(tokenizer=tokenizer, device=device, max_segment_len=512)
sampler = DynamicBatchSampler(
    dataset.get("train", False),
    collator=collator,
    max_tokens=40000,
    max_segment_len=512,
)

total_batches_dynamic = 0
total_leftover_batches_dynamic = 0
total_tokens_dynamic = 0
padding_tokens_dynamic = 0
batch_lengths_dynamic = []
for batch in tqdm(sampler):
    total_batches_dynamic += 1

    input_ids = batch.get("input_ids", [])
    total_tokens_dynamic += input_ids.numel()
    padding_tokens_dynamic += input_ids[input_ids == tokenizer.pad_token_id].numel()

    if (
        "leftovers" in batch
        and len(batch.get("leftovers", {}).get("input_ids", [])) > 0
    ):
        total_leftover_batches_dynamic += 1
        input_ids = batch.get("leftovers", {}).get("input_ids", [])
        total_tokens_dynamic += input_ids.numel()
        padding_tokens_dynamic += input_ids[input_ids == tokenizer.pad_token_id].numel()

print(f"Total Examples   : {len(sampler.dataset)}")  # Seeing the tqdm stats.
print(f"Total Batches    : {total_batches_dynamic}")  # Seeing the tqdm stats.
print(f"Total Leftovers  : {total_leftover_batches_dynamic}")  # Seeing the tqdm stats.
print(f"Padding Tokens   : {padding_tokens_dynamic}")
print(f"Input Tokens     : {total_tokens_dynamic - padding_tokens_dynamic}")
print(f"Total Tokens     : {total_tokens_dynamic}")
print(f"Padding Tokens % : {(padding_tokens_dynamic * 100) / total_tokens_dynamic}")
print("--------------------")
print()
