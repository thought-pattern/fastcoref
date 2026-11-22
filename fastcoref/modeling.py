"""Utilities for modeling."""

from json import dumps as json_dumps
from logging import (
    INFO as logging_INFO,
    basicConfig as logging_basicConfig,
    getLogger as logging_getLogger,
)

from datasets import Dataset
from numpy import empty as np_empty, float32 as np_float32, nonzero as np_nonzero
from spacy.language import Language
from torch import cuda as torch_cuda, device as torch_device, no_grad as torch_no_grad
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoTokenizer, logging as transformers_logging

from .coref_models.modeling_fcoref import FCorefModel
from .coref_models.modeling_lingmess import LingMessModel
from .utilities.collate import (
    DynamicBatchSampler,
    LeftOversCollator,
    PadCollator,
)
from .utilities.consts import FCOREF_REVISION, LINGMESS_REVISION, PRONOUNS_GROUPS
from .utilities.util import (
    align_to_char_level,
    create_clusters,
    create_mention_to_antecedent,
    encode_pretokenized,
    encode_text,
    set_seed,
)

# Setup logging
logger = logging_getLogger(__name__)
logging_basicConfig(
    format="%(asctime)s - %(levelname)s - \t %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging_INFO,
)

PRONOUN_SURFACES = set(PRONOUNS_GROUPS) | {
    "theirs",
    "these",
    "those",
    "who",
    "whom",
    "whose",
    "which",
    "oneself",
}


def is_pronoun_mention(mention: str) -> bool:
    """Return True when `mention` is a standalone pronoun rather than a named or nominal entity."""
    cleaned = str(mention or "").strip().strip(".,;:!?()[]{}\"'").lower()
    result = bool(cleaned) and cleaned in PRONOUN_SURFACES
    return result


def aligned_char_clusters(clusters, char_map) -> list:
    """Return each cluster's distinct source character spans in mention order.

    align_to_char_level gives a mention on a tokenizer special or padding position a False
    span. Subword spans within the same word align to the same character range and are
    deduplicated, so a cluster left with fewer than two distinct aligned mentions is omitted.
    """
    char_clusters = []
    omitted = 0
    for cluster in clusters:
        spans = [char_map.get(mention, (False, False))[1] for mention in cluster]
        aligned = [span for span in spans if isinstance(span, tuple)]
        omitted += len(spans) - len(aligned)
        deduped = list(dict.fromkeys(aligned))
        if len(deduped) >= 2:
            char_clusters.append(deduped)
    if omitted:
        logger.info(f"Omitted {omitted} coreference mention(s) without a source text alignment")
    return char_clusters


class CorefResult:
    """One text's predicted coreference clusters with their character spans and antecedent logits."""

    def __init__(self, text, clusters, char_map, reverse_char_map, coref_logit, text_idx):
        self.text = text
        self.clusters = clusters
        self.char_map = char_map
        self.reverse_char_map = reverse_char_map
        self.coref_logit = coref_logit
        self.text_idx = text_idx
        # Both cluster views and the resolved text read this one aligned form, so they stay in step.
        self.char_clusters = aligned_char_clusters(clusters, char_map)

    def mention_text(self, start, end) -> str:
        """Render the mention text for `(start, end)` on either string or pretokenized input."""
        if isinstance(self.text, str):
            sliced = self.text[start:end]
            return sliced
        joined = " ".join(str(token) for token in self.text[start:end])
        return joined

    def get_clusters(self, as_strings=True):
        if not as_strings:
            computed_return_value = [list(cluster) for cluster in self.char_clusters]
            return computed_return_value

        computed_return_value = [[self.mention_text(start, end) for start, end in cluster] for cluster in self.char_clusters]
        return computed_return_value

    def get_logit(self, span_i, span_j):
        if span_i not in self.reverse_char_map:
            raise ValueError(f'span_i="{self.mention_text(span_i[0], span_i[1])}" is not an entity in this model!')
        if span_j not in self.reverse_char_map:
            raise ValueError(f'span_j="{self.mention_text(span_j[0], span_j[1])}" is not an entity in this model!')

        span_i_idx = self.reverse_char_map[span_i][0]  # 0 is to get the span index
        span_j_idx = self.reverse_char_map[span_j][0]

        if span_i_idx < span_j_idx:
            computed_return_value = self.coref_logit[span_j_idx, span_i_idx]
            return computed_return_value

        computed_return_value = self.coref_logit[span_i_idx, span_j_idx]
        return computed_return_value

    def get_resolved_text(self):
        """Replace coreferent mentions with their canonical non-pronoun antecedent."""
        char_clusters = self.get_clusters(as_strings=False)
        str_clusters = self.get_clusters(as_strings=True)

        if not char_clusters:
            if isinstance(self.text, str):
                return self.text
            joined_default = " ".join(str(token) for token in self.text)
            return joined_default

        # Build replacement list: prefer non-pronoun mentions, then longest text, then earliest position.
        replacements = []
        order = 0
        for char_cluster, str_cluster in zip(char_clusters, str_clusters, strict=False):
            if len(char_cluster) < 2:
                continue
            canonical_idx = max(
                (0 if is_pronoun_mention(mention) else 1, len(mention), -idx, idx) for idx, mention in enumerate(str_cluster)
            )[3]
            canonical = str_cluster[canonical_idx]
            for i, (start, end) in enumerate(char_cluster):
                if i != canonical_idx and start < end:
                    replacements.append((start, end, canonical, order))
                    order += 1

        if not replacements:
            if isinstance(self.text, str):
                return self.text
            joined_unmodified = " ".join(str(token) for token in self.text)
            return joined_unmodified

        # Keep outer longest non-overlapping spans first, then apply in descending start order.
        ordered = sorted(replacements, key=lambda item: (-(item[1] - item[0]), item[0], item[3]))
        selected = []
        for start, end, canonical, unused_order in ordered:
            del unused_order
            if any(start < kept_end and kept_start < end for kept_start, kept_end, unused_canonical in selected):
                continue
            selected.append((start, end, canonical))

        selected.sort(key=lambda item: item[0], reverse=True)
        if isinstance(self.text, str):
            resolved = self.text
            for start, end, replacement in selected:
                resolved = resolved[:start] + replacement + resolved[end:]
            return resolved
        resolved_tokens = [str(token) for token in self.text]
        for start, end, replacement in selected:
            resolved_tokens[start:end] = [replacement]
        joined_resolved = " ".join(resolved_tokens)
        return joined_resolved

    def __str__(self):
        display_text = self.text if isinstance(self.text, str) else " ".join(str(token) for token in self.text)
        if len(display_text) > 50:
            text_to_print = f"{display_text[:50]}..."
        else:
            text_to_print = display_text
        computed_return_value = f'CorefResult(text="{text_to_print}", clusters={self.get_clusters()})'
        return computed_return_value

    def __repr__(self):
        computed_return_value = self.__str__()
        return computed_return_value


class CorefModel:
    """Coreference inference over a pretrained model, its tokenizer, a collator and a spaCy pipeline.

    A Hugging Face Hub model name downloads the pinned revision; a local weights path loads as is.
    """

    def __init__(
        self,
        model_name_or_path,
        coref_class,
        collator_class,
        enable_progress_bar,
        device: str,
        nlp: Language,
        revision: str,
    ):
        self.model_name_or_path = model_name_or_path
        self.revision = revision
        selected_device = device or ("cuda" if torch_cuda.is_available() else "cpu")
        self.device = torch_device(selected_device)
        self.n_gpu = torch_cuda.device_count()
        self.seed = 42
        self.enable_progress_bar = enable_progress_bar

        config = AutoConfig.from_pretrained(self.model_name_or_path, revision=self.revision)
        self.max_segment_len = config.coref_head.get("max_segment_len", False)
        self.max_doc_len = config.coref_head.get("max_doc_len", False) if "max_doc_len" in config.coref_head else False

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path, revision=self.revision, use_fast=True, add_prefix_space=True, verbose=False
        )

        if collator_class == PadCollator:
            self.collator = PadCollator(tokenizer=self.tokenizer, device=self.device)
        elif collator_class == LeftOversCollator:
            self.collator = LeftOversCollator(
                tokenizer=self.tokenizer,
                device=self.device,
                max_segment_len=config.coref_head.get("max_segment_len", False),
            )
        else:
            raise NotImplementedError(
                f"Class collator {type(collator_class)} is not supported! only LeftOversCollator or PadCollator supported"
            )
        if not isinstance(nlp, Language):
            raise TypeError("nlp must be an initialized spaCy Language pipeline")
        self.nlp = nlp

        self.model = coref_class.from_pretrained(
            self.model_name_or_path,
            config=config,
            revision=self.revision,
        )
        self.model.to(self.device)

        t_params, h_params = [p / 1000000 for p in self.model.num_parameters()]
        logger.info(f"Model Parameters: {t_params + h_params:.1f}M, Transformer: {t_params:.1f}M, Coref head: {h_params:.1f}M")

        set_seed(self)
        transformers_logging.set_verbosity_error()

    def create_dataset(self, texts, is_split_into_words):
        logger.info(f"Tokenize {len(texts)} inputs...")

        # Save original text ordering for later use
        dataset = {"text": texts, "idx": range(len(texts))}
        if is_split_into_words:
            dataset["tokens"] = texts

        dataset = Dataset.from_dict(dataset)
        if is_split_into_words:
            dataset = dataset.map(
                encode_pretokenized,
                batched=True,
                batch_size=10000,
                fn_kwargs={"tokenizer": self.tokenizer},
            )
        else:
            dataset = dataset.map(
                encode_text,
                batched=True,
                batch_size=10000,
                fn_kwargs={"tokenizer": self.tokenizer, "nlp": self.nlp},
            )

        return dataset

    def prepare_batches(self, dataset, max_tokens_in_batch):
        dataloader = DynamicBatchSampler(
            dataset,
            collator=self.collator,
            max_tokens=max_tokens_in_batch,
            max_segment_len=self.max_segment_len,
            max_doc_len=self.max_doc_len,
        )

        return dataloader

    def batch_inference(self, batch):
        texts = batch.get("text", [])
        subtoken_map = batch.get("subtoken_map", [])
        token_to_char = batch.get("offset_mapping", [])
        idxs = batch.get("idx", [])
        with torch_no_grad():
            outputs = self.model(batch, return_all_outputs=True)

        outputs_np = tuple(tensor.cpu().numpy() for tensor in outputs)

        span_starts, span_ends, _, coref_logits = outputs_np
        doc_indices, mention_to_antecedent = create_mention_to_antecedent(span_starts, span_ends, coref_logits)

        results = []

        for i in range(len(texts)):
            doc_mention_to_antecedent = mention_to_antecedent[np_nonzero(doc_indices == i)]
            predicted_clusters = create_clusters(doc_mention_to_antecedent)

            char_map, reverse_char_map = align_to_char_level(span_starts[i], span_ends[i], token_to_char[i], subtoken_map[i])

            result = CorefResult(
                text=texts[i],
                clusters=predicted_clusters,
                char_map=char_map,
                reverse_char_map=reverse_char_map,
                coref_logit=coref_logits[i],
                text_idx=idxs[i],
            )

            results.append(result)

        return results

    def inference(self, dataloader):
        self.model.eval()
        logger.info(f"***** Running Inference on {len(dataloader.dataset)} texts *****")

        results = []
        if self.enable_progress_bar:
            with tqdm(desc="Inference", total=len(dataloader.dataset)) as progress_bar:
                for batch in dataloader:
                    results.extend(self.batch_inference(batch))
                    progress_bar.update(n=len(batch.get("text", "")))
                skipped_count = len(getattr(dataloader, "skipped", []))
                if skipped_count:
                    progress_bar.update(n=skipped_count)
        else:
            for batch in dataloader:
                results.extend(self.batch_inference(batch))

        seen_indices = {res.text_idx for res in results}
        for example in getattr(dataloader, "skipped", []):
            idx = example.get("idx", 0)
            if idx not in seen_indices:
                seen_indices.add(idx)
                results.append(
                    CorefResult(
                        text=example.get("text", ""),
                        clusters=[],
                        char_map={},
                        reverse_char_map={},
                        coref_logit=np_empty((0, 0), dtype=np_float32),
                        text_idx=idx,
                    )
                )

        computed_return_value = sorted(results, key=lambda res: res.text_idx)
        return computed_return_value

    def predict(
        self,
        texts: object,  # similar to huggingface tokenizer inputs
        is_split_into_words: bool = False,
        max_tokens_in_batch: int = 10000,
        output_file: str = "",
    ):
        """
        texts (str, List[str], List[List[str]]) — The sequence or batch of sequences to be encoded.
        Each sequence can be a string or a list of strings (pretokenized string).
        If the sequences are provided as list of strings (pretokenized), you must set is_split_into_words=True
        (to lift the ambiguity with a batch of sequences).
        is_split_into_words - indicate if the texts input is tokenized
        """

        # Input type checking for clearer error
        def is_valid_text_input(texts, is_split_into_words):
            if isinstance(texts, str) and not is_split_into_words:
                # Strings are fine
                return True
            elif isinstance(texts, (list, tuple)):
                # List are fine as long as they are...
                if len(texts) == 0:
                    # ... empty
                    return True
                elif all([isinstance(t, str) for t in texts]):
                    # ... list of strings
                    return True
                elif all([isinstance(t, (list, tuple)) for t in texts]):
                    # ... list with an empty list or with a list of strings
                    computed_return_value = all(all(isinstance(word, str) for word in t) for t in texts)
                    return computed_return_value
                else:
                    return False
            else:
                return False

        if not is_valid_text_input(texts, is_split_into_words):
            raise ValueError(
                "text input must be of type `str` (single example), `List[str]` (batch or single pretokenized example) "
                "or `List[List[str]]` (batch of pretokenized examples)."
            )

        if not is_split_into_words and not self.nlp:
            raise ValueError(
                "Model initialized with no nlp component for tokenizing the text, please pass pretokenized text,"
                "or initialize the model with an nlp component."
            )

        is_batched = False
        if isinstance(texts, (list, tuple)):
            if len(texts) == 0:
                if output_file != "":
                    with open(output_file, "w", encoding="utf-8") as f:
                        f.write("")
                computed_return_value = []
                return computed_return_value
            is_batched = not is_split_into_words or isinstance(texts[0], (list, tuple))

        if not is_batched:
            texts = [texts]

        dataset = self.create_dataset(texts, is_split_into_words)
        dataloader = self.prepare_batches(dataset, max_tokens_in_batch)

        preds = self.inference(dataloader)
        if output_file != "":
            with open(output_file, "w", encoding="utf-8") as f:
                data = [
                    {
                        "text": p.text,
                        "clusters": p.get_clusters(as_strings=False),
                        "clusters_strings": p.get_clusters(as_strings=True),
                    }
                    for p in preds
                ]
                f.write("\n".join(map(json_dumps, data)))
        if not is_batched:
            computed_return_value = preds[0]
            return computed_return_value
        return preds


class FCoref(CorefModel):
    """The F-coref model with the leftovers collator."""

    def __init__(
        self,
        model_name_or_path="biu-nlp/f-coref",
        device: str = "",
        *,
        nlp: Language,
        enable_progress_bar=True,
        revision: str = FCOREF_REVISION,
    ):
        super().__init__(
            model_name_or_path,
            FCorefModel,
            LeftOversCollator,
            enable_progress_bar,
            device,
            nlp,
            revision,
        )


class LingMessCoref(CorefModel):
    """The LingMess model with the padding collator."""

    def __init__(
        self,
        model_name_or_path="biu-nlp/lingmess-coref",
        device: str = "",
        *,
        nlp: Language,
        enable_progress_bar=True,
        revision: str = LINGMESS_REVISION,
    ):
        super().__init__(
            model_name_or_path,
            LingMessModel,
            PadCollator,
            enable_progress_bar,
            device,
            nlp,
            revision,
        )
