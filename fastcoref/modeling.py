"""Utilities for modeling."""

from json import dumps as json_dumps
from logging import INFO as logging_INFO
from logging import basicConfig as logging_basicConfig
from logging import getLogger as logging_getLogger

from datasets import Dataset
from numpy import nonzero as np_nonzero
from spacy import load as spacy_load
from spacy.cli import download
from spacy.language import Language
from torch import cuda as torch_cuda
from torch import device as torch_device
from torch import no_grad as torch_no_grad
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoTokenizer
from transformers import logging as transformers_logging

from .coref_models.modeling_fcoref import FCorefModel
from .coref_models.modeling_lingmess import LingMessModel
from .utilities.collate import DynamicBatchSampler, LeftOversCollator, PadCollator
from .utilities.util import (
    align_to_char_level,
    create_clusters,
    create_mention_to_antecedent,
    encode,
    set_seed,
)

# Setup logging
logger = logging_getLogger(__name__)
logging_basicConfig(
    format="%(asctime)s - %(levelname)s - \t %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging_INFO,
)


class CorefResult:
    def __init__(self, text, clusters, char_map, reverse_char_map, coref_logit, text_idx):
        self.text = text
        self.clusters = clusters
        self.char_map = char_map
        self.reverse_char_map = reverse_char_map
        self.coref_logit = coref_logit
        self.text_idx = text_idx

    def get_clusters(self, as_strings=True):
        if not as_strings:
            computed_return_value = [[self.char_map[mention][1] for mention in cluster] for cluster in self.clusters]
            return computed_return_value

        computed_return_value = [
            [
                self.text[self.char_map[mention][1][0] : self.char_map[mention][1][1]]
                for mention in cluster
                if None not in self.char_map[mention]
            ]
            for cluster in self.clusters
        ]
        return computed_return_value

    def get_logit(self, span_i, span_j):
        if span_i not in self.reverse_char_map:
            raise ValueError(f'span_i="{self.text[span_i[0] : span_i[1]]}" is not an entity in this model!')
        if span_j not in self.reverse_char_map:
            raise ValueError(f'span_i="{self.text[span_j[0] : span_j[1]]}" is not an entity in this model!')

        span_i_idx = self.reverse_char_map[span_i][0]  # 0 is to get the span index
        span_j_idx = self.reverse_char_map[span_j][0]

        if span_i_idx < span_j_idx:
            computed_return_value = self.coref_logit[span_j_idx, span_i_idx]
            return computed_return_value

        computed_return_value = self.coref_logit[span_i_idx, span_j_idx]
        return computed_return_value

    def get_resolved_text(self):
        """Replace coreferent mentions with their canonical antecedent (longest mention in each cluster)."""
        char_clusters = self.get_clusters(as_strings=False)
        str_clusters = self.get_clusters(as_strings=True)

        if not char_clusters:
            return self.text

        # Build replacement list: for each cluster, the longest mention is the antecedent
        replacements = []
        for char_cluster, str_cluster in zip(char_clusters, str_clusters, strict=False):
            if len(char_cluster) < 2:
                continue
            # Pick longest mention as canonical
            canonical_idx = max(range(len(str_cluster)), key=lambda i: len(str_cluster[i]))
            canonical = str_cluster[canonical_idx]
            for i, (start, end) in enumerate(char_cluster):
                if i != canonical_idx:
                    replacements.append((start, end, canonical))

        if not replacements:
            return self.text

        # Sort by position descending so replacements don't shift indices
        replacements.sort(key=lambda r: r[0], reverse=True)
        resolved = self.text
        for start, end, replacement in replacements:
            resolved = resolved[:start] + replacement + resolved[end:]
        return resolved

    def __str__(self):
        if len(self.text) > 50:
            text_to_print = f"{self.text[:50]}..."
        else:
            text_to_print = self.text
        computed_return_value = f'CorefResult(text="{text_to_print}", clusters={self.get_clusters()})'
        return computed_return_value

    def __repr__(self):
        computed_return_value = self.__str__()
        return computed_return_value


class CorefModel:
    def __init__(
        self,
        model_name_or_path,
        coref_class,
        collator_class,
        enable_progress_bar,
        device=False,
        nlp="en_core_web_sm",
    ):
        if nlp is None:
            nlp = "en_core_web_sm"
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.seed = 42
        self.set_device()
        self.enable_progress_bar = enable_progress_bar

        config = AutoConfig.from_pretrained(self.model_name_or_path)
        self.max_segment_len = config.coref_head.get("max_segment_len", False)
        self.max_doc_len = config.coref_head.get("max_doc_len", False) if "max_doc_len" in config.coref_head else False

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True, add_prefix_space=True, verbose=False)

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
        if nlp == "en_core_web_sm":
            self.nlp = False
            logger.warning("You didn't specify a spacy model, you'll need to provide tokenized text in the `predict` function.")
        elif isinstance(nlp, Language):
            self.nlp = nlp
        else:
            try:
                self.nlp = spacy_load(nlp, exclude=["tagger", "parser", "lemmatizer", "ner", "textcat"])
            except OSError:
                # TODO: this is a workaround it is not clear how to add "en_core_web_sm" to setup.py
                download(nlp)
                self.nlp = spacy_load(nlp, exclude=["tagger", "parser", "lemmatizer", "ner", "textcat"])

        self.model = coref_class.from_pretrained(
            self.model_name_or_path,
            config=config,
        )
        self.model.to(self.device)

        t_params, h_params = [p / 1000000 for p in self.model.num_parameters()]
        logger.info(f"Model Parameters: {t_params + h_params:.1f}M, Transformer: {t_params:.1f}M, Coref head: {h_params:.1f}M")

        set_seed(self)
        transformers_logging.set_verbosity_error()

    def set_device(self):
        if self.device is False:
            self.device = "cuda" if torch_cuda.is_available() else "cpu"
        self.device = torch_device(self.device)
        self.n_gpu = torch_cuda.device_count()
        return False

    def create_dataset(self, texts, is_split_into_words):
        logger.info(f"Tokenize {len(texts)} inputs...")

        # Save original text ordering for later use
        dataset = {"text": texts, "idx": range(len(texts))}
        if is_split_into_words:
            dataset["tokens"] = texts

        dataset = Dataset.from_dict(dataset)
        dataset = dataset.map(
            encode,
            batched=True,
            batch_size=10000,
            fn_kwargs={
                "tokenizer": self.tokenizer,
                "nlp": self.nlp if not is_split_into_words else False,
            },
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
        subtoken_map = batch.get("subtoken_map", False)
        token_to_char = batch.get("offset_mapping", {})
        idxs = batch.get("idx", [])
        with torch_no_grad():
            outputs = self.model(batch, return_all_outputs=True)

        outputs_np = tuple(tensor.cpu().numpy() for tensor in outputs)

        span_starts, span_ends, mention_logits, coref_logits = outputs_np
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
        else:
            for batch in dataloader:
                results.extend(self.batch_inference(batch))

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
        if output_file is None:
            output_file = ""

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
                    computed_return_value = len(texts[0]) == 0 or isinstance(texts[0][0], str)
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

        if is_split_into_words:
            is_batched = isinstance(texts, (list, tuple)) and texts and isinstance(texts[0], (list, tuple))
        else:
            is_batched = isinstance(texts, (list, tuple))

        if not is_batched:
            texts = [texts]

        dataset = self.create_dataset(texts, is_split_into_words)
        dataloader = self.prepare_batches(dataset, max_tokens_in_batch)

        preds = self.inference(dataloader)
        if output_file != "":
            with open(output_file, "w") as f:
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
    def __init__(
        self,
        model_name_or_path="biu-nlp/f-coref",
        device=False,
        nlp="en_core_web_sm",
        enable_progress_bar=True,
    ):
        super().__init__(
            model_name_or_path,
            FCorefModel,
            LeftOversCollator,
            enable_progress_bar,
            device,
            nlp,
        )


class LingMessCoref(CorefModel):
    def __init__(
        self,
        model_name_or_path="biu-nlp/lingmess-coref",
        device=False,
        nlp="en_core_web_sm",
        enable_progress_bar=True,
    ):
        super().__init__(
            model_name_or_path,
            LingMessModel,
            PadCollator,
            enable_progress_bar,
            device,
            nlp,
        )
