"""Utilities for modeling lingmess."""

from math import sqrt as math_sqrt

from numpy import zeros as np_zeros
from torch import arange as torch_arange
from torch import cat as torch_cat
from torch import div as torch_div
from torch import einsum as torch_einsum
from torch import empty as torch_empty
from torch import gather as torch_gather
from torch import logsumexp as torch_logsumexp
from torch import matmul as torch_matmul
from torch import max as torch_max
from torch import nn
from torch import ones_like as torch_ones_like
from torch import sort as torch_sort
from torch import stack as torch_stack
from torch import sum as torch_sum
from torch import tensor as torch_tensor
from torch import topk as torch_topk
from torch import zeros as torch_zeros
from torch.nn import Dropout, LayerNorm, Linear, Module, init
from transformers import AutoModel, BertPreTrainedModel
from transformers.activations import ACT2FN

from ..utilities.consts import CATEGORIES, STOPWORDS
from ..utilities.util import (
    extract_clusters,
    extract_mentions_to_clusters,
    get_category_id,
    get_pronoun_id,
    mask_tensor,
)


class FullyConnectedLayer(Module):
    def __init__(self, config, input_dim, output_dim, dropout_prob):
        super(FullyConnectedLayer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_prob = dropout_prob

        self.dense = Linear(self.input_dim, self.output_dim)
        self.layer_norm = LayerNorm(self.output_dim, eps=config.layer_norm_eps)
        self.activation_func = ACT2FN[config.hidden_act]
        self.dropout = Dropout(self.dropout_prob)

    def forward(self, inputs):
        temp = inputs
        temp = self.dense(temp)
        temp = self.activation_func(temp)
        temp = self.layer_norm(temp)
        temp = self.dropout(temp)
        return temp


class LingMessModel(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.max_span_length = config.coref_head.get("max_span_length", 0)
        self.top_lambda = config.coref_head.get("top_lambda", False)
        self.ffnn_size = config.coref_head.get("ffnn_size", 0)
        self.dropout_prob = config.coref_head.get("dropout_prob", False)
        self.hidden_size = config.hidden_size

        self.num_cats = len(CATEGORIES) + 1  # +1 for ALL
        self.all_cats_size = self.ffnn_size * self.num_cats

        # this is how huggingface loading the class model and setting the name of the variable.
        base_model = AutoModel.from_config(config)
        LingMessModel.base_model_prefix = base_model.base_model_prefix
        LingMessModel.config_class = base_model.config_class
        setattr(self, self.base_model_prefix, base_model)

        self.start_mention_mlp = FullyConnectedLayer(config, self.hidden_size, self.ffnn_size, self.dropout_prob)
        self.end_mention_mlp = FullyConnectedLayer(config, self.hidden_size, self.ffnn_size, self.dropout_prob)

        self.mention_start_classifier = Linear(self.ffnn_size, 1)
        self.mention_end_classifier = Linear(self.ffnn_size, 1)
        self.mention_s2e_classifier = Linear(self.ffnn_size, self.ffnn_size)

        self.coref_start_all_mlps = FullyConnectedLayer(config, config.hidden_size, self.all_cats_size, self.dropout_prob)
        self.coref_end_all_mlps = FullyConnectedLayer(config, config.hidden_size, self.all_cats_size, self.dropout_prob)

        self.antecedent_s2s_all_weights = nn.Parameter(torch_empty((self.num_cats, self.ffnn_size, self.ffnn_size)))
        self.antecedent_e2e_all_weights = nn.Parameter(torch_empty((self.num_cats, self.ffnn_size, self.ffnn_size)))
        self.antecedent_s2e_all_weights = nn.Parameter(torch_empty((self.num_cats, self.ffnn_size, self.ffnn_size)))
        self.antecedent_e2s_all_weights = nn.Parameter(torch_empty((self.num_cats, self.ffnn_size, self.ffnn_size)))

        self.antecedent_s2s_all_biases = nn.Parameter(torch_empty((self.num_cats, self.ffnn_size)))
        self.antecedent_e2e_all_biases = nn.Parameter(torch_empty((self.num_cats, self.ffnn_size)))
        self.antecedent_s2e_all_biases = nn.Parameter(torch_empty((self.num_cats, self.ffnn_size)))
        self.antecedent_e2s_all_biases = nn.Parameter(torch_empty((self.num_cats, self.ffnn_size)))

        self.reset_parameters()
        self.all_tied_weights_keys = {}
        self.init_weights()

    def reset_parameters(self) -> bool:
        W = [
            self.antecedent_s2s_all_weights,
            self.antecedent_e2e_all_weights,
            self.antecedent_s2e_all_weights,
            self.antecedent_e2s_all_weights,
        ]

        B = [
            self.antecedent_s2s_all_biases,
            self.antecedent_e2e_all_biases,
            self.antecedent_s2e_all_biases,
            self.antecedent_e2s_all_biases,
        ]

        for w, b in zip(W, B, strict=False):
            init.kaiming_uniform_(w, a=math_sqrt(5))
            fan_in, _ = init._calculate_fan_in_and_fan_out(w)
            bound = 1 / math_sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(b, -bound, bound)
        return False

    def num_parameters(self) -> tuple:
        def head_filter(x):
            computed_return_value = x[1].requires_grad and any(hp in x[0] for hp in ["coref", "mention", "antecedent"])
            return computed_return_value

        head_params = filter(head_filter, self.named_parameters())
        head_params = sum(p.numel() for n, p in head_params)
        computed_return_value = super().num_parameters() - head_params, head_params
        return computed_return_value

    def get_span_mask(self, batch_size, k, max_k):
        """
        :param batch_size: int
        :param k: tensor of size [batch_size], with the required k for each example
        :param max_k: int
        :return: [batch_size, max_k] of zero-ones, where 1 stands for a valid span and 0 for a padded span
        """
        size = (batch_size, max_k)
        idx = torch_arange(max_k, device=self.device).unsqueeze(0).expand(size)
        len_expanded = k.unsqueeze(1).expand(size)
        computed_return_value = (idx < len_expanded).int()
        return computed_return_value

    def prune_topk_mentions(self, mention_logits, attention_mask):
        """
        :param mention_logits: Shape [batch_size, seq_length, seq_length]
        :param attention_mask: [batch_size, seq_length]
        :param top_lambda:
        :return:
        """
        batch_size, seq_length, _ = mention_logits.size()
        actual_seq_lengths = torch_sum(attention_mask, dim=-1)  # [batch_size]

        k = (actual_seq_lengths * self.top_lambda).int()  # [batch_size]
        max_k = int(torch_max(k))  # This is the k for the largest input in the batch, we will need to pad

        _, topk_1d_indices = torch_topk(mention_logits.view(batch_size, -1), dim=-1, k=max_k)  # [batch_size, max_k]

        span_mask = self.get_span_mask(batch_size, k, max_k)  # [batch_size, max_k]
        # drop the invalid indices and set them to the last index
        topk_1d_indices = (topk_1d_indices * span_mask) + (1 - span_mask) * (
            (seq_length**2) - 1
        )  # We take different k for each example
        # sorting for coref mention order
        sorted_topk_1d_indices, _ = torch_sort(topk_1d_indices, dim=-1)  # [batch_size, max_k]

        # gives the row index in 2D matrix
        topk_mention_start_ids = torch_div(sorted_topk_1d_indices, seq_length, rounding_mode="floor")  # [batch_size, max_k]
        topk_mention_end_ids = sorted_topk_1d_indices % seq_length  # [batch_size, max_k]

        topk_mention_logits = mention_logits[
            torch_arange(batch_size).unsqueeze(-1).expand(batch_size, max_k),
            topk_mention_start_ids,
            topk_mention_end_ids,
        ]  # [batch_size, max_k]

        # this is antecedents scores - rows mentions, cols coref mentions
        topk_mention_logits = topk_mention_logits.unsqueeze(-1) + topk_mention_logits.unsqueeze(-2)  # [batch_size, max_k, max_k]

        return (
            topk_mention_start_ids,
            topk_mention_end_ids,
            span_mask,
            topk_mention_logits,
            topk_1d_indices,
        )

    def mask_antecedent_logits(self, antecedent_logits, span_mask, categories_masks=False):
        if categories_masks is None:
            categories_masks = False
        antecedents_mask = torch_ones_like(antecedent_logits, dtype=self.dtype).tril(diagonal=-1)

        if categories_masks is not False:
            mask = antecedents_mask * span_mask.unsqueeze(1).unsqueeze(-1)
            mask *= categories_masks
        else:
            mask = antecedents_mask * span_mask.unsqueeze(-1)

        antecedent_logits = mask_tensor(antecedent_logits, mask)
        return antecedent_logits

    def get_clusters_labels(self, span_starts, span_ends, all_clusters):
        """
        :param span_starts: [batch_size, max_k]
        :param span_ends: [batch_size, max_k]
        :param all_clusters: [batch_size, max_cluster_size, max_clusters_num, 2]
        :return: [batch_size, max_k, max_k + 1] - [b, i, j] == 1 if j is antecedent of i
        """
        batch_size, max_k = span_starts.size()
        new_cluster_labels = np_zeros((batch_size, max_k, max_k))

        span_starts_cpu = span_starts.cpu().tolist()
        span_ends_cpu = span_ends.cpu().tolist()
        all_clusters_cpu = all_clusters.cpu().tolist()

        for b, (starts, ends, gold_clusters) in enumerate(zip(span_starts_cpu, span_ends_cpu, all_clusters_cpu, strict=False)):
            gold_clusters = extract_clusters(gold_clusters)
            mention_to_gold_clusters = extract_mentions_to_clusters(gold_clusters)
            for i, (start, end) in enumerate(zip(starts, ends, strict=False)):
                if (start, end) not in mention_to_gold_clusters:
                    continue
                for j, (a_start, a_end) in enumerate(list(zip(starts, ends, strict=False))[:i]):
                    if (a_start, a_end) in mention_to_gold_clusters[(start, end)]:
                        new_cluster_labels[b, i, j] = 1

        new_cluster_labels = torch_tensor(new_cluster_labels, device=self.device)
        return new_cluster_labels

    def get_categories_labels(self, tokens, subtoken_map, new_token_map, span_starts, span_ends):
        batch_size, max_k = span_starts.size()

        spans = []
        for b, (starts, ends) in enumerate(zip(span_starts.cpu().tolist(), span_ends.cpu().tolist(), strict=False)):
            doc_spans = []
            for start, end in zip(starts, ends, strict=False):
                token_indices = [new_token_map[b].get(idx, False) for idx in set(subtoken_map[b][start : end + 1]) - {False}]
                span = {tokens[b][idx].lower() for idx in token_indices if idx is not None}
                pronoun_id = get_pronoun_id(span)
                doc_spans.append((span - STOPWORDS, pronoun_id))
            spans.append(doc_spans)

        categories_labels = np_zeros((batch_size, max_k, max_k)) - 1
        for b in range(batch_size):
            for i in range(max_k):
                for j in list(range(max_k))[:i]:
                    categories_labels[b, i, j] = get_category_id(spans[b][i], spans[b][j])

        categories_labels = torch_tensor(categories_labels, device=self.device)
        categories_masks = [categories_labels == cat_id for cat_id in range(self.num_cats - 1)] + [categories_labels != -1]
        categories_masks = torch_stack(categories_masks, dim=1).int()
        return categories_labels, categories_masks

    def get_marginal_log_likelihood_loss(self, logits, labels, span_mask):
        gold_coref_logits = mask_tensor(logits, labels)  # [batch_size, num_cats + 1, max_k, max_k]

        gold_log_sum_exp = torch_logsumexp(gold_coref_logits, dim=-1)  # [batch_size, num_cats + 1, max_k]
        all_log_sum_exp = torch_logsumexp(logits, dim=-1)  # [batch_size, num_cats + 1, max_k]
        losses = all_log_sum_exp - gold_log_sum_exp  # [batch_size, num_cats + 1, max_k]

        # zero the loss of padded spans
        span_mask = span_mask.unsqueeze(1)  # [batch_size, 1, max_k]
        losses = losses * span_mask  # [batch_size, num_cats, max_k]

        # normalize loss by spans
        per_span_loss = losses.mean(dim=-1)  # [batch_size, num_cats + 1]

        # normalize loss by document
        loss_per_cat = per_span_loss.mean(dim=0)  # [num_cats + 1]

        # normalize loss by category
        loss = loss_per_cat.sum()
        return loss

    def get_mention_mask(self, mention_logits_or_weights):
        """
        Returns a tensor of size [batch_size, seq_length, seq_length] where valid spans
        (start <= end < start + max_span_length) are 1 and the rest are 0
        :param mention_logits_or_weights: Either the span mention logits or weights, size [batch_size, seq_length, seq_length]
        """
        mention_mask = torch_ones_like(mention_logits_or_weights, dtype=self.dtype)
        mention_mask = mention_mask.triu(diagonal=0)
        mention_mask = mention_mask.tril(diagonal=self.max_span_length - 1)
        return mention_mask

    def calc_mention_logits(self, start_mention_reps, end_mention_reps):
        start_mention_logits = self.mention_start_classifier(start_mention_reps).squeeze(-1)  # [batch_size, seq_length]
        end_mention_logits = self.mention_end_classifier(end_mention_reps).squeeze(-1)  # [batch_size, seq_length]

        temp = self.mention_s2e_classifier(start_mention_reps)  # [batch_size, seq_length]
        joint_mention_logits = torch_matmul(temp, end_mention_reps.permute([0, 2, 1]))  # [batch_size, seq_length, seq_length]

        mention_logits = joint_mention_logits + start_mention_logits.unsqueeze(-1) + end_mention_logits.unsqueeze(-2)
        mention_mask = self.get_mention_mask(mention_logits)  # [batch_size, seq_length, seq_length]
        mention_logits = mask_tensor(mention_logits, mention_mask)  # [batch_size, seq_length, seq_length]
        return mention_logits

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_cats, self.ffnn_size)
        x = x.view(*new_x_shape)
        computed_return_value = x.permute(0, 2, 1, 3)
        return computed_return_value  # bnkf/bnlg

    def calc_coref_logits(self, start_reps, end_reps):
        # see discussion on einsum: https://discuss.pytorch.org/t/batch-matrix-multiplication-of-3d-tensors/153644/4

        all_starts = self.transpose_for_scores(self.coref_start_all_mlps(start_reps))
        all_ends = self.transpose_for_scores(self.coref_end_all_mlps(end_reps))

        logits = (
            torch_einsum(
                "bnkf, nfg, bnlg -> bnkl",
                all_starts,
                self.antecedent_s2s_all_weights,
                all_starts,
            )
            + torch_einsum(
                "bnkf, nfg, bnlg -> bnkl",
                all_ends,
                self.antecedent_e2e_all_weights,
                all_ends,
            )
            + torch_einsum(
                "bnkf, nfg, bnlg -> bnkl",
                all_starts,
                self.antecedent_s2e_all_weights,
                all_ends,
            )
            + torch_einsum(
                "bnkf, nfg, bnlg -> bnkl",
                all_ends,
                self.antecedent_e2s_all_weights,
                all_starts,
            )
        )

        biases = (
            torch_einsum("bnkf, nf -> bnk", all_starts, self.antecedent_s2s_all_biases).unsqueeze(-2)
            + torch_einsum("bnkf, nf -> bnk", all_ends, self.antecedent_e2e_all_biases).unsqueeze(-2)
            + torch_einsum("bnkf, nf -> bnk", all_ends, self.antecedent_s2e_all_biases).unsqueeze(-2)
            + torch_einsum("bnkf, nf -> bnk", all_starts, self.antecedent_e2s_all_biases).unsqueeze(-2)
        )

        computed_return_value = logits + biases
        return computed_return_value

    def get_all_labels(self, clusters_labels, categories_masks):
        batch_size, max_k, _ = clusters_labels.size()

        categories_labels = clusters_labels.unsqueeze(1).repeat(1, self.num_cats, 1, 1) * categories_masks
        all_labels = torch_cat(
            (categories_labels, clusters_labels.unsqueeze(1)), dim=1
        )  # for the combined loss (L_coref + L_tasks)

        # null cluster
        zeros = torch_zeros((batch_size, self.num_cats + 1, max_k, 1), device=self.device)
        all_labels = torch_cat((all_labels, zeros), dim=-1)  # [batch_size, num_cats + 1, max_k, max_k + 1]
        no_antecedents = 1 - torch_sum(all_labels, dim=-1).bool().float()
        all_labels[:, :, :, -1] = no_antecedents

        return all_labels

    def forward_transformer(self, batch):
        input_ids = batch.get("input_ids", [])
        attention_mask = batch.get("attention_mask", False)

        if "leftovers" not in batch:
            outputs = self.base_model(input_ids, attention_mask=attention_mask)
            sequence_output = outputs.last_hidden_state
        else:
            docs, segments, segment_len = input_ids.size()
            input_ids, attention_mask = (
                input_ids.view(-1, segment_len),
                attention_mask.view(-1, segment_len),
            )

            outputs = self.base_model(input_ids, attention_mask=attention_mask)
            sequence_output = outputs.last_hidden_state

            attention_mask = attention_mask.view((docs, segments * segment_len))  # [docs, seq_len]
            sequence_output = sequence_output.view((docs, segments * segment_len, -1))  # [docs, seq_len, dim]

            leftovers_ids, leftovers_mask = (
                batch.get("leftovers", {}).get("input_ids", []),
                batch.get("leftovers", {}).get("attention_mask", False),
            )
            if len(leftovers_ids) > 0:
                res_outputs = self.base_model(leftovers_ids, attention_mask=leftovers_mask)
                res_sequence_output = res_outputs.last_hidden_state

                attention_mask = torch_cat([attention_mask, leftovers_mask], dim=1)
                sequence_output = torch_cat([sequence_output, res_sequence_output], dim=1)

        return sequence_output, attention_mask

    def forward(self, batch, gold_clusters=False, return_all_outputs=False):
        if gold_clusters is None:
            gold_clusters = False
        tokens, subtoken_map, new_token_map = (
            batch.get("tokens", 0),
            batch.get("subtoken_map", False),
            batch.get("new_token_map", False),
        )

        sequence_output, attention_mask = self.forward_transformer(batch)

        # Compute representations
        start_mention_reps = self.start_mention_mlp(sequence_output)
        end_mention_reps = self.end_mention_mlp(sequence_output)

        # mention scores
        mention_logits = self.calc_mention_logits(start_mention_reps, end_mention_reps)

        # prune mentions
        (
            mention_start_ids,
            mention_end_ids,
            span_mask,
            topk_mention_logits,
            topk_1d_indices,
        ) = self.prune_topk_mentions(mention_logits, attention_mask)

        categories_labels, categories_masks = self.get_categories_labels(
            tokens, subtoken_map, new_token_map, mention_start_ids, mention_end_ids
        )

        batch_size, max_k = mention_start_ids.size()
        size = (batch_size, max_k, self.hidden_size)
        # gather reps
        topk_start_reps = torch_gather(sequence_output, dim=1, index=mention_start_ids.unsqueeze(-1).expand(size))
        topk_end_reps = torch_gather(sequence_output, dim=1, index=mention_end_ids.unsqueeze(-1).expand(size))

        # antecedent scores by category
        categories_logits = self.calc_coref_logits(topk_start_reps, topk_end_reps)

        final_logits = categories_logits * categories_masks
        final_logits = final_logits.sum(dim=1) + topk_mention_logits
        categories_logits = categories_logits + topk_mention_logits.unsqueeze(1)

        # lower logits of padded spans or different category.
        final_logits = self.mask_antecedent_logits(final_logits, span_mask)
        categories_logits = self.mask_antecedent_logits(categories_logits, span_mask, categories_masks)

        # adding zero logits for null span
        final_logits = torch_cat(
            (final_logits, torch_zeros((batch_size, max_k, 1), device=self.device)),
            dim=-1,
        )  # [batch_size, max_k, max_k + 1]
        categories_logits = torch_cat(
            (
                categories_logits,
                torch_zeros((batch_size, self.num_cats, max_k, 1), device=self.device),
            ),
            dim=-1,
        )  # [batch_size, num_cats, max_k, max_k + 1]

        if return_all_outputs:
            outputs = (mention_start_ids, mention_end_ids, mention_logits, final_logits)
        else:
            outputs = tuple()

        if gold_clusters is not False:
            clusters_labels = self.get_clusters_labels(mention_start_ids, mention_end_ids, gold_clusters)
            all_labels = self.get_all_labels(clusters_labels, categories_masks)
            all_logits = torch_cat((categories_logits, final_logits.unsqueeze(1)), dim=1)

            loss = self.get_marginal_log_likelihood_loss(all_logits, all_labels, span_mask)
            outputs = (
                (loss,)
                + outputs
                + (
                    categories_labels,
                    clusters_labels,
                )
            )

        return outputs
