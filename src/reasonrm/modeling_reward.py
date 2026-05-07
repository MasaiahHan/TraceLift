from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model, Qwen2PreTrainedModel
from transformers.utils import ModelOutput


DEFAULT_DIMENSION_NAMES = (
    "task_understanding",
    "plan_quality",
    "step_coherence",
    "action_support",
    "non_leakage",
)
DEFAULT_DIMENSION_WEIGHTS = (0.20, 0.20, 0.20, 0.20, 0.20)


def ensure_reward_config(config):
    if not hasattr(config, "rm_dimension_names"):
        config.rm_dimension_names = list(DEFAULT_DIMENSION_NAMES)
    if not hasattr(config, "rm_dimension_weights"):
        config.rm_dimension_weights = list(DEFAULT_DIMENSION_WEIGHTS)
    if not hasattr(config, "rm_num_labels"):
        config.rm_num_labels = 5
    if not hasattr(config, "rm_use_total_head"):
        config.rm_use_total_head = True
    if not hasattr(config, "classifier_dropout"):
        config.classifier_dropout = 0.0

    if len(config.rm_dimension_names) != len(config.rm_dimension_weights):
        raise ValueError("rm_dimension_names and rm_dimension_weights must have the same length")
    return config


@dataclass
class ReasonRewardModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    total_scores: Optional[torch.FloatTensor] = None
    derived_total_scores: Optional[torch.FloatTensor] = None
    total_head_scores: Optional[torch.FloatTensor] = None
    expected_dimension_scores: Optional[torch.FloatTensor] = None
    dimension_logits: Optional[Tuple[torch.FloatTensor, ...]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class Qwen2ForReasonRewardModel(Qwen2PreTrainedModel):
    def __init__(self, config):
        config = ensure_reward_config(config)
        super().__init__(config)
        self.model = Qwen2Model(config)
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.dimension_names = list(config.rm_dimension_names)
        self.dimension_heads = nn.ModuleList(
            [nn.Linear(config.hidden_size, config.rm_num_labels) for _ in self.dimension_names]
        )
        self.total_head = nn.Linear(config.hidden_size, 1) if config.rm_use_total_head else None

        self.register_buffer(
            "label_values",
            torch.arange(config.rm_num_labels, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "dimension_weights",
            torch.tensor(config.rm_dimension_weights, dtype=torch.float32),
            persistent=False,
        )
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def _pool_last_token(self, hidden_states, attention_mask):
        if attention_mask is None:
            return hidden_states[:, -1, :]

        seq_lens = attention_mask.to(hidden_states.device, dtype=torch.long).sum(dim=-1) - 1
        seq_lens = torch.clamp(seq_lens, min=0)
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        return hidden_states[batch_indices, seq_lens]

    def _expected_score(self, logits):
        probs = F.softmax(logits, dim=-1)
        return (probs * self.label_values.to(logits.device)).sum(dim=-1)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        pooled_state = self._pool_last_token(outputs.last_hidden_state, attention_mask)
        pooled_state = self.dropout(pooled_state)

        dimension_logits = tuple(head(pooled_state) for head in self.dimension_heads)
        expected_dimension_scores = torch.stack(
            [self._expected_score(logits) for logits in dimension_logits],
            dim=-1,
        )
        derived_total_scores = (
            expected_dimension_scores * self.dimension_weights.to(expected_dimension_scores.device)
        ).sum(dim=-1) / float(self.config.rm_num_labels - 1)

        total_head_scores = None
        total_scores = derived_total_scores
        if self.total_head is not None:
            total_head_scores = torch.sigmoid(self.total_head(pooled_state).squeeze(-1))
            total_scores = 0.5 * (derived_total_scores + total_head_scores)

        if not return_dict:
            return (total_scores, derived_total_scores, total_head_scores, expected_dimension_scores) + outputs[1:]

        return ReasonRewardModelOutput(
            total_scores=total_scores,
            derived_total_scores=derived_total_scores,
            total_head_scores=total_head_scores,
            expected_dimension_scores=expected_dimension_scores,
            dimension_logits=dimension_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.get("config")
        if config is not None:
            ensure_reward_config(config)
        kwargs.setdefault("ignore_mismatched_sizes", True)
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

    def score(self, input_ids, attention_mask=None):
        outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        return {
            name: logits
            for name, logits in zip(self.dimension_names, outputs.dimension_logits)
        }, outputs.total_scores
