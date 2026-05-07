import importlib.util
import inspect
import logging
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)

from .data import ProblemGroupDataset, ReasonRewardDataCollator
from .modeling_reward import DEFAULT_DIMENSION_NAMES, DEFAULT_DIMENSION_WEIGHTS, Qwen2ForReasonRewardModel


LOGGER = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "Local base model path"})
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional tokenizer path; defaults to model_name_or_path"},
    )
    load_in_4bit: bool = field(default=False)
    use_lora: bool = field(default=False)
    lora_r: int = field(default=32)
    lora_alpha: int = field(default=64)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: str = field(
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    use_total_head: bool = field(default=True)
    dimension_names: str = field(default=",".join(DEFAULT_DIMENSION_NAMES))
    dimension_weights: str = field(default=",".join(str(x) for x in DEFAULT_DIMENSION_WEIGHTS))


@dataclass
class DataArguments:
    train_file: str = field(metadata={"help": "Train problem-group json/jsonl"})
    eval_file: str = field(metadata={"help": "Eval problem-group json/jsonl"})
    max_length: int = field(default=3072)
    num_negatives: int = field(default=4)
    task_name: str = field(default="code")


@dataclass
class RewardTrainingArguments(TrainingArguments):
    dimension_loss_type: str = field(default="ce", metadata={"help": "Dimension loss: ce or mse"})
    loss_dim_weight: float = field(default=1.0)
    loss_total_weight: float = field(default=0.5)
    loss_posneg_weight: float = field(default=0.7)
    loss_negneg_weight: float = field(default=0.0)
    negneg_margin: float = field(default=0.15)
    huber_delta: float = field(default=1.0)


class ReasonRewardTrainer(Trainer):
    def _compute_group_losses(self, scores, gold_total, candidate_is_positive, group_sizes):
        posneg_losses = []
        negneg_losses = []
        offset = 0

        for group_size in group_sizes.tolist():
            group_scores = scores[offset : offset + group_size]
            group_gold = gold_total[offset : offset + group_size]
            group_positive_mask = candidate_is_positive[offset : offset + group_size]

            positive_indices = torch.nonzero(group_positive_mask, as_tuple=False).squeeze(-1)
            if positive_indices.numel() != 1:
                raise ValueError("each group must contain exactly one positive sample")

            pos_score = group_scores[positive_indices[0]]
            neg_scores = group_scores[~group_positive_mask]
            if neg_scores.numel() > 0:
                posneg_losses.append(-F.logsigmoid(pos_score - neg_scores).mean())

            if self.args.loss_negneg_weight > 0:
                neg_gold = group_gold[~group_positive_mask]
                for i in range(neg_scores.size(0)):
                    for j in range(i + 1, neg_scores.size(0)):
                        gap = neg_gold[i] - neg_gold[j]
                        if abs(float(gap)) < self.args.negneg_margin:
                            continue
                        if gap > 0:
                            higher_score = neg_scores[i]
                            lower_score = neg_scores[j]
                        else:
                            higher_score = neg_scores[j]
                            lower_score = neg_scores[i]
                        negneg_losses.append(-F.logsigmoid(higher_score - lower_score))

            offset += group_size

        zero = scores.new_zeros(())
        posneg_loss = torch.stack(posneg_losses).mean() if posneg_losses else zero
        negneg_loss = torch.stack(negneg_losses).mean() if negneg_losses else zero
        return posneg_loss, negneg_loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        inputs = dict(inputs)
        dimension_labels = inputs.pop("dimension_labels")
        gold_total = inputs.pop("gold_total")
        candidate_is_positive = inputs.pop("candidate_is_positive")
        group_sizes = inputs.pop("group_sizes")

        outputs = model(**inputs, return_dict=True)

        if self.args.dimension_loss_type == "ce":
            dim_losses = []
            for dim_index, logits in enumerate(outputs.dimension_logits):
                dim_losses.append(F.cross_entropy(logits, dimension_labels[:, dim_index]))
            loss_dim = torch.stack(dim_losses).sum()
        elif self.args.dimension_loss_type == "mse":
            dimension_targets = dimension_labels.to(dtype=outputs.expected_dimension_scores.dtype)
            dim_losses = F.mse_loss(
                outputs.expected_dimension_scores,
                dimension_targets,
                reduction="none",
            ).mean(dim=0)
            loss_dim = dim_losses.sum()
        else:
            raise ValueError("dimension_loss_type must be 'ce' or 'mse'")

        regression_scores = outputs.total_head_scores
        if regression_scores is None:
            regression_scores = outputs.derived_total_scores
        loss_total = F.huber_loss(regression_scores, gold_total, delta=self.args.huber_delta)

        loss_posneg, loss_negneg = self._compute_group_losses(
            outputs.total_scores,
            gold_total,
            candidate_is_positive,
            group_sizes,
        )

        loss = (
            self.args.loss_dim_weight * loss_dim
            + self.args.loss_total_weight * loss_total
            + self.args.loss_posneg_weight * loss_posneg
            + self.args.loss_negneg_weight * loss_negneg
        )

        if return_outputs:
            outputs.loss = loss
            return loss, outputs
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            loss = loss.mean().detach()

        if prediction_loss_only:
            return loss, None, None

        logits = outputs.total_scores.detach()
        labels = inputs["gold_total"].detach()
        return loss, logits, labels


def parse_csv_floats(raw):
    values = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError("expected at least one numeric value")
    return values


def parse_csv_strings(raw):
    values = [x.strip() for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError("expected at least one string value")
    return values


def normalize_training_arg_aliases(args):
    params = inspect.signature(TrainingArguments.__init__).parameters
    has_eval_strategy = "eval_strategy" in params
    has_evaluation_strategy = "evaluation_strategy" in params

    if has_eval_strategy and not has_evaluation_strategy:
        return ["--eval_strategy" if arg == "--evaluation_strategy" else arg for arg in args]
    if has_evaluation_strategy and not has_eval_strategy:
        return ["--evaluation_strategy" if arg == "--eval_strategy" else arg for arg in args]
    return args


def build_quantization_config(model_args):
    if not model_args.load_in_4bit:
        return None
    if importlib.util.find_spec("bitsandbytes") is None:
        raise ImportError("load_in_4bit=True requires bitsandbytes to be installed")
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def require_peft():
    try:
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    except ImportError as exc:
        raise ImportError("use_lora=True or load_in_4bit=True requires peft to be installed") from exc
    return LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training


def disable_peft_bitsandbytes_dispatch():
    """Keep regular LoRA usable in environments with a broken bitsandbytes install."""
    try:
        import peft.tuners.lora.model as lora_model
    except ImportError:
        return

    lora_model.is_bnb_available = lambda: False
    lora_model.is_bnb_4bit_available = lambda: False


def reward_modules_to_save(num_dimensions, use_total_head):
    modules = ["dimension_heads.{}".format(i) for i in range(num_dimensions)]
    if use_total_head:
        modules.append("total_head")
    return modules


def set_reward_heads_trainable(model):
    base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    for module_name in ("dimension_heads", "total_head"):
        module = getattr(base_model, module_name, None)
        if module is None:
            continue
        modules_to_save_found = False
        for child in module.modules():
            modules_to_save = getattr(child, "modules_to_save", None)
            if modules_to_save is None:
                continue
            modules_to_save_found = True
            for param in modules_to_save.parameters():
                param.requires_grad = True
        if modules_to_save_found:
            continue
        for param in module.parameters():
            param.requires_grad = True


def enable_input_require_grads_for_checkpointing(model):
    target = model
    if hasattr(target, "enable_input_require_grads"):
        target.enable_input_require_grads()
        return

    base_model = target.get_base_model() if hasattr(target, "get_base_model") else target
    if hasattr(base_model, "enable_input_require_grads"):
        base_model.enable_input_require_grads()


def trainer_tokenizer_kwarg(tokenizer):
    trainer_params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_params:
        return {"processing_class": tokenizer}
    return {"tokenizer": tokenizer}


def build_compute_metrics(num_negatives):
    group_size = num_negatives + 1

    def compute_metrics(eval_prediction):
        predictions = eval_prediction.predictions
        labels = eval_prediction.label_ids
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        predictions = np.asarray(predictions, dtype=np.float32).reshape(-1)
        labels = np.asarray(labels, dtype=np.float32).reshape(-1)
        metrics = {
            "total_mae": float(np.mean(np.abs(predictions - labels))),
            "total_rmse": float(np.sqrt(np.mean(np.square(predictions - labels)))),
        }

        usable = (len(predictions) // group_size) * group_size
        if usable:
            grouped_predictions = predictions[:usable].reshape(-1, group_size)
            pos_scores = grouped_predictions[:, :1]
            neg_scores = grouped_predictions[:, 1:]
            pair_correct = pos_scores > neg_scores
            metrics["posneg_pair_acc"] = float(np.mean(pair_correct))
            metrics["posneg_group_acc"] = float(np.mean(pos_scores[:, 0] > np.max(neg_scores, axis=1)))
        return metrics

    return compute_metrics


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, RewardTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses(
        args=normalize_training_arg_aliases(sys.argv[1:])
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )
    LOGGER.info("loading datasets")
    set_seed(training_args.seed)
    if training_args.remove_unused_columns:
        LOGGER.warning("forcing remove_unused_columns=False because the collator needs raw problem groups")
        training_args.remove_unused_columns = False
    if not training_args.label_names:
        training_args.label_names = ["dimension_labels", "gold_total", "candidate_is_positive", "group_sizes"]

    dimension_names = parse_csv_strings(model_args.dimension_names)
    dimension_weights = parse_csv_floats(model_args.dimension_weights)
    if len(dimension_names) != len(dimension_weights):
        raise ValueError("dimension_names and dimension_weights must match in length")

    tokenizer_name = model_args.tokenizer_name_or_path or model_args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=False)
    config.rm_dimension_names = dimension_names
    config.rm_dimension_weights = dimension_weights
    config.rm_num_labels = 5
    config.rm_use_total_head = model_args.use_total_head
    config.pad_token_id = tokenizer.pad_token_id
    if training_args.gradient_checkpointing:
        config.use_cache = False

    quantization_config = build_quantization_config(model_args)
    peft_components = None
    if model_args.load_in_4bit or model_args.use_lora:
        peft_components = require_peft()

    model = Qwen2ForReasonRewardModel.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        torch_dtype=torch.bfloat16 if training_args.bf16 else None,
        quantization_config=quantization_config,
        device_map="auto" if model_args.load_in_4bit else None,
    )

    if model_args.load_in_4bit:
        _, _, _, prepare_model_for_kbit_training = peft_components
        model = prepare_model_for_kbit_training(model)
        set_reward_heads_trainable(model)

    if model_args.use_lora:
        if not model_args.load_in_4bit:
            disable_peft_bitsandbytes_dispatch()
        LoraConfig, TaskType, get_peft_model, _ = peft_components
        target_modules = parse_csv_strings(model_args.lora_target_modules)
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=target_modules,
            modules_to_save=reward_modules_to_save(len(dimension_names), model_args.use_total_head),
        )
        model = get_peft_model(model, peft_config)
        set_reward_heads_trainable(model)
        if training_args.gradient_checkpointing:
            enable_input_require_grads_for_checkpointing(model)
        model.print_trainable_parameters()

    train_dataset = ProblemGroupDataset(data_args.train_file)
    eval_dataset = ProblemGroupDataset(data_args.eval_file)
    data_collator = ReasonRewardDataCollator(
        tokenizer=tokenizer,
        max_length=data_args.max_length,
        num_negatives=data_args.num_negatives,
        task_name=data_args.task_name,
        dimension_names=dimension_names,
    )

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(data_args.num_negatives),
    )
    trainer_kwargs.update(trainer_tokenizer_kwarg(tokenizer))
    trainer = ReasonRewardTrainer(**trainer_kwargs)

    trainer.train()
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
