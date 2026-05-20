<div align="center">
  <h1>
    <img src="assets/logo.png" alt="TraceLift logo" width="52" align="center">
    Correct Is Not Enough:<br>
    Training Reasoning Planners with Executor-Grounded Rewards
  </h1>

  <p>
    <strong>TraceLift</strong> trains reasoning planners with rubric-based Reason RMs and executor-grounded utility.
  </p>

  <p>
    Tianyang Han<sup>1,*</sup> · Hengyu Shi<sup>2,*</sup> · Junjie Hu<sup>2,*</sup> ·
    Xu Yang<sup>1</sup> · Zhiling Wang<sup>2</sup> · Junhao Su<sup>2,‡,†</sup>
  </p>

  <p>
    <sup>1</sup>D<sup>4</sup> Lab &nbsp;&nbsp; <sup>2</sup>Independent Researcher<br>
    <sup>*</sup>Equal contribution &nbsp;&nbsp; <sup>‡</sup>Corresponding author &nbsp;&nbsp; <sup>†</sup>Project leader
  </p>

  <p>
    <a href="https://arxiv.org/abs/2605.03862">📄 arXiv</a> ·
    <a href="https://huggingface.co/ScottHan/TraceLift">🤗 HF Model</a> ·
    <a href="https://huggingface.co/datasets/ScottHan/TraceLift-Groups">📦 HF Dataset</a>
  </p>
</div>

## 🗺️ Plan

- ✅ 📄 Release paper to arXiv
- ✅ 📦 Release TraceLift-Groups data
- ✅ 🤗 Release Reason RM checkpoints
- ✅ 💻 Release training code

## ⚙️ Install

```bash
git clone https://github.com/MasaiahHan/TraceLift.git
cd TraceLift
conda create -n tracelift python=3.10 -y
conda activate tracelift
python -m pip install -r requirements.txt
python -m pip install -e .
```

Optional dependencies:

```bash
python -m pip install -e '.[deepspeed]'
python -m pip install -e '.[qlora]'
```

Recommended base packages are `torch`, `transformers`, `datasets`, `peft`, `accelerate`, `trl`, and `safetensors`. DeepSpeed is only needed when using the provided multi-GPU DeepSpeed configs.

## 🚀 Quick Start

Download the released assets from Hugging Face:

- 🤗 Models: https://huggingface.co/ScottHan/TraceLift
- 📦 Data: https://huggingface.co/ScottHan/TraceLift

After download, place or symlink the assets into the expected paths:

```text
data/
reward_models/
```

Minimal Reason RM loading example:

```python
import torch
from transformers import AutoTokenizer

from reasonrm.modeling_reward import Qwen2ForReasonRewardModel

model = Qwen2ForReasonRewardModel.from_pretrained(
    "reward_models/math-rm-full-ce",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("reward_models/math-rm-full-ce")
```

## 📚 Data Format

The released data is already grouped. Each group contains one positive reasoning trace and a bank of negative reasoning traces with rubric labels.

Code data:

- `data/code_rm/train_groups.jsonl`: 2,198 reward-model training groups.
- `data/code_rm/dev_groups.jsonl`: 244 validation groups.
- `data/code_rm/prepared/train_groups_seed18_new_full.jsonl`: 2,998 code GRPO training groups.
- `data/code_rm/code_test_oracles.full.statement.jsonl`: code reward tests extracted from problem statements.

Math data:

- `data/math_rm/gsm8k_even_3000/train_groups_min4.jsonl`: 2,690 reward-model/GRPO training groups.
- `data/math_rm/gsm8k_even_3000/dev_groups_min4.jsonl`: 299 validation groups.
- `data/math_rm/gsm8k_even_3000/prepare_summary_min4.json`: data preparation summary.

Minimal group schema:

```json
{
  "problem_id": "example_000001",
  "source": "code",
  "task_type": "algorithm",
  "problem": "Problem statement...",
  "reference_solution": "Reference answer or solution when available",
  "positive_pool": [
    {
      "reasoning": "A correct reasoning trace.",
      "rubric": {
        "task_understanding": 4,
        "plan_quality": 4,
        "step_coherence": 4,
        "action_support": 4,
        "non_leakage": 4,
        "total": 1.0
      }
    }
  ],
  "negative_bank": [
    {
      "reasoning": "A flawed reasoning trace.",
      "negative_kind": "wrong_algorithm_choice",
      "rubric": {
        "task_understanding": 2,
        "plan_quality": 1,
        "step_coherence": 1,
        "action_support": 1,
        "non_leakage": 3,
        "total": 0.35
      }
    }
  ]
}
```

Rubric dimension labels are integers from `0` to `4`. The `total` score is normalized to `[0, 1]`.

## 🧠 Reward Model

The Reason RM wraps a Qwen2 backbone and replaces the language-model head with reward heads:

- Five rubric classification heads, one per dimension.
- One total-score regression head.
- Last-token pooling over the final hidden states.

For each dimension, the model predicts logits over labels `0..4`. The rubric-derived score is the weighted average of expected dimension scores, normalized to `[0, 1]`. When the total head is enabled, the final RM score is:

```text
rm_score = 0.5 * rubric_derived_score + 0.5 * sigmoid(total_head)
```

Training loss:

```text
L = loss_dim_weight * L_dim
  + loss_total_weight * L_total
  + loss_posneg_weight * L_posneg
  + loss_negneg_weight * L_negneg
```

Default settings:

- `L_dim`: cross-entropy over the five rubric heads.
- `L_total`: Huber loss between `sigmoid(total_head)` and normalized total score, with `delta = 1.0`.
- `L_posneg`: pairwise positive-vs-negative ranking loss.
- `L_negneg`: disabled by default.
- Default loss weights: `1.0`, `0.5`, `0.7`, `0.0`.

## 🏋️ Train Reward Models

Code RM:

```bash
BASE_MODEL=Qwen/Qwen2.5-7B-Instruct \
NPROC_PER_NODE=8 \
DEEPSPEED_CONFIG=configs/deepspeed_zero3_bf16.json \
bash configs/rm/train_code_rm_qwen25_7b_instruct.sh
```

Math RM:

```bash
BASE_MODEL=Qwen/Qwen2.5-7B-Instruct \
NPROC_PER_NODE=8 \
DEEPSPEED_CONFIG=configs/deepspeed_zero3_bf16.json \
bash configs/rm/train_math_rm_qwen25_7b_instruct.sh
```

Common overrides:

```bash
OUTPUT_DIR=outputs/my-rm
TRAIN_FILE=data/code_rm/train_groups.jsonl
USE_LORA=True
LORA_R=32
LORA_ALPHA=64
MAX_STEPS=1000
MAX_LENGTH=3072
```

Run a command check without launching training:

```bash
DRY_RUN=1 bash configs/rm/train_code_rm_qwen25_7b_instruct.sh
```

## 🎯 GRPO Training

TraceLift GRPO uses two external services during training:

- `EXECUTOR_BASE_URL`: OpenAI-compatible executor or solver service.
- `RM_SCORE_URL`: Reason RM scoring service for reason+executor training.

The executor is task-specific:

- Code: receives a programming problem and policy reasoning, then returns a complete Python program.
- Math: receives a math problem and policy reasoning, then returns the final numeric answer.

The RM scoring service should accept batched items:

```json
{
  "items": [
    {
      "problem": "Problem statement...",
      "reasoning": "Policy reasoning...",
      "task_name": "math",
      "task_type": "math"
    }
  ],
  "max_length": 3072,
  "batch_size": 1
}
```

It may return either a list of scores or an object with a `scores`, `results`, `outputs`, or `data` field. Scores should be floats in `[0, 1]`.

### Code GRPO

Executor-only baseline:

```bash
MODEL_PATH=Qwen/Qwen2.5-7B \
EXECUTOR_BASE_URL=http://localhost:8000/v1 \
EXECUTOR_MODEL=qwen2.5-7b \
NPROC_PER_NODE=8 \
bash configs/grpo/code_exec_only.sh
```

TraceLift reason+executor:

```bash
MODEL_PATH=Qwen/Qwen2.5-7B \
EXECUTOR_BASE_URL=http://localhost:8000/v1 \
EXECUTOR_MODEL=qwen2.5-7b \
RM_SCORE_URL=http://localhost:8001/score \
NPROC_PER_NODE=8 \
bash configs/grpo/code_reason_exec.sh
```

The code trainer asks the policy to write reasoning only. The executor turns `problem + reasoning` into code, and local statement tests provide the executable reward. The reason reward uses the RM score gated by executor utility.

### Math GRPO

Executor-only baseline:

```bash
MODEL_PATH=Qwen/Qwen2.5-7B \
EXECUTOR_BASE_URL=http://localhost:8000/v1 \
EXECUTOR_MODEL=qwen2.5-7b \
NPROC_PER_NODE=8 \
bash configs/grpo/math_exec_only.sh
```

TraceLift reason+executor:

```bash
MODEL_PATH=Qwen/Qwen2.5-7B \
EXECUTOR_BASE_URL=http://localhost:8000/v1 \
EXECUTOR_MODEL=qwen2.5-7b \
RM_SCORE_URL=http://localhost:8001/score \
NPROC_PER_NODE=8 \
bash configs/grpo/math_reason_exec.sh
```

The math trainer also asks the policy to write reasoning only. The solver consumes the reasoning and returns the final answer. The reason+executor reward uses:

```text
reward = 0.5 * e2e_reward + 0.5 * rm_score * uplift
uplift = solver_success(problem, reasoning) - solver_success(problem, no_reasoning)
```

Useful GRPO overrides:

```bash
OUTPUT_DIR=outputs/my-policy
MAX_STEPS=600
NUM_GENERATIONS=4
USE_LORA=true
LORA_R=16
LORA_ALPHA=32
EXECUTOR_REPEATS=3
BASELINE_REPEATS=3
```

Command check:

```bash
DRY_RUN=1 bash configs/grpo/code_reason_exec.sh
DRY_RUN=1 bash configs/grpo/math_reason_exec.sh
```

## 📝 Citation

If you find TraceLift useful for your research, please cite:

```bibtex
@misc{han2026correctisnotenough,
  title={Correct Is Not Enough: Training Reasoning Planners with Executor-Grounded Rewards},
  author={Han, Tianyang and Shi, Hengyu and Hu, Junjie and Yang, Xu and Wang, Zhiling and Su, Junhao},
  year={2026},
  eprint={2605.03862},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2605.03862}
}
```
