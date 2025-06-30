# Knowledge Distillation with torchtune
This repository extends the official [torchtune](https://github.com/pytorch/torchtune) **knowledge distillation recipe for single devices** and the torchtune **loss module** with **open-source contributions** aimed at improving flexibility, transparency, and performance. This repository contains implementations and experiments on **knowledge distillation**, where a Llama 3.2 3B model is distilled into a Llama 3.2 1B student model on a single GPU.

## Key Contribution
- Flexible **layer-to-layer alignment** between teacher and student models  
  > Allows custom mappings across unequal layer depths (e.g., 16-layer student ↔ 28-layer teacher), supporting partial hidden state supervision.

- Support for multiple **divergence-based loss functions**  
  > Includes Forward KL, Reverse KL, Symmetric KL (λ-balanced), Jensen-Shannon Divergence (JSD), and Total Variation Distance (TVD) as plug-and-play modules.

- **Intermediate representation (IR) supervision** via cosine similarity  
  > Cosine loss between normalized projected hidden states enhances semantic transfer across architectures, replacing underweighted MSE.

- Modular **loss scaling** framework  
  > Enables manual or dynamic weighting of cross-entropy, KD, and IR losses for interpretable, balanced training dynamics.

- Memory-efficient training with **chunked loss computation** and structured logging  
  > Processes long sequences in slices for stability and supports fine-grained loss reporting via Weights & Biases.

Core contributions are implemented in the modified `knowledge_distillation_single_device.py` and `kd_losses.py` files.

---


## What’s different from the original `torchtune` upstream?

| Area | `torchtune` | **This repo** |
|------|---------------------|---------------|
| Knowledge-transfer loss | Forward KL only | Forward / Reverse / Symmetric KL, Jensen-Shannon (JS), Total Variation (TVD) *(see `src/kd_losses.py`)* |
| Intermediate-representation loss | — | Cosine similarity on chosen hidden-state pairs, with optional projection & layer-norm |
| Layer alignment | — | `layer_mapping` config lets you align **any** student layer to **any** teacher layer (e.g. `[[0,15],[1,20],[2,25]]`) |
| Loss weighting | Fixed `(1−kd)·CE + kd·KD` | `ce / kd / ir` **scales** or fallback `kd_ratio` |
| Memory handling | Full-sequence logits | Works with chunked logits for long contexts |
| Logging | Overall loss | Per-component CE / KD / IR, grad-norm, tokens/s – via Weights & Biases or stdout |
| Hidden states | Off | Optional `output_hidden_states` flag per model |
| Code footprint | 1 recipe | 1 recipe **+** modular `kd_losses.py` (plug-in for other projects) |

---

## Overview

We distill:
- **Teacher:** Llama 3.2 3B (baseline + LoRA fine-tuned)
- **Student:** Llama 3.2 1B (baseline + LoRA fine-tuned)

Trained on the `alpaca.cleaned.dataset` and evaluated using:
- **TruthfulQA**
- **HellaSwag**
- **CommonSenseQA**

### Best performance for distilled models:

From the `3B-LoRA-to-1B` configuration using Jensen-Shannon divergence and cosine-aligned intermediate representation loss:

- **TruthfulQA (mc2 acc):** 0.4652
- **HellaSwag (acc):** 0.4583
- **CommonSenseQA (acc):** 0.5627

This setup used a custom layer mapping strategy and manual loss scaling with:
- `ce: 0.5`, `kd: 0.3`, `ir: 0.2`
- Layer mapping: `[[0,15], [1,20], [2,25]]` (early student layers → deeper teacher layers)

---

## Key implementation details
Everything below is adjustable in configs/distillation.yaml – no Python edits required.

### Emit hidden states
```yaml
model:
  _component_: torchtune.models.llama3_1b
  output_hidden_states: true

teacher_model:
  _component_: torchtune.models.llama3_3b
  output_hidden_states: true
```

### Custom layer mapping:
```yaml
  - [0, 15]   # student layer 0 → teacher layer 15
  - [1, 20]
  - [2, 25]
```
The recipe reads this into `self._layer_mapping`, where it becomes `self._layer_mapping = [[0, 15], [1, 20], [2, 25]]`

### Student → teacher projection:
```python
# kd_recipe.py
self.projections = nn.ModuleList([
    nn.Linear(2048, 3072, bias=False)   # 2048 = student dim, 3072 = teacher dim
    for _ in range(3)                   # one per layer-pair above
])
```
(Model dimensions are hard-coded in the kd_recipe - future optimization potential)

### Cosine-similarity IR loss:
```python
s_h_proj = self.projections[i](student_hiddens[i])
s_h_proj = F.layer_norm(s_h_proj, s_h_proj.shape[-1:])
t_h      = F.layer_norm(teacher_hiddens[i], t_h.shape[-1:])

ir_loss += (1 - F.cosine_similarity(s_h_proj, t_h, dim=-1)).mean()
])
```
Want MSE instead? Replace the cosine line with `ir_loss += F.mse_loss(s_h_proj, t_h)`

### Choose KD divergence:
```yaml
# distillation.yaml
kd_loss:
  _component_: kd_losses.JSDistanceLoss    # or ForwardKLLoss, ReverseKLLoss …
```
All implementations live in src/kd_losses.py

### Manual loss-scaling (CE / KD / IR):
```yaml
loss_scaling:
  enabled: true
  manual_scales:
    ce: 0.50
    kd: 0.30
    ir: 0.20
```
If you prefer the simple torchtune blend:
```yaml
loss_scaling:
  enabled: false
kd_ratio: 0.5
```

### Chunked logits (long-context friendly):
```yaml
loss:
  _component_: torchtune.losses.CEWithChunkedOutputLoss
  num_output_chunks: 8   # splits each sequence into 8 chunks
```
The recipe automatically passes lists of logits to both CE & KD losses.

---

## Quickstart

```bash
# 1. Install
git clone https://github.com/<your-handle>/llama-distillation-torchtune.git
cd llama-distillation-torchtune
pip install -r requirements.txt          

# 2. Download weights (or point YAML to HF IDs)
huggingface-cli download meta-llama/Llama-3-3b --local-dir models/3B
huggingface-cli download meta-llama/Llama-3-1b --local-dir models/1B

# 3. Train
torchrun --nproc_per_node 1 -m torchtune.run \
         --config configs/distillation.yaml
```
Optional eval:
```bash
python -m lm_eval \
       --model hf \
       --model_args pretrained=checkpoints/distilled-1B \
       --tasks truthfulqa_mc2,hellaswag,commonsense_qa
```









