# Knowledge distillation with torchtune
This repository extends the official [torchtune](https://github.com/pytorch/torchtune) framework by modifying torchtune's official `recipes/knowledge_distillation_single_device.py`, `torchtune/modules/loss/kd_losses.py`, and `torchtune/modules/loss/__init__.py` with **open-source contributions** aimed at improving flexibility, transparency, and performance. The repository contains implementations and experiments on **knowledge distillation**, where a Llama 3.2 3B teacher model is distilled into a Llama 3.2 1B student model on a single GPU.

## Key contribution
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

Core contributions are implemented in the [`src/kd_recipe.py`](src/kd_recipe.py) and  [`src/kd_losses.py`](src/kd_losses.py) files and can be configured in the [`configs/distillation.yaml`](configs/distillation.yaml) file.


---


## What’s different from the original torchtune upstream?

| Area | torchtune | **This repo** |
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

### Benchmark results — baseline vs best-distilled 1B model

| Task / Metric        | 1B **baseline**<br>(no KD, no LoRA) | **Best distilled 1B**<br>(3B-LoRA → 1B, JSD + IR) | ▲ Δ Absolute | ▲ Δ Percent |
| -------------------- | ------------------------------------ | ---------------------------------------------------- | ------------ | ----------- |
| TruthfulQA (mc2 acc) | 0.4387                               | **0.4652**                                           | +0.0265  | +6.0 %  |
| HellaSwag (acc)      | 0.4508                               | **0.4583**                                           | +0.0075  | +1.7 %  |
| CommonSenseQA (acc)  | 0.5536                               | **0.5627**                                           | +0.0091  | +1.6 %  |

**Best-distilled recipe details**

* KD loss: Jensen–Shannon divergence
* IR supervision: Cosine similarity on layer map `[[0,15], [1,20], [2,25]]` (early student layers → deeper teacher layers)
* Manual loss scales: `ce 0.5`, `kd 0.3`, `ir 0.2`

This configuration beats the plain 1B baseline on every benchmark while still fitting on a single GPU.

---

## Key implementation details
Every .yaml example below is adjustable in configs/distillation.yaml.

### Emit hidden states
```yaml
# .yaml
model:
  _component_: torchtune.models.llama3_1b
  output_hidden_states: true

teacher_model:
  _component_: torchtune.models.llama3_3b
  output_hidden_states: true
```

### Custom layer mapping:
```yaml
# .yaml
  - [0, 15]   # student layer 0 → teacher layer 15
  - [1, 20]
  - [2, 25]
```

### Student → teacher projection (auto-sized):
```python
# kd_recipe.py
s_dim = self._model.config.hidden_size       # 2048 for Llama-3 1B
t_dim = self._teacher_model.config.hidden_size # 3072 for Llama-3 3B
self.projections = nn.ModuleList(
    [nn.Linear(s_dim, t_dim, bias=False) for _ in layer_mapping]
)

```

### Cosine-similarity IR loss:
```python
# kd_recipe.py
s_h_proj = self.projections[i](student_hiddens[i])
s_h_proj = F.layer_norm(s_h_proj, s_h_proj.shape[-1:])
t_h      = F.layer_norm(teacher_hiddens[i], t_h.shape[-1:])

ir_loss += (1 - F.cosine_similarity(s_h_proj, t_h, dim=-1)).mean()
])
```
Want MSE instead? Replace the cosine line with `ir_loss += F.mse_loss(s_h_proj, t_h)`

### Choose KD divergence:
```yaml
# .yaml
kd_loss:
  _component_: kd_losses.JSDistanceWithChunkedOutputLoss    # or ForwardKLLoss, ReverseKLLoss …
```
All implementations live in src/kd_losses.py

### Manual loss-scaling (CE / KD / IR):
```yaml
# .yaml
loss_scaling:
  enabled: true
  manual_scales:
    ce: 0.50
    kd: 0.30
    ir: 0.20
```
If you prefer the simple torchtune blend:
```yaml
# .yaml
loss_scaling:
  enabled: false
kd_ratio: 0.5
```

### Chunked logits (long-context friendly):
```yaml
# .yaml
loss:
  _component_: torchtune.losses.CEWithChunkedOutputLoss
  num_output_chunks: 8   # splits each sequence into 8 chunks
```
The recipe automatically passes lists of logits to both CE & KD losses.

---









