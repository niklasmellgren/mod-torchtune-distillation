# Knowledge distillation with torchtune
This repository extends the official [torchtune](https://github.com/pytorch/torchtune) library by modifying torchtune's implementations for single device knowledge distillation with open-source contributions aimed at improving flexibility, transparency, and performance.

## Key contributions
- Flexible **layer-to-layer alignment** between teacher and student models  
  > Allows custom mappings across unequal layer depths (e.g., 16-layer student ↔ 28-layer teacher), supporting partial hidden state supervision.

- Support for multiple **divergence-based loss functions**  
  > Includes Forward KL (the only implementation in the official torchtune loss module). I extend this by including Reverse KL, Symmetric KL (λ-balanced), Jensen-Shannon Divergence (JSD), and Total Variation Distance (TVD).

- **Intermediate representation (IR) supervision** via cosine similarity  
  > Cosine loss between normalized projected hidden states enhances semantic transfer across architectures, replacing underweighted MSE.

- Modular **loss scaling** framework  
  > Enables manual or dynamic weighting of cross-entropy, KD, and IR losses for interpretable, balanced training dynamics.

- Memory-efficient training with **chunked loss computation** and structured logging  
  > Processes long sequences in slices for stability and supports fine-grained loss reporting via Weights & Biases.

Core contributions are implemented in the [`src/kd_recipe.py`](src/kd_recipe.py) and  [`src/kd_losses.py`](src/kd_losses.py) files and can be configured in the [`configs/distillation.yaml`](configs/distillation.yaml) file.

---

## Overview

We distill:
- **Teacher:** Llama 3.2 3B (baseline + LoRA fine-tuned)
- **Student:** Llama 3.2 1B (baseline + LoRA fine-tuned)

Trained on the `alpaca.cleaned.dataset` and evaluated using:
- **TruthfulQA (truthfulqa mc2)**: The best-performing distilled model for this evaluation task, 3B-LoRA-to-1B, maps the early student layers to the deeper layers of the teacher model ([[0, 15], [1,20], [2,25]]) using Jensen-Shannon divergence (JSD) as KL divergence, and a loss scaling of ce: 0.5, kd: 0.3, ir: 0.2 with cosine similarity for intermediate representation alignment. This model achieves the highest score of 0.4652.
- **HellaSwag (acc norm)**: The best performance on hellaswag acc norm (0.6115) is achieved by the distilled model 3B-LoRA-to-1B, which maps layers evenly with 25%, 50%, and 75% ([[4,7], [8,14], [12,21]]) and uses ForwardKL (FKL) as KL divergence with a loss scaling of ce: 0.5, kd: 0.3, ir: 0.2 and cosine similarity for intermediate representation alignment.
- **HellaSwag (acc) and CommonSenseQA (acc)**: The 3B-LoRA-to-1B model with kd ratio: 0.5 and JSD as KL divergence scores the highest on both hellaswag acc (0.4583) and commonsense qa acc (0.5627).

These findings successfully demonstrate that the modifications to the open-source knowledge distillation recipe and the loss module from torchtune results in boosted metrics across multiple evaluation tasks, achieving high scores across the board.

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










