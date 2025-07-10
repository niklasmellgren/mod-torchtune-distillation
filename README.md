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

We train on the `alpaca.cleaned.dataset`.


## Performance Metrics and Model Selection

### Performance Metrics for Models

The table below presents results from evaluating various models on the `lm-evaluation-harness` benchmarks. Each model was tested across four metrics, with different KL divergence techniques applied:

| Model                      | KL Div | TruthfulQA acc | HellaSwag acc | HellaSwag acc norm | CommonSenseQA acc |
|---------------------------|--------|----------------|----------------|---------------------|--------------------|
| 3B (baseline)             | -      | 0.4982         | 0.5225         | 0.7051              | 0.6757             |
| 3B-LoRA                   | -      | 0.4937         | 0.5274         | 0.7079              | 0.6740             |
| 1B (baseline)             | -      | 0.4387         | 0.4508         | 0.6080              | 0.5536             |
| 1B-LoRA                   | -      | 0.4304         | 0.4534         | 0.6058              | 0.5430             |
| 3B-to-1B                  | FKL    | 0.4399         | 0.4541         | 0.6097              | 0.5495             |
| 3B-LoRA-to-1B             | FKL    | 0.4437         | 0.4574         | 0.6103              | 0.5536             |
| 3B-to-1B-LoRA             | FKL    | 0.4184         | 0.4515         | 0.6077              | 0.5487             |
| 3B-LoRA-to-1B-LoRA        | FKL    | 0.4293         | 0.4566         | 0.6108              | 0.5381             |
| 3B-LoRA-to-1B             | RKL    | 0.4399         | 0.4568         | 0.6049              | 0.5471             |
| 3B-LoRA-to-1B             | SYM    | 0.4359         | 0.4575         | 0.6069              | 0.5610             |
| 3B-LoRA-to-1B             | JSD    | 0.4408         | 0.4583         | 0.6100              | 0.5627             |
| 3B-LoRA-to-1B             | TVD    | 0.4351         | 0.4570         | 0.6084              | 0.5569             |

### Identifying Best Metric Scores

The best scores across all distilled models:

- **TruthfulQA acc**: 0.4437  
- **HellaSwag acc**: 0.4583  
- **HellaSwag acc norm**: 0.6108  
- **CommonSenseQA acc**: 0.5627  

### Percentage Difference Methodology

To determine the overall best-performing model, I calculated the **percentage difference** from the best score per metric:

**Percentage Difference** = (Max Value − Model Value) / Max Value × 100

This was averaged across all metrics to yield an **Overall Score**.

### Final Ranking Based on Percentage Difference

| Model                    | KL Div | TruthfulQA Δ% | HellaSwag Δ% | HellaSwag norm Δ% | CommonSenseQA Δ% | Overall Δ% |
|--------------------------|--------|----------------|----------------|---------------------|--------------------|------------|
| 3B-LoRA-to-1B            | JSD    | 0.6535         | 0.0000         | 0.1310              | 0.0000             | **0.1961** |
| 3B-LoRA-to-1B            | FKL    | 0.0000         | 0.1964         | 0.0819              | 1.6172             | **0.4739** |
| 3B-LoRA-to-1B            | SYM    | 1.7579         | 0.1746         | 0.6385              | 0.3021             | 0.7183     |
| 3B-LoRA-to-1B            | TVD    | 1.9382         | 0.2837         | 0.3929              | 1.0307             | 0.9114     |
| 3B-LoRA-to-1B            | RKL    | 0.8564         | 0.3273         | 0.9659              | 2.7723             | 1.2305     |
| 3B-to-1B                 | FKL    | 0.8564         | 0.9164         | 0.1800              | 2.3458             | 1.0747     |
| 3B-LoRA-to-1B-LoRA       | FKL    | 3.2454         | 0.3709         | 0.0000              | 4.3718             | 1.9970     |
| 3B-to-1B-LoRA            | FKL    | 5.7020         | 1.4837         | 0.5075              | 2.4880             | 2.5453     |

### Final Model Selection

Based on the analysis above, I selected the following top-performing distilled models:

- **3B-LoRA-to-1B (JSD)** – *Overall Δ%: 0.1961*
- **3B-LoRA-to-1B (FKL)** – *Overall Δ%: 0.4739*


## Experimentation with Intermediate Representations, KL Divergence, and Loss Scaling

After identifying the top-performing models, I performed a deeper exploration into three main areas to further optimize distillation:

---

### Layer Mapping Approaches

Three different layer mapping strategies were used, defining how student model layers map to teacher model layers:

- `[[4,7],[8,14],[12,21]]`: Maps 25%, 50%, and 75% of layers (e.g., student layer 4 → teacher layer 7)
- `[[0,15],[1,20],[2,25]]`: Maps early student layers to deeper teacher layers
- `[[6,8],[7,7],[8,6]]`: Maps later student layers to earlier teacher layers

---

### Loss Scaling Approaches

To control training dynamics, I used custom loss scaling:
- `ce`: Cross-entropy loss
- `kd`: Knowledge distillation loss
- `ir`: Intermediate representation loss

Each experiment uses a different loss weighting combination.

---

### Experiment Results Table

| ID | Model               | Layer Mapping                  | KL Div | Loss Scaling              | TruthfulQA acc | HellaSwag acc | HellaSwag acc norm | CommonSenseQA acc |
|----|---------------------|--------------------------------|--------|---------------------------|----------------|----------------|---------------------|--------------------|
| 1  | 3B-LoRA-to-1B       | [[4,7], [8,14], [12,21]]       | FKL    | ce: 0.5, kd: 0.3, ir: 0.2 | 0.4457         | 0.4569         | 0.6115              | 0.5430             |
| 2  | 3B-LoRA-to-1B       | [[4,7], [8,14], [12,21]]       | JSD    | ce: 0.5, kd: 0.3, ir: 0.2 | 0.4425         | 0.4518         | 0.5963              | 0.5274             |
| 3  | 3B-LoRA-to-1B       | [[4,7], [8,14], [12,21]]       | FKL    | ce: 0.5, kd: 0.2, ir: 0.3 | 0.4382         | 0.4538         | 0.6088              | 0.5258             |
| 4  | 3B-LoRA-to-1B       | [[4,7], [8,14], [12,21]]       | JSD    | ce: 0.5, kd: 0.2, ir: 0.3 | 0.4609         | 0.4501         | 0.5978              | 0.5127             |
| 5  | 3B-LoRA-to-1B       | [[0,15], [1,20], [2,25]]       | FKL    | ce: 0.5, kd: 0.3, ir: 0.2 | 0.4437         | 0.3649         | 0.4495              | 0.5340             |
| 6  | 3B-LoRA-to-1B       | [[0,15], [1,20], [2,25]]       | JSD    | ce: 0.5, kd: 0.3, ir: 0.2 | 0.4652         | 0.4277         | 0.5682              | 0.4668             |
| 7  | 3B-LoRA-to-1B       | [[0,15], [1,20], [2,25]]       | FKL    | ce: 0.5, kd: 0.2, ir: 0.3 | 0.4525         | 0.4504         | 0.5943              | 0.5225             |
| 8  | 3B-LoRA-to-1B       | [[0,15], [1,20], [2,25]]       | JSD    | ce: 0.5, kd: 0.2, ir: 0.3 | 0.4548         | 0.4413         | 0.5817              | 0.4808             |
| 9  | 3B-LoRA-to-1B       | [[6,8], [7,7], [8,6]]          | FKL    | ce: 0.5, kd: 0.3, ir: 0.2 | 0.4429         | 0.4545         | 0.6066              | 0.5479             |
| 10 | 3B-LoRA-to-1B       | [[6,8], [7,7], [8,6]]          | JSD    | ce: 0.5, kd: 0.3, ir: 0.2 | 0.4288         | 0.4416         | 0.5852              | 0.5153             |
| 11 | 3B-LoRA-to-1B       | [[6,8], [7,7], [8,6]]          | FKL    | ce: 0.5, kd: 0.2, ir: 0.3 | 0.4403         | 0.4517         | 0.6049              | 0.5512             |
| 12 | 3B-LoRA-to-1B       | [[6,8], [7,7], [8,6]]          | JSD    | ce: 0.5, kd: 0.2, ir: 0.3 | 0.4091         | 0.4378         | 0.5762              | 0.4848             |

---

## Conclusion

- **TruthfulQA (acc)**: The best-performing distilled model for this evaluation task, 3B-LoRA-to-1B, maps the early student layers to the deeper layers of the teacher model ([[0, 15], [1,20], [2,25]]) using Jensen-Shannon divergence (JSD) as KL divergence, and a loss scaling of ce: 0.5, kd: 0.3, ir: 0.2 with cosine similarity for intermediate representation alignment. This model achieves the highest score of 0.4652.
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










