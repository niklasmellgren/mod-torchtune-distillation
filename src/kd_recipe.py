# recipes/knowledge_distillation_single_device.py
# -----------------------------------------------------------------------------
# Original file: torchtune/recipes/knowledge_distillation_single_device.py
# Modifications: © 2025 <your-name>  (BSD-style license inherited from upstream)
#
# What changed?
#   • Hidden-state distillation (IR loss) with cosine similarity
#   • Layer-mapping list so any student layer can align with any teacher layer
#   • Small projection layers to bridge student/teacher hidden dims
#   • Multiple KD divergences wired via cfg  (F-KL / R-KL / Sym-KL / JS / TVD)
#   • Manual loss-scaling block (CE, KD, IR) with YAML control
#   • Chunked-logit compatibility for very long sequences
#   • Extra metric logging (class, kd, ir, tokens/sec, grad-norm)
# -----------------------------------------------------------------------------
# Usage tip: If you ever change hidden-state sizes (student/teacher projections), 
# just adjust the nn.Linear(2048, 3072, …) to match the models you are using.

from __future__ import annotations

import sys
import time
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union
from warnings import warn

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, ListConfig
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

import torchtune.modules.common_utils as common_utils
from torchtune import config, modules, training, utils
from torchtune.data import padded_collate_packed, padded_collate_sft
from torchtune.datasets import ConcatDataset
from torchtune.modules.peft import (
    get_adapter_params,
    get_adapter_state_dict,
    get_lora_module_names,
    get_merged_lora_ckpt,
    set_trainable_params,
    validate_missing_and_unexpected_for_lora,
)
from torchtune.recipe_interfaces import FTRecipeInterface
from torchtune.training import DummyProfiler, PROFILER_KEY

log = utils.get_logger("DEBUG")


class KDRecipeSingleDevice(FTRecipeInterface):
    """
    Distil a large teacher model into a smaller student on a single GPU.

    The base implementation is torchtune’s original recipe.  This fork adds:

      1. Intermediate-representation (IR) supervision
         • Cosine loss between projected student & teacher hidden states.
         • `layer_mapping` in YAML decides which layers align.

      2. Projection layers
         • Hard-coded 2048→3072 Linear layers so 1B-dim → 3B-dim.
           (Change by swapping in nn.Linear(in_dim, out_dim) as desired.)

      3. Multiple KD divergences
         • Select ForwardKL, ReverseKL, SymKL, JS, TVD via `kd_loss` cfg.

      4. Manual loss scaling
         • Switch on `loss_scaling.enabled` and set ce/kd/ir weights.

      5. Chunked logits
         • Both CE and KD losses can consume a *list* of logits produced in
           num_output_chunks slices to save memory on long contexts.

      6. Extra logging
         • CE, KD, IR, tokens/sec, grad-norm, GPU mem.

    All other torchtune conveniences (bf16, LoRA, bitsandbytes, profiler,
    activation-checkpointing, torch.compile) are preserved.
    """

    # ------------------------------------------------------------------ #
    # Init / bookkeeping
    # ------------------------------------------------------------------ #

    def __init__(self, cfg: DictConfig) -> None:
        # Device and dtype
        self._device = utils.get_device(cfg.device)
        self._dtype = training.get_dtype(cfg.dtype, device=self._device)

        # Projection modules: one per hidden-state pair we distil
        self.projections = nn.ModuleList(
            [
                nn.Linear(2048, 3072, bias=False).to(self._device, dtype=self._dtype)
                for _ in range(3)
            ]
        )

        if self._dtype == torch.float16:
            raise ValueError("fp16 is disabled in this recipe; use fp32 or bf16.")

        # Logging cadence & peak-mem toggle
        self._output_dir = cfg.output_dir
        self._log_every_n_steps = cfg.get("log_every_n_steps", 1)
        self._log_peak_memory_stats = bool(cfg.get("log_peak_memory_stats", False))
        if self._log_peak_memory_stats and self._device.type != "cuda":
            log.info("Disabling peak-memory stats (CPU device).")
            self._log_peak_memory_stats = False

        # Training-state counters
        self.seed = training.set_seed(cfg.seed)
        self.epochs_run = 0
        self.total_epochs = cfg.epochs
        self.max_steps_per_epoch = cfg.max_steps_per_epoch
        self.global_step = 0

        # Flags
        self._resume_from_checkpoint = cfg.resume_from_checkpoint
        self._save_adapter_weights_only = cfg.get("save_adapter_weights_only", False)
        self._gradient_accumulation_steps = cfg.gradient_accumulation_steps
        self._clip_grad_norm = cfg.get("clip_grad_norm")
        self._kd_ratio = cfg.get("kd_ratio", 0.5)

    # ------------------------------------------------------------------ #
    # Checkpoint helpers
    # ------------------------------------------------------------------ #

    def load_checkpoint(self, cfg_checkpointer: DictConfig) -> Dict[str, Any]:
        """
        Return dict with MODEL_KEY, and optionally ADAPTER_KEY & recipe-state.
        """
        self._checkpointer = config.instantiate(
            cfg_checkpointer, should_load_recipe_state=self._resume_from_checkpoint
        )
        ckpt = self._checkpointer.load_checkpoint()

        if self._resume_from_checkpoint:
            if training.ADAPTER_KEY not in ckpt:
                raise ValueError("No adapter weights in checkpoint.")
            self._update_recipe_state(ckpt)
        return ckpt

    def load_teacher_checkpoint(self, cfg_checkpointer: DictConfig) -> Dict[str, Any]:
        """Light wrapper to read teacher weights via another Checkpointer."""
        return config.instantiate(cfg_checkpointer).load_checkpoint()

    def _update_recipe_state(self, ckpt: Dict[str, Any]) -> None:
        """Sync epoch/seed counters with checkpoint; warn on cfg mismatch."""
        try:
            self.epochs_run = ckpt[training.EPOCHS_KEY]
            if self.seed != ckpt[training.SEED_KEY]:
                warn("Using seed from checkpoint.")
                self.seed = ckpt[training.SEED_KEY]
            if self.max_steps_per_epoch != ckpt[training.MAX_STEPS_KEY]:
                warn("max_steps_per_epoch differs; using checkpoint value.")
                self.max_steps_per_epoch = ckpt[training.MAX_STEPS_KEY]
            if self.total_epochs != ckpt[training.TOTAL_EPOCHS_KEY]:
                warn("total_epochs differs; keeping config value.")
        except KeyError as exc:
            raise KeyError("Checkpoint incomplete; cannot resume.") from exc

    # ------------------------------------------------------------------ #
    # Full setup
    # ------------------------------------------------------------------ #

    def setup(self, cfg: DictConfig) -> None:
        """
        Instantiate models, data, optimizer, schedulers, profiler, etc.
        """
        self._metric_logger = config.instantiate(cfg.metric_logger)
        self._metric_logger.log_config(cfg)

        # store yaml so _loss_step can read loss_scaling.enabled
        self._config = cfg
        self._compile = cfg.compile

        # load checkpoints
        ckpt_student = self.load_checkpoint(cfg.checkpointer)
        ckpt_teacher = self.load_teacher_checkpoint(cfg.teacher_checkpointer)

        common_utils._use_low_cpu_ram = cfg.get("low_cpu_ram", False)

        # models
        self._model = self._setup_model(
            cfg.model,
            cfg.enable_activation_checkpointing,
            cfg.compile,
            ckpt_student[training.MODEL_KEY],
            ckpt_student.get(training.ADAPTER_KEY),
        )
        self._teacher_model = self._setup_teacher_model(
            cfg.teacher_model, ckpt_teacher[training.MODEL_KEY]
        )

        # tokenizer, optimizer, losses
        self._tokenizer = config.instantiate(cfg.tokenizer)
        self._optimizer = self._setup_optimizer(cfg.optimizer, ckpt_student.get(training.OPT_KEY))
        self._loss_fn = config.instantiate(cfg.loss)
        self._kd_loss_fn = config.instantiate(cfg.kd_loss)
        if self._compile:
            self._loss_fn = training.compile_loss(self._loss_fn)
            self._kd_loss_fn = training.compile_loss(self._kd_loss_fn)

        # chunked logits require both loss_fns to agree on num_output_chunks
        if self._loss_fn.__class__.__name__ == "CEWithChunkedOutputLoss":
            self._model.set_num_output_chunks(self._loss_fn.num_output_chunks)
            self._teacher_model.set_num_output_chunks(self._loss_fn.num_output_chunks)
            assert self._loss_fn.num_output_chunks == self._kd_loss_fn.num_output_chunks

        # dataset / loader
        self._sampler, self._dataloader = self._setup_data(
            cfg.dataset, cfg.shuffle, cfg.batch_size
        )

        # derive steps/epoch after dataloader is ready
        self._steps_per_epoch = len(self._dataloader) // self._gradient_accumulation_steps
        if self.max_steps_per_epoch and self.max_steps_per_epoch < self._steps_per_epoch:
            self._steps_per_epoch = self.max_steps_per_epoch
        self.global_step = self.epochs_run * self._steps_per_epoch

        # LR scheduler & profiler
        self._lr_scheduler = self._setup_lr_scheduler(
            cfg.lr_scheduler, self.total_epochs * self._steps_per_epoch, self.global_step - 1
        )
        self._profiler = self._setup_profiler(cfg.get(PROFILER_KEY, None))

        # allocate tensor used to append ignore_index token during label shift
        self.ignore_labels_cache = torch.full(
            (cfg.batch_size, 1), self._loss_fn.ignore_index, device=self._device
        )

        # save YAML layer_mapping list (needed in _loss_step)
        self._layer_mapping: List[List[int]] = cfg.get("layer_mapping", [])

    # ------------------------------------------------------------------ #
    # Builders (profiler / model / teacher / optimizer / sched / data)
    # ------------------------------------------------------------------ #

    # ... [builder helper methods unchanged from previous answer] ...
    # (They already have concise doc-strings; left out here for brevity.)
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    # Loss computation for one micro-batch
    # ------------------------------------------------------------------ #

    def _loss_step(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return (total_loss, kd_loss, ir_loss) for a single micro-batch.

        * total_loss = blended CE+KD(+IR) according to manual scales or kd_ratio.
        * kd_loss    = divergence between student & teacher logits.
        * ir_loss    = cosine distance between mapped hidden states.

        Note: student/teacher forward may return either
              tensor          -> logits only
              [h1, …, hN, L]  -> hidden states + logits
              [h1, …, hN, [L1 … Lk]] when chunked loss is active
        """
        # alias
        tokens, labels = batch["tokens"], batch["labels"]
        mask = batch.get("mask")
        input_pos = batch.get("input_pos")

        # fwd passes
        student_out = self._model(tokens, mask=mask, input_pos=input_pos)
        with torch.no_grad():
            teacher_out = self._teacher_model(tokens, mask=mask, input_pos=input_pos)

        # shift labels one token right (causal)
        labels = torch.hstack(
            (labels[..., 1:], self.ignore_labels_cache[: labels.size(0)])
        )

        # split outputs
        student_logits, student_hid = self._split_output(student_out)
        teacher_logits, teacher_hid = self._split_output(teacher_out)

        # CE + KD
        ce_loss = self._loss_fn(student_logits, labels)
        kd_loss = self._kd_loss_fn(student_logits, teacher_logits, labels)

        # IR loss (cosine); only if hidden states were requested
        ir_loss = torch.tensor(0.0, device=self._device)
        if student_hid and teacher_hid and self._layer_mapping:
            for idx, (s_idx, t_idx) in enumerate(self._layer_mapping):
                s = student_hid[s_idx]
                t = teacher_hid[t_idx]
                s_proj = self.projections[idx](s)
                s_proj = F.layer_norm(s_proj, s_proj.shape[-1:])
                t = F.layer_norm(t, t.shape[-1:])
                ir_loss += 1.0 - F.cosine_similarity(s_proj, t, dim=-1).mean()
            ir_loss /= len(self._layer_mapping)

        # blend
        if self._config.loss_scaling.enabled:
            w = self._config.loss_scaling.manual_scales
            total = w.ce * ce_loss + w.kd * kd_loss + w.ir * ir_loss
        else:
            total = (1 - self._kd_ratio) * ce_loss + self._kd_ratio * (kd_loss + ir_loss)
        return total, kd_loss, ir_loss

    @staticmethod
    def _split_output(
        output: Union[torch.Tensor, List[torch.Tensor]]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Normalize model output into (logit_list, hidden_list).

        If model returns:
          * Tensor            -> logits = [tensor], hidden = []
          * [h1..hN, L]       -> logits = [L],       hidden = [h1..hN]
          * [h1..hN, [L1..]]  -> logits = [L1..],    hidden = [h1..hN]
        """
        if isinstance(output, list):
            hidden = output[:-1]
            logits = output[-1] if isinstance(output[-1], list) else [output[-1]]
        else:
            hidden, logits = [], [output]
        return logits, hidden

    # ------------------------------------------------------------------ #
    # Training loop
    # ------------------------------------------------------------------ #

    def train(self) -> None:
        """Runs epoch/step loops with gradient accumulation and logging."""
        if self._compile:
            log.info("torch.compile active – first forward may be slower.")

        t0 = time.perf_counter()
        run_cls = run_kd = run_ir = n_tokens = 0.0

        with self._profiler as prof:
            for epoch in range(self.epochs_run, self.total_epochs):
                self._sampler.set_epoch(epoch)
                pbar = tqdm(total=self._steps_per_epoch, leave=False)

                for step, batch in enumerate(self._dataloader):
                    if self.max_steps_per_epoch and (
                        step // self._gradient_accumulation_steps
                    ) == self.max_steps_per_epoch:
                        break

                    # push batch to device
                    batch = {k: v.to(self._device) for k, v in batch.items()}

                    # unmasked token count
                    ntok = (batch["labels"] != self._loss_fn.ignore_index).sum()
                    n_tokens += ntok

                    total_loss, kd_loss, ir_loss = self._loss_step(batch)
                    run_cls += total_loss * ntok
                    run_kd += kd_loss * ntok
                    run_ir += ir_loss * ntok

                    total_loss.backward()

                    if (step + 1) % self._gradient_accumulation_steps == 0:
                        # clip / step / lr-sched
                        training.scale_grads(self._model, 1 / n_tokens)
                        if self._clip_grad_norm:
                            grad_norm = torch.nn.utils.clip_grad_norm_(
                                self._model.parameters(), float(self._clip_grad_norm)
                            )
                        self._optimizer.step()
                        self._optimizer.zero_grad(set_to_none=True)
                        self._lr_scheduler.step()
                        self.global_step += 1

                        # metrics
                        cls_log = run_cls.item() / n_tokens
                        kd_log = run_kd.item() / n_tokens
                        ir_log = run_ir.item() / n_tokens
                        loss_log = (1 - self._kd_ratio) * cls_log + self._kd_ratio * (
                            kd_log + ir_log
                        )

                        pbar.update(1)
                        pbar.set_description(f"{epoch+1}|{self.global_step}|{loss_log:.4f}")

                        if self.global_step % self._log_every_n_steps == 0:
                            elapsed = time.perf_counter() - t0
                            log_dict = {
                                "loss": loss_log,
                                "class_loss": cls_log,
                                "kd_loss": kd_log,
                                "ir_loss": ir_log,
                                "lr": self._optimizer.param_groups[0]["lr"],
                                "tokens_per_sec": n_tokens / elapsed,
                            }
                            if self._device.type == "cuda" and self._log_peak_memory_stats:
                                log_dict.update(training.get_memory_stats(self._device))
                            if self._clip_grad_norm:
                                log_dict["grad_norm"] = grad_norm
                            self._metric_logger.log_dict(log_dict, step=self.global_step)

                        # reset running windows
                        run_cls = run_kd = run_ir = n_tokens = 0.0
                        t0 = time.perf_counter()

                    prof.step()

                self.epochs_run += 1
                self.save_checkpoint(epoch)
                pbar.close()

    # ------------------------------------------------------------------ #
    # Save checkpoint (merged + adapter + recipe state)
    # ------------------------------------------------------------------ #

    def save_checkpoint(self, epoch: int) -> None:
        """Persist model and state for resume or deployment."""
        ckpt: Dict[str, Any] = {}
        mid_epoch = epoch + 1 < self.total_epochs
        if mid_epoch:
            ckpt.update(
                {
                    training.OPT_KEY: self._optimizer.state_dict(),
                    training.SEED_KEY: self.seed,
                    training.EPOCHS_KEY: self.epochs_run,
                    training.TOTAL_EPOCHS_KEY: self.total_epochs,
                    training.MAX_STEPS_KEY: self.max_steps_per_epoch,
                }
            )

        merged_weights = get_merged_lora_ckpt(
            {k: v.cpu() for k, v in self._model.state_dict().items()},
            rank=self._lora_rank,
            alpha=self._lora_alpha,
        )
        ckpt[training.MODEL_KEY] = merged_weights
        ckpt[training.ADAPTER_KEY] = get_adapter_state_dict(self._model.state_dict())
        ckpt[training.ADAPTER_CONFIG] = {
            "r": self._lora_rank,
            "lora_alpha": self._lora_alpha,
            "target_modules": get_lora_module_names(
                self._lora_attn_modules,
                self._apply_lora_to_mlp,
                self._apply_lora_to_output,
            ),
            "peft_type": "LORA",
        }

        self._checkpointer.save_checkpoint(
            ckpt,
            epoch=epoch,
            intermediate_checkpoint=mid_epoch,
            adapter_only=self._save_adapter_weights_only,
        )

    # ------------------------------------------------------------------ #

    def cleanup(self) -> None:
        """Flush metric logger / profiler handles."""
        self._metric_logger.close()


@config.parse
def recipe_main(cfg: DictConfig) -> None:
    """Entrypoint so `torchrun -m torchtune.run --config ...` works."""
    config.log_config("KDRecipeSingleDevice", cfg)
    rec = KDRecipeSingleDevice(cfg)
    rec.setup(cfg)
    rec.train()
    rec.cleanup()


if __name__ == "__main__":
    sys.exit(recipe_main())
