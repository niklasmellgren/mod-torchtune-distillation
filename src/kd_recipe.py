%%writefile recipes/knowledge_distillation_single_device.py
# -----------------------------------------------------------------------------
# Torchtune – Knowledge-Distillation (single-GPU) - Modified recipe
# Original upstream file: torchtune/recipes/knowledge_distillation_single_device.py
# Modifications by Niklas Mellgren
#
# What changed?
#   • IR-loss (cosine) on hidden-state pairs                cfg.layer_mapping
#   • Projection layers auto-sized  student→teacher dim     (built at runtime)
#   • Plug-in KD losses (F-KL / R-KL / Sym-KL / JS / TVD)   cfg.kd_loss
#   • Optional manual CE/KD/IR scaling                      cfg.loss_scaling
#   • Chunked-logit compatibility for long contexts
#   • Extra metric logging (CE/KD/IR, tok/s, grad-norm, mem)
# -----------------------------------------------------------------------------
# Usage tip:
#   Just set `layer_mapping` in your YAML and the code builds the right number
#   of projection layers with   in_dim = student.config.hidden_size
#                               out_dim = teacher.config.hidden_size
# -----------------------------------------------------------------------------

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


# --------------------------------------------------------------------------- #
#  Recipe                                                                     #
# --------------------------------------------------------------------------- #
class KDRecipeSingleDevice(FTRecipeInterface):
    """
    Distil a large teacher LLM into a smaller student on **one GPU**.

    Add-ons compared with torchtune’s stock recipe
    ------------------------------------------------
    • Cosine IR-loss on arbitrary hidden-state pairs (cfg.layer_mapping)
    • Projection layers sized dynamically to bridge hidden-dim mismatch
    • Several KD-divergence flavours pluggable via YAML
    • Optional manual CE/KD/IR scaling (cfg.loss_scaling.enabled)
    • Works with chunked-logit CE/KD losses for long contexts
    • Extra runtime metrics
    """

    # ------------------------------------------------------------------ #
    # Construction                                                       #
    # ------------------------------------------------------------------ #
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        # Device & precision --------------------------------------------------
        self._device = utils.get_device(cfg.device)
        self._dtype = training.get_dtype(cfg.dtype, device=self._device)
        if self._dtype == torch.float16:
            raise ValueError("fp16 unsupported in this recipe; use fp32 or bf16.")

        # Log & misc ----------------------------------------------------------
        self._output_dir = cfg.output_dir
        self._log_every_n_steps = cfg.get("log_every_n_steps", 1)
        self._log_peak_memory_stats = bool(cfg.get("log_peak_memory_stats", False))
        if self._log_peak_memory_stats and self._device.type != "cuda":
            self._log_peak_memory_stats = False

        # Counters & flags ----------------------------------------------------
        self.seed = training.set_seed(cfg.seed)
        self.epochs_run = 0
        self.total_epochs = cfg.epochs
        self.max_steps_per_epoch = cfg.max_steps_per_epoch
        self.global_step = 0

        self._resume_from_checkpoint = cfg.resume_from_checkpoint
        self._save_adapter_weights_only = cfg.get("save_adapter_weights_only", False)
        self._gradient_accumulation_steps = cfg.gradient_accumulation_steps
        self._clip_grad_norm = cfg.get("clip_grad_norm")
        self._kd_ratio = cfg.get("kd_ratio", 0.5)

        # Filled later in setup() --------------------------------------------
        self.projections: Optional[nn.ModuleList] = None
        self._layer_mapping: List[List[int]] = []

    # ------------------------------------------------------------------ #
    # Checkpoint helpers                                                 #
    # ------------------------------------------------------------------ #
    def load_checkpoint(self, cp_cfg: DictConfig) -> Dict[str, Any]:
        self._checkpointer = config.instantiate(
            cp_cfg, should_load_recipe_state=self._resume_from_checkpoint
        )
        ckpt = self._checkpointer.load_checkpoint()
        if self._resume_from_checkpoint:
            if training.ADAPTER_KEY not in ckpt:
                raise ValueError("Adapter weights missing in resume checkpoint.")
            self._update_recipe_state(ckpt)
        return ckpt

    @staticmethod
    def load_teacher_checkpoint(cp_cfg: DictConfig) -> Dict[str, Any]:
        return config.instantiate(cp_cfg).load_checkpoint()

    def _update_recipe_state(self, ckpt: Dict[str, Any]) -> None:
        try:
            self.epochs_run = ckpt[training.EPOCHS_KEY]
            if self.seed != ckpt[training.SEED_KEY]:
                warn("Seed mismatch – using checkpoint seed.")
                self.seed = ckpt[training.SEED_KEY]
            if self.max_steps_per_epoch != ckpt[training.MAX_STEPS_KEY]:
                warn("max_steps_per_epoch mismatch – using checkpoint value.")
                self.max_steps_per_epoch = ckpt[training.MAX_STEPS_KEY]
            if self.total_epochs != ckpt[training.TOTAL_EPOCHS_KEY]:
                warn("total_epochs differs; keeping cfg value.")
        except KeyError as exc:
            raise KeyError("Checkpoint incomplete; cannot resume.") from exc

    # ------------------------------------------------------------------ #
    # Setup                                                              #
    # ------------------------------------------------------------------ #
    def setup(self, cfg: DictConfig) -> None:  # noqa: C901
        """Instantiate everything: models → data → optimiser → profiler."""
        self._metric_logger = config.instantiate(cfg.metric_logger)
        self._metric_logger.log_config(cfg)

        self._config = cfg           # for loss-scaling check
        self._compile = cfg.compile

        ckpt_student = self.load_checkpoint(cfg.checkpointer)
        ckpt_teacher = self.load_teacher_checkpoint(cfg.teacher_checkpointer)

        common_utils._use_low_cpu_ram = cfg.get("low_cpu_ram", False)

        # -------------- Models ---------------------------------------------
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

        # Build projection layers dynamically
        self._layer_mapping = [list(p) for p in cfg.get("layer_mapping", [])]
        s_dim = self._model.config.hidden_size
        t_dim = self._teacher_model.config.hidden_size
        self.projections = nn.ModuleList(
            [
                nn.Linear(s_dim, t_dim, bias=False).to(self._device, dtype=self._dtype)
                for _ in range(len(self._layer_mapping))
            ]
        )
        log.info(f"Created {len(self.projections)} projection layers  {s_dim}->{t_dim}")

        # -------------- Tokeniser / losses / optimiser ----------------------
        self._tokenizer = config.instantiate(cfg.tokenizer)
        self._optimizer = self._setup_optimizer(cfg.optimizer, ckpt_student.get(training.OPT_KEY))
        self._loss_fn = config.instantiate(cfg.loss)
        self._kd_loss_fn = config.instantiate(cfg.kd_loss)
        if self._compile:
            self._loss_fn = training.compile_loss(self._loss_fn)
            self._kd_loss_fn = training.compile_loss(self._kd_loss_fn)

        # When chunked-logit loss is used, both CE & KD must agree on chunks
        if self._loss_fn.__class__.__name__ == "CEWithChunkedOutputLoss":
            self._model.set_num_output_chunks(self._loss_fn.num_output_chunks)
            self._teacher_model.set_num_output_chunks(self._loss_fn.num_output_chunks)
            assert self._loss_fn.num_output_chunks == self._kd_loss_fn.num_output_chunks

        # -------------- Data ------------------------------------------------
        self._sampler, self._dataloader = self._setup_data(
            cfg.dataset, cfg.shuffle, cfg.batch_size
        )
        self._steps_per_epoch = len(self._dataloader) // self._gradient_accumulation_steps
        if self.max_steps_per_epoch and self.max_steps_per_epoch < self._steps_per_epoch:
            self._steps_per_epoch = self.max_steps_per_epoch
        self.global_step = self.epochs_run * self._steps_per_epoch

        # -------------- LR-sched & profiler --------------------------------
        self._lr_scheduler = self._setup_lr_scheduler(
            cfg.lr_scheduler,
            num_training_steps=self.total_epochs * self._steps_per_epoch,
            last_epoch=self.global_step - 1,
        )
        self._profiler = self._setup_profiler(cfg.get(PROFILER_KEY, None))

        # Pre-allocate label-shift ignore token
        self.ignore_labels_cache = torch.full(
            (cfg.batch_size, 1), self._loss_fn.ignore_index, device=self._device
        )

    # ------------------------------------------------------------------ #
    # Helper builders (profiler / model / teacher / optimiser / data …)  #
    # ------------------------------------------------------------------ #
    def _setup_profiler(
        self, cfg_profiler: Optional[DictConfig]
    ) -> Union[torch.profiler.profile, DummyProfiler]:
        """Return real or dummy profiler context."""
        if cfg_profiler is None:
            cfg_profiler = DictConfig({"enabled": False})
        if cfg_profiler.get("_component_", None) is None:
            cfg_profiler["_component_"] = "torchtune.training.setup_torch_profiler"
        profiler, pcfg = config.instantiate(cfg_profiler)
        self.profiler_profile_memory = pcfg.get("profile_memory", False)
        if pcfg["enabled"]:
            self.profiler_wait_steps = pcfg["wait_steps"]
            self.profiler_warmup_steps = pcfg["warmup_steps"]
            self.profiler_active_steps = pcfg["active_steps"]
        return profiler

    # ----------------------- model (student) -------------------------------
    def _setup_model(
        self,
        cfg_model: DictConfig,
        activation_ckpt: bool,
        compile_model: bool,
        base_state: Dict[str, Any],
        lora_state: Optional[Dict[str, Any]],
    ) -> nn.Module:
        """Instantiate student model + (optionally) merge LoRA weights."""
        output_hidden = cfg_model.pop("output_hidden_states", None)
        with training.set_default_dtype(self._dtype), self._device:
            model = config.instantiate(cfg_model)
        if output_hidden is not None:
            model.output_hidden_states = output_hidden

        # LoRA plumbing (unchanged from upstream)
        self._lora_rank = cfg_model.lora_rank
        self._lora_alpha = cfg_model.lora_alpha
        self._lora_attn_modules = list(cfg_model.lora_attn_modules)
        self._apply_lora_to_mlp = cfg_model.apply_lora_to_mlp
        self._apply_lora_to_output = getattr(cfg_model, "apply_lora_to_output", False)
        self.adapter_params = get_adapter_params(model)
        self._is_dora = any("magnitude" in k for k in self.adapter_params.keys())
        set_trainable_params(model, self.adapter_params)

        if compile_model:
            training.compile_model(model)
        if activation_ckpt:
            training.set_activation_checkpointing(
                model, auto_wrap_policy={modules.TransformerSelfAttentionLayer}
            )

        base_missing, base_unexp = model.load_state_dict(base_state, strict=False)
        if self._is_dora:
            for m in model.modules():
                if hasattr(m, "initialize_dora_magnitude"):
                    m.initialize_dora_magnitude()
        if lora_state:
            lora_missing, lora_unexp = model.load_state_dict(lora_state, strict=False)
        else:
            lora_missing = lora_unexp = None
        validate_missing_and_unexpected_for_lora(
            self._lora_attn_modules,
            self._apply_lora_to_mlp,
            self._apply_lora_to_output,
            base_missing,
            base_unexp,
            lora_missing,
            lora_unexp,
        )
        training.validate_expected_param_dtype(self.adapter_params.items(), self._dtype)
        return model

    # ----------------------- model (teacher) -------------------------------
    def _setup_teacher_model(self, cfg: DictConfig, state: Dict[str, Any]) -> nn.Module:
        """Instantiate frozen teacher model."""
        output_hidden = cfg.pop("output_hidden_states", None)
        with training.set_default_dtype(self._dtype), self._device:
            model = config.instantiate(cfg)
        if output_hidden is not None:
            model.output_hidden_states = output_hidden
        model.load_state_dict(state)
        model.eval()
        training.validate_expected_param_dtype(model.named_parameters(), self._dtype)
        return model

    # ----------------------- optimiser / sched -----------------------------
    def _setup_optimizer(
        self, cfg_opt: DictConfig, opt_state: Optional[Dict[str, Any]]
    ) -> Optimizer:
        optim = config.instantiate(cfg_opt, self._model.parameters())
        if opt_state:
            optim.load_state_dict(opt_state)
        return optim

    def _setup_lr_scheduler(
        self, cfg_sched: DictConfig, num_training_steps: int, last_epoch: int
    ):
        return config.instantiate(
            cfg_sched,
            self._optimizer,
            num_training_steps=num_training_steps,
            last_epoch=last_epoch,
        )

    # ----------------------- data ------------------------------------------
    def _setup_data(
        self, cfg_ds: DictConfig, shuffle: bool, batch_size: int
    ) -> Tuple[DistributedSampler, DataLoader]:
        if isinstance(cfg_ds, ListConfig):
            datasets = [config.instantiate(c, self._tokenizer) for c in cfg_ds]
            ds = ConcatDataset(datasets)
            packed = False
        else:
            ds = config.instantiate(cfg_ds, self._tokenizer)
            packed = cfg_ds.get("packed", False)

        sampler = DistributedSampler(ds, num_replicas=1, rank=0, shuffle=shuffle, seed=0)
        loader = DataLoader(
            ds,
            sampler=sampler,
            batch_size=batch_size,
            drop_last=True,
            collate_fn=(
                partial(
                    padded_collate_sft,
                    padding_idx=self._tokenizer.pad_id,
                    ignore_idx=self._loss_fn.ignore_index,
                )
                if not packed
                else padded_collate_packed
            ),
        )
        return sampler, loader

    # ----------------------- checkpoint save -------------------------------
    def save_checkpoint(self, epoch: int) -> None:
        """Save merged weights, LoRA adapters & (optionally) optimiser state."""
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

        merged = get_merged_lora_ckpt(
            {k: v.cpu() for k, v in self._model.state_dict().items()},
            rank=self._lora_rank,
            alpha=self._lora_alpha,
        )
        ckpt[training.MODEL_KEY] = merged
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
            ckpt, epoch=epoch, intermediate_checkpoint=mid_epoch, adapter_only=self._save_adapter_weights_only
        )

    # ------------------------------------------------------------------ #
    # Loss step                                                          #
    # ------------------------------------------------------------------ #
    def _loss_step(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return total, kd, ir loss for **one micro-batch**."""
        tokens, labels = batch["tokens"], batch["labels"]
        mask, input_pos = batch.get("mask"), batch.get("input_pos")

        s_out = self._model(tokens, mask=mask, input_pos=input_pos)
        with torch.no_grad():
            t_out = self._teacher_model(tokens, mask=mask, input_pos=input_pos)

        labels = torch.hstack((labels[..., 1:], self.ignore_labels_cache[: labels.size(0)]))

        s_logits, s_hid = self._split_output(s_out)
        t_logits, t_hid = self._split_output(t_out)

        ce_loss = self._loss_fn(s_logits, labels)
        kd_loss = self._kd_loss_fn(s_logits, t_logits, labels)

        ir_loss = torch.zeros((), device=self._device)
        if s_hid and t_hid and self._layer_mapping:
            for idx, (s_idx, t_idx) in enumerate(self._layer_mapping):
                s = F.layer_norm(self.projections[idx](s_hid[s_idx]), (1,))
                t = F.layer_norm(t_hid[t_idx], (1,))
                ir_loss += 1.0 - F.cosine_similarity(s, t, dim=-1).mean()
            ir_loss /= len(self._layer_mapping)

        if self._config.loss_scaling.enabled:
            w = self._config.loss_scaling.manual_scales
            total = w.ce * ce_loss + w.kd * kd_loss + w.ir * ir_loss
        else:
            total = (1 - self._kd_ratio) * ce_loss + self._kd_ratio * (kd_loss + ir_loss)
        return total, kd_loss, ir_loss

    @staticmethod
    def _split_output(
        out: Union[torch.Tensor, List[torch.Tensor]]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        if isinstance(out, list):
            hid = out[:-1]
            logits = out[-1] if isinstance(out[-1], list) else [out[-1]]
        else:
            hid, logits = [], [out]
        return logits, hid

    # ------------------------------------------------------------------ #
    # Training loop                                                      #
    # ------------------------------------------------------------------ #
    def train(self) -> None:  # noqa: C901
        """Epoch / step loop with grad-accumulation & rich logging."""
        if self._compile:
            log.info("torch.compile enabled – first fwd pass will be slower.")

        t0 = time.perf_counter()
        run_ce = run_kd = run_ir = n_tok = 0.0

        with self._profiler as prof:
            for epoch in range(self.epochs_run, self.total_epochs):
                self._sampler.set_epoch(epoch)
                pbar = tqdm(total=self._steps_per_epoch, leave=False)

                for step, batch in enumerate(self._dataloader):
                    if self.max_steps_per_epoch and (
                        step // self._gradient_accumulation_steps
                    ) == self.max_steps_per_epoch:
                        break

                    batch = {k: v.to(self._device) for k, v in batch.items()}
                    tok_here = (batch["labels"] != self._loss_fn.ignore_index).sum()
                    n_tok += tok_here

                    total, kd_loss, ir_loss = self._loss_step(batch)
                    run_ce += total * tok_here
                    run_kd += kd_loss * tok_here
                    run_ir += ir_loss * tok_here

                    total.backward()
                    if (step + 1) % self._gradient_accumulation_steps == 0:
                        training.scale_grads(self._model, 1 / n_tok)
                        if self._clip_grad_norm:
                            grad_norm = torch.nn.utils.clip_grad_norm_(self._model.parameters(), float(self._clip_grad_norm))
                        self._optimizer.step()
                        self._optimizer.zero_grad(set_to_none=True)
                        self._lr_scheduler.step()
                        self.global_step += 1

                        ce_log = run_ce.item() / n_tok
                        kd_log = run_kd.item() / n_tok
                        ir_log = run_ir.item() / n_tok
                        loss_log = (1 - self._kd_ratio) * ce_log + self._kd_ratio * (kd_log + ir_log)

                        pbar.update(1)
                        pbar.set_description(f"{epoch+1}|{self.global_step}|{loss_log:.4f}")

                        if self.global_step % self._log_every_n_steps == 0:
                            elapsed = time.perf_counter() - t0
                            log_dict = {
                                "loss": loss_log,
                                "class_loss": ce_log,
                                "kd_loss": kd_log,
                                "ir_loss": ir_log,
                                "lr": self._optimizer.param_groups[0]["lr"],
                                "tokens_per_sec": n_tok / elapsed,
                            }
                            if self._device.type == "cuda" and self._log_peak_memory_stats:
                                log_dict.update(training.get_memory_stats(self._device))
                            if self._clip_grad_norm:
                                log_dict["grad_norm"] = grad_norm
                            self._metric_logger.log_dict(log_dict, step=self.global_step)

                        run_ce = run_kd = run_ir = n_tok = 0.0
                        t0 = time.perf_counter()

                    prof.step()

                self.epochs_run += 1
                self.save_checkpoint(epoch)
                pbar.close()

    # ------------------------------------------------------------------ #
    # House-keeping                                                      #
    # ------------------------------------------------------------------ #
    def cleanup(self) -> None:
        self._metric_logger.close()


# --------------------------------------------------------------------------- #
# Entrypoint                                                                  #
# --------------------------------------------------------------------------- #
@config.parse
def recipe_main(cfg: DictConfig) -> None:
    """Entry-point so `torchrun -m torchtune.run --config …` works."""
    config.log_config("KDRecipeSingleDevice", cfg)
    rec = KDRecipeSingleDevice(cfg)
    rec.setup(cfg)
    rec.train()
    rec.cleanup()


if __name__ == "__main__":
    sys.exit(recipe_main())
