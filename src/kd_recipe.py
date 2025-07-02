#  torchtune - knowledge-distillation (single-GPU) - extended recipe
#
#  Original upstream: torchtune/recipes/knowledge_distillation_single_device.py
#  Modifications by Niklas Mellgren
# -----------------------------------------------------------------------------
#  CHANGE-LOG (vs upstream)
#  • Hidden-state distillation (IR loss) – cosine distance on arbitrary layer
#    pairs defined in YAML `layer_mapping`.
#  • Projection layer bank auto-sized from student/teacher hidden_dims.
#  • Multiple divergence choices for KD: F-KL / R-KL / Sym-KL / JS / TVD.
#  • Optional manual CE/KD/IR weighting via `loss_scaling` block in YAML.
#  • Compatible with chunked-logit CE/KD losses (long contexts, low VRAM).
#  • Extra runtime metrics (CE/KD/IR losses, tokens/sec, grad-norm, mem).
# -----------------------------------------------------------------------------
#  QUICK USAGE
#      layer_mapping:
#        - [4, 7]          # student layer-4 → teacher layer-7
#        - [8, 14]
#        - [12, 21]
#
#      loss_scaling:
#        enabled: true
#        manual_scales: {ce: 0.5, kd: 0.3, ir: 0.2}
#
#  After editing layer_mapping you do not have to touch code: a projection
#  layer of shape (student_hidden, teacher_hidden) is built automatically for
#  each mapping entry.
# -----------------------------------------------------------------------------
"""
torchtune single-GPU knowledge-distillation recipe with hidden-state­-level loss.

High-level flow
---------------
1.  Parse YAML → build student & frozen teacher models.
2.  Create `len(layer_mapping)` projection layers (student→teacher hidden dims).
3.  For each micro-batch:
     • CE loss = teacher-guided cross-entropy (chunked compatible)
     • KD loss = logit divergence   (F-KL / R-KL / Sym-KL / JS / TVD)
     • IR loss = cosine distance between mapped hidden-state pairs
   Optionally scale CE/KD/IR with manual weights.
4.  Gradient-accumulate, clip, optimiser-step, LR-scheduler-step.
5.  Log rich metrics + save checkpoints (merged weights & LoRA adapters).

Why hidden-state distillation?
------------------------------
Matching only logits often leaves internal representations free to diverge,
which hurts generalisation.  Adding a (cheap) cosine term nudges the student to
*think* like the teacher, not just answer like it.

Key YAML knobs (summary)
------------------------
`layer_mapping`        which layers align → drives number of projections  
`kd_loss`              choose divergence implementation class  
`loss_scaling.enabled` True → use manual ce/kd/ir weights  
`activation_checkpointing` trade memory ↔ speed  
`dtype`                bf16 / fp32 (fp16 not supported)  
"""

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

# Create a module-wide logger (honours TorchTune’s log-level flags)
log = utils.get_logger("DEBUG")


# =============================================================================
#  Recipe class
# =============================================================================
class KDRecipeSingleDevice(FTRecipeInterface):
    """
    Distil a **teacher** LLM into a **student** LLM on *one* CUDA device.

    Additions beyond upstream recipe
    --------------------------------
    1. Hidden-state (“IR”) cosine loss on arbitrary layer pairs.
    2. Auto-built projection layers bridging hidden-dim mismatch.
    3. Pluggable KD divergences selected in YAML.
    4. Manual CE/KD/IR scaling (optional).
    5. Support for chunked logits when sequence ≫ VRAM.
    6. Extra logging & cleaner progress-bar.

    bf16, LoRA, bitsandbytes, activation checkpointing, torch.compile, W&B, etc. 
    remains fully available.
    """

    # --------------------------------------------------------------------- #
    # Init – light, only cheap stuff (heavy lifting in `setup`)
    # --------------------------------------------------------------------- #
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        # Device & precision --------------------------------------------------
        self._device = utils.get_device(cfg.device)
        self._dtype = training.get_dtype(cfg.dtype, device=self._device)
        if self._dtype == torch.float16:
            raise ValueError("fp16 unsupported – choose fp32 or bf16.")

        # Misc bookkeeping ----------------------------------------------------
        self.seed = training.set_seed(cfg.seed)
        self.total_epochs = cfg.epochs
        self.max_steps_per_epoch = cfg.max_steps_per_epoch
        self._gradient_accumulation_steps = cfg.gradient_accumulation_steps
        self._resume_from_checkpoint = cfg.resume_from_checkpoint
        self._save_adapter_weights_only = cfg.get("save_adapter_weights_only", False)
        self._clip_grad_norm = cfg.get("clip_grad_norm")
        self._kd_ratio = cfg.get("kd_ratio", 0.5)

        # Logging cadence -----------------------------------------------------
        self._output_dir = cfg.output_dir
        self._log_every_n_steps = cfg.get("log_every_n_steps", 1)
        self._log_peak_memory_stats = bool(cfg.get("log_peak_memory_stats", False))
        if self._log_peak_memory_stats and self._device.type != "cuda":
            self._log_peak_memory_stats = False

        # Will be filled in setup() ------------------------------------------
        self.projections: Optional[nn.ModuleList] = None
        self._layer_mapping: List[List[int]] = []
        self.epochs_run = 0
        self.global_step = 0

    # --------------------------------------------------------------------- #
    # Checkpoint helpers
    # --------------------------------------------------------------------- #
    def load_checkpoint(self, cp_cfg: DictConfig) -> Dict[str, Any]:
        """Load *student* checkpoint and optionally restore recipe state."""
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
        """Plain load for frozen teacher weights."""
        return config.instantiate(cp_cfg).load_checkpoint()

    # ⬇ helper merges YAML/ckpt counters & warns on mismatch
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
                warn("total_epochs differs; keeping YAML setting.")
        except KeyError as exc:
            raise KeyError("Checkpoint incomplete; cannot resume.") from exc

    # --------------------------------------------------------------------- #
    # SETUP – heavy lifting (models, data, optimiser…)
    # --------------------------------------------------------------------- #
    def setup(self, cfg: DictConfig) -> None:  # noqa: C901 – long but clear
        """
        Instantiate **everything** – models → projections → data → optimiser →
        LR scheduler → profiler.

        Called once before training / evaluation.
        """
        # ---------------------- Logger / W&B / TB ---------------------------
        self._metric_logger = config.instantiate(cfg.metric_logger)
        self._metric_logger.log_config(cfg)

        self._config = cfg                # keep YAML for loss-scaling check
        self._compile = cfg.compile       # torch.compile flag

        # ---------------------- Load checkpoints ----------------------------
        ckpt_student = self.load_checkpoint(cfg.checkpointer)
        ckpt_teacher = self.load_teacher_checkpoint(cfg.teacher_checkpointer)

        common_utils._use_low_cpu_ram = cfg.get("low_cpu_ram", False)

        # ---------------------- Build student + teacher ---------------------
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

        # ---------------------- Projections (hidden-dim bridge) ------------
        self._layer_mapping = [list(pair) for pair in cfg.layer_mapping]
        s_dim = self._model.config.hidden_size
        t_dim = self._teacher_model.config.hidden_size
        self.projections = nn.ModuleList(
            [
                nn.Linear(s_dim, t_dim, bias=False).to(self._device, dtype=self._dtype)
                for _ in range(len(self._layer_mapping))
            ]
        )
        log.info(f"Built {len(self.projections)} projection layers: {s_dim} → {t_dim}")

        # ---------------------- Tokeniser / losses / optimiser -------------
        self._tokenizer = config.instantiate(cfg.tokenizer)
        self._optimizer = self._setup_optimizer(cfg.optimizer, ckpt_student.get(training.OPT_KEY))
        self._loss_fn = config.instantiate(cfg.loss)
        self._kd_loss_fn = config.instantiate(cfg.kd_loss)
        if self._compile:
            self._loss_fn = training.compile_loss(self._loss_fn)
            self._kd_loss_fn = training.compile_loss(self._kd_loss_fn)

        # Synchronise chunk count when using chunked CE/KD losses
        if self._loss_fn.__class__.__name__ == "CEWithChunkedOutputLoss":
            self._model.set_num_output_chunks(self._loss_fn.num_output_chunks)
            self._teacher_model.set_num_output_chunks(self._loss_fn.num_output_chunks)
            assert self._loss_fn.num_output_chunks == self._kd_loss_fn.num_output_chunks

        # ---------------------- Data loader ---------------------------------
        self._sampler, self._dataloader = self._setup_data(
            cfg.dataset, cfg.shuffle, cfg.batch_size
        )
        self._steps_per_epoch = len(self._dataloader) // self._gradient_accumulation_steps
        if self.max_steps_per_epoch and self.max_steps_per_epoch < self._steps_per_epoch:
            self._steps_per_epoch = self.max_steps_per_epoch
        self.global_step = self.epochs_run * self._steps_per_epoch

        # ---------------------- LR scheduler & profiler ---------------------
        self._lr_scheduler = self._setup_lr_scheduler(
            cfg.lr_scheduler,
            num_training_steps=self.total_epochs * self._steps_per_epoch,
            last_epoch=self.global_step - 1,
        )
        self._profiler = self._setup_profiler(cfg.get(PROFILER_KEY, None))

        # Pre-allocate one column of ignore_index for causal-label shift
        self.ignore_labels_cache = torch.full(
            (cfg.batch_size, 1), self._loss_fn.ignore_index, device=self._device
        )

    # --------------------------------------------------------------------- #
    # Builders (profiler / model / teacher / optimiser / sched / data)
    # --------------------------------------------------------------------- #
    # … see code below – all functions include inline comments for clarity.

    # ---------------- profiler -------------------------------------------
    def _setup_profiler(
        self, cfg_profiler: Optional[DictConfig]
    ) -> Union[torch.profiler.profile, DummyProfiler]:
        """
        Build a `torch.profiler.profile` or a dummy no-op context depending on
        YAML.  You rarely need to edit this.
        """
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

    # ---------------- student model ---------------------------------------
    def _setup_model(
        self,
        cfg_model: DictConfig,
        activation_ckpt: bool,
        compile_model: bool,
        base_state: Dict[str, Any],
        lora_state: Optional[Dict[str, Any]],
    ) -> nn.Module:
        """
        Instantiate **student** model, attach LoRA adapters, compile & checkpoint.
        """
        hidden_flag = cfg_model.pop("output_hidden_states", None)
        with training.set_default_dtype(self._dtype), self._device:
            model = config.instantiate(cfg_model)
        if hidden_flag is not None:
            model.output_hidden_states = hidden_flag

        # LoRA plumbing (mostly upstream code) ------------------------------
        self._lora_rank = cfg_model.lora_rank
        self._lora_alpha = cfg_model.lora_alpha
        self._lora_attn_modules = list(cfg_model.lora_attn_modules)
        self._apply_lora_to_mlp = cfg_model.apply_lora_to_mlp
        self._apply_lora_to_output = getattr(cfg_model, "apply_lora_to_output", False)

        self.adapter_params = get_adapter_params(model)
        self._is_dora = any("magnitude" in p for p in self.adapter_params.keys())
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

    # ---------------- teacher model ---------------------------------------
    def _setup_teacher_model(self, cfg: DictConfig, state: Dict[str, Any]) -> nn.Module:
        """Instantiate **frozen** teacher model."""
        hidden_flag = cfg.pop("output_hidden_states", None)
        with training.set_default_dtype(self._dtype), self._device:
            model = config.instantiate(cfg)
        if hidden_flag is not None:
            model.output_hidden_states = hidden_flag
        model.load_state_dict(state)
        model.eval()
        training.validate_expected_param_dtype(model.named_parameters(), self._dtype)
        return model

    # ---------------- optimiser & LR-sched -------------------------------
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

    # ---------------- data loader ----------------------------------------
    def _setup_data(
        self, cfg_ds: DictConfig, shuffle: bool, batch_size: int
    ) -> Tuple[DistributedSampler, DataLoader]:
        """
        Build DataLoader (+DistributedSampler stub) for map-style datasets.

        *Packed* datasets already contain pre-packed token blocks and therefore
        use a different collate_fn.
        """
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
            drop_last=True,  # keeps shape consistent when flex-attention active
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

    # ------------------------------------------------------------------ #
    # Checkpoint save (merged weights + adapters + recipe state)         #
    # ------------------------------------------------------------------ #
    def save_checkpoint(self, epoch: int) -> None:
        """
        Persist model & training state.  Two flavours:

        • intermediate (epoch < last): merged + adapters + optimiser + recipe
        • final          (epoch == last): merged + adapters   (optim not needed)
        """
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
            ckpt,
            epoch=epoch,
            intermediate_checkpoint=mid_epoch,
            adapter_only=self._save_adapter_weights_only,
        )

    # ------------------------------------------------------------------ #
    # Single micro-batch loss                                            #
    # ------------------------------------------------------------------ #
    def _loss_step(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute **total**, **kd**, **ir** loss for a micro-batch.

        • CE loss is wrapped inside `self._loss_fn`  (may be chunked)
        • KD divergence wrapped inside `self._kd_loss_fn` (same chunking)
        • IR loss: mean cosine distance across mapped hidden pairs
        """
        tokens, labels = batch["tokens"], batch["labels"]
        mask, input_pos = batch.get("mask"), batch.get("input_pos")

        # -------- forward passes (student + frozen teacher) -----------------
        s_out = self._model(tokens, mask=mask, input_pos=input_pos)
        with torch.no_grad():
            t_out = self._teacher_model(tokens, mask=mask, input_pos=input_pos)

        # -------- label shift (causal LM) -----------------------------------
        labels = torch.hstack((labels[..., 1:], self.ignore_labels_cache[: labels.size(0)]))

        # -------- split outputs into logits [+hiddens] ----------------------
        s_logits, s_hid = self._split_output(s_out)
        t_logits, t_hid = self._split_output(t_out)

        # -------- CE + KD loss ---------------------------------------------
        ce_loss = self._loss_fn(s_logits, labels)
        kd_loss = self._kd_loss_fn(s_logits, t_logits, labels)

        # -------- IR cosine loss -------------------------------------------
        ir_loss = torch.zeros((), device=self._device)
        if s_hid and t_hid and self._layer_mapping:
            for idx, (s_idx, t_idx) in enumerate(self._layer_mapping):
                s_vec = F.layer_norm(self.projections[idx](s_hid[s_idx]), (1,))
                t_vec = F.layer_norm(t_hid[t_idx], (1,))
                ir_loss += 1.0 - F.cosine_similarity(s_vec, t_vec, dim=-1).mean()
            ir_loss /= len(self._layer_mapping)

        # -------- blend total ----------------------------------------------
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
        Normalise model output into *(logits_list, hidden_list)*.

        Handles three forms:
            • Tensor                          → logits=[T]      hidden=[]
            • [h1 … hN, logits]               → logits=[logits] hidden=[h1…hN]
            • [h1 … hN, [logits_0 … logits_M]]→ logits=list(..) hidden=[h1…hN]
        """
        if isinstance(output, list):
            hidden = output[:-1]
            logits = output[-1] if isinstance(output[-1], list) else [output[-1]]
        else:
            hidden, logits = [], [output]
        return logits, hidden

    # ------------------------------------------------------------------ #
    # Training loop                                                      #
    # ------------------------------------------------------------------ #
    def train(self) -> None:  # noqa: C901 – long but readable
        """Epoch/step loop with grad-accumulation and detailed logging."""
        if self._compile:
            log.info("torch.compile active – first forward will be slower.")

        t0 = time.perf_counter()
        run_ce = run_kd = run_ir = n_tok = 0.0

        with self._profiler as prof:
            for epoch in range(self.epochs_run, self.total_epochs):
                self._sampler.set_epoch(epoch)
                pbar = tqdm(total=self._steps_per_epoch, leave=False)

                for step, batch in enumerate(self._dataloader):
                    # -------- optional max_steps_per_epoch -------------------
                    if self.max_steps_per_epoch and (
                        step // self._gradient_accumulation_steps
                    ) == self.max_steps_per_epoch:
                        break

                    # -------- move batch to device ---------------------------
                    batch = {k: v.to(self._device) for k, v in batch.items()}
                    tok_here = (batch["labels"] != self._loss_fn.ignore_index).sum()
                    n_tok += tok_here

                    # -------- forward + backprop ----------------------------
                    total, kd_loss, ir_loss = self._loss_step(batch)
                    run_ce += total * tok_here
                    run_kd += kd_loss * tok_here
                    run_ir += ir_loss * tok_here
                    total.backward()

                    # -------- grad-accum boundary ---------------------------
                    if (step + 1) % self._gradient_accumulation_steps == 0:
                        training.scale_grads(self._model, 1 / n_tok)
                        if self._clip_grad_norm:
                            grad_norm = torch.nn.utils.clip_grad_norm_(self._model.parameters(), float(self._clip_grad_norm))
                        self._optimizer.step()
                        self._optimizer.zero_grad(set_to_none=True)
                        self._lr_scheduler.step()
                        self.global_step += 1

                        # ---- metrics --------------------------------------
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

                        # Reset running windows
                        run_ce = run_kd = run_ir = n_tok = 0.0
                        t0 = time.perf_counter()

                    prof.step()

                self.epochs_run += 1
                self.save_checkpoint(epoch)
                pbar.close()

    # ------------------------------------------------------------------ #
    # Cleanup                                                            #
    # ------------------------------------------------------------------ #
    def cleanup(self) -> None:
        """Flush metric logger / profiler handles."""
        self._metric_logger.close()


# =============================================================================
#  Entrypoint
# =============================================================================
@config.parse
def recipe_main(cfg: DictConfig) -> None:
    """
    Script-level entry point – invoked by torchtune launcher, e.g.

        torchrun -m torchtune.run --config path/to/your.yaml
    """
    config.log_config("KDRecipeSingleDevice", cfg)
    rec = KDRecipeSingleDevice(cfg)
    rec.setup(cfg)
    rec.train()
    rec.cleanup()


if __name__ == "__main__":
    sys.exit(recipe_main())
