import os
import sys
import itertools
from typing import Any, Dict, List, Set
import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2.data import MetadataCatalog, build_detection_train_loader, DatasetMapper
from detectron2.engine import AutogradProfiler, DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator, verify_results
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)

sys.path.append(".")
from sparseinst import add_sparse_inst_config, COCOMaskEvaluator



import time # Make sure this import is present
from detectron2.utils.events import get_event_storage # Make sure this import is present

class Trainer(DefaultTrainer):

    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """
        super().__init__(cfg)
        # Get gradient accumulation steps from config, default to 1 (no accumulation)
        self.grad_accum_steps = cfg.SOLVER.get('GRADIENT_ACCUMULATION_STEPS', 1)
        if self.grad_accum_steps > 1:
            # Ensure gradients are zeroed before the first training iteration
            self.optimizer.zero_grad()
            if comm.is_main_process():
                print(f"Gradient accumulation enabled: Will accumulate gradients over {self.grad_accum_steps} steps.")

    def _call_hooks(self, hook_name):
        """
        Helper method to call registered hooks. Replicates logic from SimpleTrainer.
        """
        for h in self._hooks:
            # Use getattr to call the hook method if it exists
            hook_method = getattr(h, hook_name, None)
            if callable(hook_method):
                hook_method(self) # Pass trainer instance to hook method

    def run_step(self):
        """
        Implement the standard training logic with gradient accumulation.
        Overrides DefaultTrainer.run_step.
        """
        assert self.model.training, "[Trainer] model was changed to eval mode!"
        start = time.perf_counter()

        # Get data
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        # === Hooks: before_step ===
        storage = get_event_storage()
        # Ensure storage iteration is aligned with trainer iteration
        storage.iter = self.iter
        self._call_hooks("before_step")
        # ==========================

        # Forward pass
        loss_dict = self.model(data)
        losses = sum(loss_dict.values())

        # Check for invalid losses
        if not torch.isfinite(losses):
            raise FloatingPointError(
                "Loss became infinite or NaN at iteration={}!\nLosses: {}".format(
                    self.iter, loss_dict
                )
            )

        # Scale loss average over accumulation steps
        if self.grad_accum_steps > 1:
            losses = losses / self.grad_accum_steps

        # Backward pass - accumulates gradients
        losses.backward()

        # === Log metrics ===
        # Reduce losses across devices for logging. Log every iteration.
        loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
        # Calculate the total loss reduced across devices. If accumulating, use the scaled loss.
        total_loss_reduced = sum(loss for loss in loss_dict_reduced.values())
        if self.grad_accum_steps > 1:
            total_loss_reduced = total_loss_reduced / self.grad_accum_steps

        if comm.is_main_process():
            storage.put_scalar("data_time", data_time)
            # Log the total loss (potentially scaled if accumulating)
            storage.put_scalar("total_loss", total_loss_reduced)
            # Log individual loss components (unscaled) if available
            if len(loss_dict_reduced) > 1:
                storage.put_scalars(**loss_dict_reduced)
        # ==================

        # === Hooks: after_backward ===
        self._call_hooks("after_backward")
        # =============================

        # Optimizer step and gradient zeroing (conditional)
        # Perform step only after accumulating gradients for grad_accum_steps
        if (self.iter + 1) % self.grad_accum_steps == 0:
            # === Hooks: before_optimizer_step ===
            self._call_hooks("before_optimizer_step")
            # ====================================

            self.optimizer.step()

            # Zero gradients *after* stepping, ready for the next accumulation cycle
            self.optimizer.zero_grad()

        # === Hooks: after_step ===
        # Called after each iteration, regardless of whether an optimizer step was performed.
        self._call_hooks("after_step")
        # =========================


    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                    ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOMaskEvaluator(dataset_name, ("segm", ), True, output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "cityscapes_instance":
            assert (
                    torch.cuda.device_count() >= comm.get_rank() # type: ignore
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                    torch.cuda.device_count() >= comm.get_rank() # type: ignore
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_optimizer(cls, cfg, model):
        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for key, value in model.named_parameters(recurse=True):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if "backbone" in key:
                lr = lr * cfg.SOLVER.BACKBONE_MULTIPLIER
            # for transformer
            if "patch_embed" in key or "cls_token" in key:
                weight_decay = 0.0
            if "norm" in key:
                weight_decay = 0.0
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        def maybe_add_full_model_gradient_clipping(optim):  # optim: the optimizer class
            # detectron2 doesn't have full  model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                    cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                    and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                    and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val) # type: ignore
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR, amsgrad=cfg.SOLVER.AMSGRAD
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def build_train_loader(cls, cfg):
        if cfg.MODEL.SPARSE_INST.DATASET_MAPPER == "SparseInstDatasetMapper":
            from sparseinst import SparseInstDatasetMapper
            mapper = SparseInstDatasetMapper(cfg, is_train=True)
        else:
            # Keep the original behavior if no specific mapper is set
            mapper = DatasetMapper(cfg, is_train=True) if cfg.INPUT.CUSTOM_AUG == '' else \
                build_detection_train_loader(cfg).dataset.dataset._map_func # Fallback logic might vary
            # Or simply:
            # mapper = None # If default build_detection_train_loader handles mapper internally
        return build_detection_train_loader(cfg, mapper=mapper)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_sparse_inst_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "sparseinst" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="sparseinst")
    return cfg


def main(args):
    cfg = setup(args)

    # --- W&B Integration Start ---
    if comm.is_main_process():
        # Initialize wandb run
        # Make sure to set your project name and optionally entity
        # Remove project and entity args if you want to use defaults or environment variables
        run = wandb.init(
            project="your-detectron2-project",  # Replace with your project name
            # entity="your-wandb-entity",    # Optional: replace with your W&B entity (username or team)
            sync_tensorboard=True,       # Auto-sync TensorBoard metrics
            config=cfg,                  # Log Detectron2 config
            name=f"run-{cfg.OUTPUT_DIR.split('/')[-1]}", # Optional: set a run name
            resume="allow",              # Allow resuming runs
            # group="experiment-group",   # Optional: Group runs together
            # job_type="training",        # Optional: Categorize the run
        )
        # Optional: Define metrics for better W&B dashboard visualization
        # wandb.define_metric("train/total_loss", summary="min")
        # wandb.define_metric("validation/coco_eval/bbox/AP", summary="max")
    # --- W&B Integration End ---


    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume)
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        # --- W&B Integration: Finish run after eval ---
        if comm.is_main_process():
            if run: run.finish()
        # --- End W&B ---
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    res = trainer.train()

    # --- W&B Integration: Finish run after training ---
    if comm.is_main_process():
        if run: run.finish()
    # --- End W&B ---
    return res


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    # Ensure wandb is disabled in non-main processes launched by detectron2's launch util
    if not comm.is_main_process():
        os.environ["WANDB_DISABLED"] = "true"

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
