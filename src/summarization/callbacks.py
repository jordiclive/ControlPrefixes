import logging
import os
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_only
from utils_conditional import save_json

import types
import warnings

from torch.optim.lr_scheduler import _LRScheduler, LambdaLR


def count_trainable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


logger = logging.getLogger(__name__)


class Seq2SeqLoggingCallback(pl.Callback):
    def on_batch_end(self, trainer, pl_module):
        lrs = {
            f"lr_group_{i}": param["lr"]
            for i, param in enumerate(pl_module.trainer.optimizers[0].param_groups)
        }
        pl_module.logger.log_metrics(lrs)

    @rank_zero_only
    def _write_logs(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        type_path: str,
        save_generations=True,
    ) -> None:
        logger.info(
            f"***** {type_path} results at step {trainer.global_step:05d} *****"
        )
        metrics = trainer.callback_metrics
        trainer.logger.log_metrics(
            {
                k: v
                for k, v in metrics.items()
                if k not in ["log", "progress_bar", "preds"]
            }
        )
        # Log results
        od = Path(pl_module.hparams.output_dir)
        if type_path == "test":
            results_file = od / "test_results.txt"
            generations_file = od / "test_generations.txt"
        else:

            results_file = od / f"{type_path}_results/{trainer.global_step:05d}.txt"
            generations_file = (
                od / f"{type_path}_generations/{trainer.global_step:05d}.txt"
            )
            results_file.parent.mkdir(exist_ok=True)
            generations_file.parent.mkdir(exist_ok=True)
        with open(results_file, "a+") as writer:
            for key in sorted(metrics):
                if key in ["log", "progress_bar", "preds"]:
                    continue
                val = metrics[key]
                if isinstance(val, torch.Tensor):
                    val = val.item()
                msg = f"{key}: {val:.6f}\n"
                writer.write(msg)

        if not save_generations:
            return

        if "preds" in metrics:
            content = "\n".join(metrics["preds"])
            generations_file.open("w+").write(content)

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        try:
            npars = pl_module.model.model.num_parameters()
        except AttributeError:
            npars = pl_module.model.num_parameters()

        n_trainable_pars = count_trainable_parameters(pl_module)
        trainer.logger.log_metrics(
            {"n_params": npars, "mp": npars / 1e6, "grad_mp": n_trainable_pars / 1e6}
        )

    @rank_zero_only
    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        save_json(pl_module.metrics, pl_module.metrics_save_path)
        return self._write_logs(trainer, pl_module, "test")

    @rank_zero_only
    def on_validation_end(self, trainer: pl.Trainer, pl_module):
        save_json(pl_module.metrics, pl_module.metrics_save_path)


def bespoke_scheduler(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )

    def lr_lambda2(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps * 3 - current_step)
            / float(max(1, num_training_steps * 3 - num_warmup_steps)),
        )

    return LambdaLR(optimizer, [lr_lambda, lr_lambda2], last_epoch)
