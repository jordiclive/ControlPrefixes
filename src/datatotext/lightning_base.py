import argparse
import logging
import os
from pathlib import Path
from typing import Any, Dict

import pytorch_lightning as pl
from callbacks import bespoke_scheduler
from controlprefixes import ControlPrefixes
from pytorch_lightning.utilities import rank_zero_info
from transformers import (
    AdamW,
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedTokenizer,
)
from transformers.modeling_t5 import T5ForConditionalGeneration
from transformers.optimization import (
    Adafactor,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)

logger = logging.getLogger(__name__)

arg_to_scheduler = {
    "linear": get_linear_schedule_with_warmup,
    "cosine": get_cosine_schedule_with_warmup,
    "cosine_w_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
    "polynomial": get_polynomial_decay_schedule_with_warmup,
}
arg_to_scheduler_choices = sorted(arg_to_scheduler.keys())
arg_to_scheduler_metavar = "{" + ", ".join(arg_to_scheduler_choices) + "}"


class PrefixTransformer(pl.LightningModule):
    def __init__(
        self,
        hparams: argparse.Namespace,
        num_labels=None,
        config=None,
        tokenizer=None,
        seq2seq_model=None,
        **config_kwargs,
    ):
        """Initialize a model, tokenizer and config."""
        super().__init__()

        self.save_hyperparameters(hparams)
        self.step_count = 0
        self.output_dir = Path(self.hparams.output_dir)
        cache_dir = self.hparams.cache_dir if self.hparams.cache_dir else None
        rank_zero_info("the cache dir is {}".format(cache_dir))
        if config is None:
            self.config = AutoConfig.from_pretrained(
                self.hparams.config_name
                if self.hparams.config_name
                else self.hparams.model_name_or_path,
                **({"num_labels": num_labels} if num_labels is not None else {}),
                cache_dir=cache_dir,
                **config_kwargs,
            )
        else:
            self.config: PretrainedConfig = config

        extra_model_params = (
            "encoder_layerdrop",
            "decoder_layerdrop",
            "dropout",
            "attention_dropout",
        )
        for p in extra_model_params:
            if getattr(self.hparams, p, None):
                assert hasattr(
                    self.config, p
                ), f"model config doesn't have a `{p}` attribute"
                setattr(self.config, p, getattr(self.hparams, p))

        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.hparams.tokenizer_name
                if self.hparams.tokenizer_name
                else self.hparams.model_name_or_path,
                cache_dir=cache_dir,
            )
            if self.hparams.new_tokens:
                new_tokens = ["<H>", "<R>", "<T>"]
                if self.hparams.control_token_DART:
                    new_tokens.extend(
                        [
                            "<e2e>",
                            "<webnlg_old>",
                            "<WikiTableQuestions_lily>",
                            "<WikiSQL_decl_sents>",
                            "<WikiTableQuestions_mturk>",
                            "<WikiSQL_lily>",
                        ]
                    )

                self.new_token_len = len(new_tokens)
                new_tokens_vocab = {}
                new_tokens_vocab["additional_special_tokens"] = []
                for idx, t in enumerate(new_tokens):
                    new_tokens_vocab["additional_special_tokens"].append(t)
                num_added_toks = self.tokenizer.add_special_tokens(new_tokens_vocab)
                rank_zero_info("We have added %s tokens", num_added_toks)
        else:
            self.tokenizer: PreTrainedTokenizer = tokenizer

        self.config.preseqlen = self.hparams.preseqlen
        if self.hparams.control_prefixes:
            if self.hparams.DART:
                self.config.preseqlen += self.hparams.m_prefix_len
            else:
                self.config.preseqlen += self.hparams.m_prefix_len * 2

        self.config.use_prefix = True

        self.seq2seq_model_type = AutoModel

        if seq2seq_model is None:
            self.seq2seq_model = T5ForConditionalGeneration.from_pretrained(
                self.hparams.model_name_or_path,
                from_tf=bool(".ckpt" in self.hparams.model_name_or_path),
                config=self.config,
                cache_dir=cache_dir,
            )
        else:
            self.seq2seq_model = seq2seq_model

        self.seq2seq_model.resize_token_embeddings(len(self.tokenizer))

        config_prefix = AutoConfig.from_pretrained(
            self.hparams.model_name_or_path, cache_dir=cache_dir
        )
        self.model_type = config_prefix.model_type

        if self.hparams.optim_prefix == "yes":
            optim_prefix_bool = True
        elif self.hparams.optim_prefix == "no":
            optim_prefix_bool = False
        else:
            assert False, "model_args.optim_prefix should be either yes or no"

        rank_zero_info(self.model_type)
        config_prefix._my_arg_tune_mode = self.hparams.tuning_mode
        config_prefix._my_arg_task_mode = self.hparams.task_mode
        config_prefix._my_arg_control = True
        config_prefix.train_weights = False
        config_prefix.optim_prefix = optim_prefix_bool
        config_prefix.preseqlen = self.hparams.preseqlen
        config_prefix.use_infix = self.hparams.format_mode == "infix"
        config_prefix.format_mode = self.hparams.format_mode
        config_prefix.prefix_dropout = self.hparams.prefix_dropout
        config_prefix.vocab_size = len(self.tokenizer)
        config_prefix.DART = self.hparams.DART

        config_prefix.lowdata = "lowdata" in self.hparams.output_dir
        if config_prefix.lowdata and self.hparams.use_lowdata_token == "yes":
            config_prefix.lowdata_token = self.tokenizer(
                [self.hparams.lowdata_token], add_prefix_space=True
            )["input_ids"]
            rank_zero_info(self.hparams.lowdata_token)
            rank_zero_info(config_prefix.lowdata_token)
            rank_zero_info(self.tokenizer.pad_token_id)

        config_prefix.mid_dim = self.hparams.mid_dim
        config_prefix.new_token_len = self.new_token_len

        if self.hparams.control_prefixes:
            config_prefix.m_prefix_mid_dim = self.hparams.m_prefix_mid_dim
            config_prefix.m_prefix_len = self.hparams.m_prefix_len
        if self.hparams.unseen:
            config_prefix.unseen = True

        if self.hparams.prefixModel_name_or_path is not None:
            rank_zero_info("loading from {}".format(hparams.prefixModel_name_or_path))

            self.model = ControlPrefixes.from_pretrained(
                self.hparams.prefixModel_name_or_path,
                from_tf=bool(".ckpt" in self.hparams.prefixModel_name_or_path),
                cache_dir=cache_dir,
                config=config_prefix,
            )
        else:
            self.model = ControlPrefixes(config_prefix)

    def load_hf_checkpoint(self, *args, **kwargs):
        assert False, "why need to load model here?"
        self.model = self.model_type.from_pretrained(*args, **kwargs)

    def get_lr_scheduler(self):
        get_schedule_func = arg_to_scheduler[self.hparams.lr_scheduler]
        scheduler = get_schedule_func(
            self.opt,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.total_steps,
        )
        rank_zero_info(f"warm up {self.hparams.warmup_steps}")
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return scheduler

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""

        if self.hparams.different_scheduler:
            cefr_params = [
                p
                for n, p in self.named_parameters()
                if any(nd in n for nd in ["CEFR_matrices.wte"])
            ]
            no_cefr_params = [
                p
                for n, p in self.named_parameters()
                if not any(nd in n for nd in ["CEFR_matrices.wte"])
            ]
            optimizer_grouped_parameters = [
                {
                    "params": no_cefr_params,
                    "weight_decay": self.hparams.weight_decay,
                },
                {
                    "params": cefr_params,
                    "weight_decay": self.hparams.weight_decay,
                },
            ]
            if self.hparams.adafactor:
                optimizer = Adafactor(
                    optimizer_grouped_parameters,
                    lr=self.hparams.learning_rate,
                    scale_parameter=False,
                    relative_step=False,
                )

            else:
                optimizer = AdamW(
                    optimizer_grouped_parameters,
                    lr=self.hparams.learning_rate,
                    eps=self.hparams.adam_epsilon,
                )
            self.opt = optimizer

            scheduler = bespoke_scheduler(
                self.opt,
                num_warmup_steps=self.hparams.warmup_steps,
                num_training_steps=self.total_steps,
            )
            rank_zero_info(f"warm up {self.hparams.warmup_steps}")
            scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

            return [optimizer], [scheduler]

    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)

    def test_epoch_end(self, outputs):
        return self.validation_end(outputs)

    @property
    def total_steps(self) -> int:
        """The number of total training steps that will be run. Used for lr scheduler purposes."""
        num_devices = max(1, self.hparams.gpus)
        if self.hparams.original_batch_size is not None:
            effective_batch_size = (
                self.hparams.original_batch_size
                * self.hparams.accumulate_grad_batches
                * num_devices
            )
        else:
            effective_batch_size = (
                self.hparams.train_batch_size
                * self.hparams.accumulate_grad_batches
                * num_devices
            )
        dataset_size = len(self.train_loader.dataset)
        return (dataset_size / effective_batch_size) * self.hparams.max_epochs

    def setup(self, mode):
        if mode == "fit":
            self.train_loader = self.get_dataloader(
                "train", self.hparams.train_batch_size, shuffle=True
            )

    def get_dataloader(self, type_path, batch_size, shuffle=False):
        raise NotImplementedError("You must implement this for your task")

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.get_dataloader("dev", self.hparams.eval_batch_size, shuffle=False)

    def test_dataloader(self):
        return self.get_dataloader("test", self.hparams.eval_batch_size, shuffle=False)

    def _feature_file(self, mode):
        return os.path.join(
            self.hparams.data_dir,
            "cached_{}_{}_{}".format(
                mode,
                list(filter(None, self.hparams.model_name_or_path.split("/"))).pop(),
                str(self.hparams.max_seq_length),
            ),
        )

    @pl.utilities.rank_zero_only
    def save_checkpoint(self, trainer) -> None:
        rank_zero_info("Saving the the checkpoint.")
        return

    @pl.utilities.rank_zero_only
    def on_save_checkpoint(self, checkpoint: Dict[str, Any], filepath=None) -> None:
        rank_zero_info("SEQ", self.seq2seq_model.shared.trainable_weight)
        self.model.es.trainable_weight = self.seq2seq_model.shared.trainable_weight
        rank_zero_info("Prefix_stored_weight", self.model.es.trainable_weight)
        save_path = self.output_dir.joinpath("checkpoint-curr_best")

        self.model.config.save_step = self.step_count
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        rank_zero_info("SAVING TO checkpoint {}".format(save_path))

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        parser.add_argument(
            "--model_name_or_path",
            default="t5-large",
            type=str,
            required=False,
            help="Path to pretrained model or model identifier from huggingface.co/models",
        )

        parser.add_argument(
            "--prefixModel_name_or_path",
            default=None,
            type=str,
            help="Path to pretrained prefix model or model identifier from huggingface.co/models",
        )

        parser.add_argument(
            "--prefix_mode",
            default="activation",
            type=str,
            help="embedding or activation",
        )

        parser.add_argument(
            "--preseqlen",
            default=200,
            type=int,
            help="the length of the prefix.",
        )

        parser.add_argument(
            "--optim_prefix",
            default="yes",
            type=str,
            help="use the task specific optimization of the prefix.",
        )

        parser.add_argument(
            "--different_scheduler",
            default=False,
            type=bool,
            help="use a different lr scheduler for control prefixes and main prefix",
        )

        parser.add_argument(
            "--tuning_mode",
            default="prefixtune",
            type=str,
            help="Could be prefixtune or finetune",
        )

        parser.add_argument(
            "--prefix_dropout",
            default=0.0,
            type=float,
            help="the dropout rate for our prefix model.",
        )

        parser.add_argument(
            "--use_dropout",
            default="no",
            type=str,
            help="whether to dropout the main model during training. ",
        )

        parser.add_argument(
            "--mid_dim",
            default=800,
            type=int,
            help="the dimension of the intermediate layer of themain  prefix reparameterization",
        )
        parser.add_argument(
            "--m_prefix_mid_dim",
            default=512,
            type=int,
            help="the dimension of the intermediate layer of the control prefix reparameterizations",
        )

        parser.add_argument(
            "--m_prefix_len",
            default=1,
            type=int,
            help="the control prefix length",
        )
        parser.add_argument(
            "--unseen",
            default=False,
            type=bool,
            help="Initializing a control prefix for unseen categories to zero",
        )

        parser.add_argument(
            "--format_mode",
            default="cat",
            type=str,
            help="whether to look at the input again, including [infix, cat, peek, nopeek]",
        )

        parser.add_argument(
            "--use_lowdata_token",
            default="yes",
            type=str,
            help="whether or not to use the lowdata token, ",
        )

        parser.add_argument(
            "--lowdata_token",
            default="summarize",
            type=str,
            help="the low data token to use. ",
        )

        parser.add_argument(
            "--config_name",
            default="",
            type=str,
            help="Pretrained config name or path if not the same as model_name",
        )
        parser.add_argument(
            "--tokenizer_name",
            default=None,
            type=str,
            help="Pretrained tokenizer name or path if not the same as model_name",
        )
        parser.add_argument(
            "--cache_dir",
            default="/content/gdrive/MyDrive/cache_dir",
            type=str,
            help="Where do you want to store the pre-trained models downloaded from s3",
        )
        parser.add_argument(
            "--encoder_layerdrop",
            type=float,
            help="Encoder layer dropout probability (Optional). Goes into model.config",
        )
        parser.add_argument(
            "--decoder_layerdrop",
            type=float,
            help="Decoder layer dropout probability (Optional). Goes into model.config",
        )
        parser.add_argument(
            "--dropout",
            type=float,
            help="Dropout probability (Optional). Goes into model.config",
        )
        parser.add_argument(
            "--control_prefixes",
            type=bool,
            default=False,
            help="if using control prefixes",
        )

        parser.add_argument(
            "--new_tokens",
            type=bool,
            default=False,
            help="if using demarcation tokens <H>, <R>, <T> that need to be learnable",
        )
        parser.add_argument(
            "--attention_dropout",
            type=float,
            help="Attention dropout probability (Optional). Goes into model.config",
        )
        parser.add_argument(
            "--learning_rate",
            default=5e-5,
            type=float,
            help="The target learning rate.",
        )
        parser.add_argument(
            "--lr_scheduler",
            default="linear",
            choices=arg_to_scheduler_choices,
            metavar=arg_to_scheduler_metavar,
            type=str,
            help="Learning rate scheduler",
        )
        parser.add_argument(
            "--weight_decay",
            default=0.0,
            type=float,
            help="Weight decay if we apply some.",
        )
        parser.add_argument(
            "--adam_epsilon",
            default=1e-8,
            type=float,
            help="Epsilon for Adam optimizer.",
        )
        parser.add_argument(
            "--warmup_steps",
            default=0,
            type=int,
            help="Linear warmup over warmup_steps.",
        )
        parser.add_argument(
            "--num_workers", default=4, type=int, help="kwarg passed to DataLoader"
        )
        parser.add_argument(
            "--num_train_epochs", dest="max_epochs", default=30, type=int
        )
        parser.add_argument("--original_batch_size", default=None, type=int)
        parser.add_argument(
            "--hf_checkpoint",
            default=False,
            type=bool,
            help="if want to save a hf model checkpoint from a lightning ckpt",
        )
        parser.add_argument("--test_checkpoint", default=None, type=str)

        parser.add_argument("--train_batch_size", default=8, type=int)
        parser.add_argument("--eval_batch_size", default=6, type=int)
        parser.add_argument("--adafactor", action="store_true")


def add_generic_args(parser, root_dir) -> None:
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument("--n_tpu_cores", dest="tpu_cores", type=int)
    parser.add_argument(
        "--max_grad_norm",
        dest="gradient_clip_val",
        default=1.0,
        type=float,
        help="Max gradient norm",
    )

    parser.add_argument(
        "--do_predict", default=True, type=bool, help="Whether to run training."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        dest="accumulate_grad_batches",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--seed", type=int, default=101, help="random seed for initialization"
    )
    parser.add_argument(
        "--control_token_DART",
        type=bool,
        default=False,
        help="if using control tokens for DART source",
    )

    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir.",
    )


def generic_train(
    model,
    args: argparse.Namespace,
    early_stopping_callback=False,
    logger=True,  # can pass WandbLogger() here
    extra_callbacks=[],
    checkpoint_callback=None,
    logging_callback=None,
    **extra_train_kwargs,
):
    pl.seed_everything(args.seed)

    odir = Path(model.hparams.output_dir)
    odir.mkdir(exist_ok=True)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.output_dir,
        monitor="val_" + args.val_metric,
        mode="max",
        save_top_k=args.save_top_k,
        save_last=True,
    )

    if early_stopping_callback is not False:
        extra_callbacks.append(early_stopping_callback)

    rank_zero_info("the max number of epochs is {}".format(args.max_epochs))
    rank_zero_info("early stopping", early_stopping_callback)
    rank_zero_info("checkpoint_callback", checkpoint_callback)
    rank_zero_info("logging", logging_callback)

    trainer = pl.Trainer.from_argparse_args(
        args,
        max_epochs=args.max_epochs,
        weights_summary=None,
        callbacks=[logging_callback] + extra_callbacks,
        logger=logger,
        checkpoint_callback=checkpoint_callback,
    )

    print("args.do_Train:", not (args.skip_train))

    if not args.skip_train:
        trainer.fit(model)

    return trainer
