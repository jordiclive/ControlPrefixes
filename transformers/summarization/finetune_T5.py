#!/usr/bin/env python

#!/usr/bin/env python

import argparse
import glob
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from callbacks import Seq2SeqLoggingCallback
from transformers import MBartTokenizer, T5ForConditionalGeneration
from transformers.modeling_bart import shift_tokens_right
from utils2 import (
    ROUGE_KEYS,
    LegacySeq2SeqDataset,
    Seq2SeqDataset,
    assert_all_frozen,
    calculate_bleu,
    calculate_rouge,
    flatten_list,
    freeze_params,
    get_git_info,
    label_smoothed_nll_loss,
    lmap,
    pickle_save,
    save_git_info,
    use_task_specific_params,
)

import wandb

from lightning_base_T5 import add_generic_args, generic_train, PrefixTransformer  # noqa


os.system(f'rm -rf /Users/jordi/Desktop/data/X_SUM/o')


# import wandb



logger = logging.getLogger(__name__)


class PrefixSummarizationModule(PrefixTransformer):
    mode = "summarization"
    loss_names = ["loss"]
    metric_names = ROUGE_KEYS
    default_val_metric = "rouge2"

    def __init__(self, hparams, **kwargs):
        if hparams.sortish_sampler and hparams.gpus > 1:
            hparams.replace_sampler_ddp = False
        elif hparams.max_tokens_per_batch is not None:
            if hparams.gpus > 1:
                raise NotImplementedError("Dynamic Batch size does not work for multi-gpu training")
            if hparams.sortish_sampler:
                raise ValueError("--sortish_sampler and --max_tokens_per_batch may not be used simultaneously")
        super().__init__(hparams, num_labels=None, mode=self.mode, **kwargs)
        use_task_specific_params(self.model, "GEC")
        #if self.hparams.git:
            #save_git_info(self.hparams.output_dir)
        self.metrics_save_path = Path(self.output_dir) / "metrics.json"
        self.hparams_save_path = Path(self.output_dir) / "hparams.pkl"
        pickle_save(self.hparams, self.hparams_save_path)
        self.step_count = 0
        self.metrics = defaultdict(list)
        self.model_type = self.config.model_type
        self.vocab_size = self.config.tgt_vocab_size if self.model_type == "fsmt" else self.config.vocab_size

        if self.hparams.T5_preamble:
            p = "summarize: " #"translate from Graph to English:"
        else:
            p = None
        self.dataset_kwargs: dict = dict(
            data_dir=self.hparams.data_dir,
            max_source_length=self.hparams.max_source_length,
            prefix=p
        )
        n_observations_per_split = {
            "train": self.hparams.n_train,
            "val": self.hparams.n_val,
            "test": self.hparams.n_test,
        }
        self.n_obs = {k: v if v >= 0 else None for k, v in n_observations_per_split.items()}

        self.target_lens = {
            "train": self.hparams.max_target_length,
            "val": self.hparams.val_max_target_length,
            "test": self.hparams.test_max_target_length,
        }
        assert self.target_lens["train"] <= self.target_lens["val"], f"target_lens: {self.target_lens}"
        assert self.target_lens["train"] <= self.target_lens["test"], f"target_lens: {self.target_lens}"

        self.num_workers = 4


        # self.seq2seq_model.encoder.embed_tokens.trainable_weight.requires_grad = True
        ##
        print('FREEZING ENTIRE seq2seq model.')
        freeze_params(self.seq2seq_model)
        assert_all_frozen(self.seq2seq_model)
        # if self.hparams.freeze_encoder:
        #     freeze_params(self.model.get_encoder())
        #     assert_all_frozen(self.model.get_encoder())
        #if self.hparams.git:
            #self.hparams.git_sha = get_git_info()["repo_sha"]
        self.decoder_start_token_id = None  # default to config
        if self.model.config.decoder_start_token_id is None and isinstance(self.tokenizer, MBartTokenizer):
            self.decoder_start_token_id = self.tokenizer.lang_code_to_id[hparams.tgt_lang]
            self.model.config.decoder_start_token_id = self.decoder_start_token_id
        self.dataset_class = (
            Seq2SeqDataset if hasattr(self.tokenizer, "prepare_seq2seq_batch") else LegacySeq2SeqDataset
        )
        self.eval_beams = self.model.config.num_beams if self.hparams.eval_beams is None else self.hparams.eval_beams
        assert self.eval_beams >= 1, f"got self.eval_beams={self.eval_beams}. Need an integer > 1"
        if self.hparams.eval_max_gen_length is not None:
            self.eval_max_length = self.hparams.eval_max_gen_length
        else:
            self.eval_max_length = self.model.config.max_length
        self.val_metric = self.default_val_metric if self.hparams.val_metric is None else self.hparams.val_metric

        self.training_acc_across_batches_at_curr_epoch = []

        self.eval_max_length = 60
        self.eval_min_length = 10
        self.eval_beams = 6

        print('for decoding, eval_max_length={}, '
              'eval_min_length={}, eval_beams={}'.format(self.eval_max_length, self.eval_min_length, self.eval_beams))


    def freeze_embeds(self):
        """Freeze token embeddings and positional embeddings for bart, just token embeddings for t5."""
        if self.model_type == "t5":
            freeze_params(self.model.shared)
            for d in [self.model.encoder, self.model.decoder]:
                freeze_params(d.embed_tokens)
        elif self.model_type == "fsmt":
            for d in [self.model.model.encoder, self.model.model.decoder]:
                freeze_params(d.embed_positions)
                freeze_params(d.embed_tokens)
        else:
            freeze_params(self.model.model.shared)
            for d in [self.model.model.encoder, self.model.model.decoder]:
                freeze_params(d.embed_positions)
                freeze_params(d.embed_tokens)

    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, frozen_model=self.seq2seq_model, **kwargs)

    def ids_to_clean_text(self, generated_ids: List[int]):
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return lmap(str.strip, gen_text)

    def _step(self, batch: dict) -> Tuple:
        pad_token_id = self.tokenizer.pad_token_id
        src_ids, src_mask = batch["input_ids"], batch["attention_mask"]
        tgt_ids = batch["labels"]

        decoder_input_ids = self.seq2seq_model._shift_right(tgt_ids)

        outputs = self(src_ids, attention_mask=src_mask, decoder_input_ids=decoder_input_ids, use_cache=False,
                       use_prefix=True)

        lm_logits = outputs[0]
        if self.hparams.label_smoothing == 0:
            # Same behavior as modeling_bart.py, besides ignoring pad_token_id
            ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)

            # assert lm_logits.shape[-1] == self.vocab_size
            # print(lm_logits.shape, tgt_ids.shape, lm_logits.shape[-1] )
            loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), tgt_ids.view(-1))
        else:
            lprobs = torch.nn.functional.log_softmax(lm_logits, dim=-1)
            loss, nll_loss = label_smoothed_nll_loss(
                lprobs, tgt_ids, self.hparams.label_smoothing, ignore_index=pad_token_id
            )
        return (loss,)

    # def on_after_backward(self):
    #     print('ACCcumulate')

    @property
    def pad(self) -> int:
        return self.tokenizer.pad_token_id

    def training_step(self, batch, batch_idx) -> Dict:


        loss_tensors = self._step(batch)

        logs = {name: loss for name, loss in zip(self.loss_names, loss_tensors)}
        # tokens per batch
        logs["tpb"] = batch["input_ids"].ne(self.pad).sum() + batch["labels"].ne(self.pad).sum()
        logs["bs"] = batch["input_ids"].shape[0]
        logs["src_pad_tok"] = batch["input_ids"].eq(self.pad).sum()
        logs["src_pad_frac"] = batch["input_ids"].eq(self.pad).float().mean()

        self.training_acc_across_batches_at_curr_epoch.append(loss_tensors[0].item())
        self.log_dict(logs)
        loss = loss_tensors[0]
        return {"loss": loss}

    def on_epoch_end(self):
        #todo work out inf loss

        #train_acc_mean = np.mean(self.training_acc_across_batches_at_curr_epoch)
        train_acc_mean = np.mean([i if abs(i) < 10 else 10 for i in self.training_acc_across_batches_at_curr_epoch])
        self.log_dict({'train_loss': train_acc_mean})
        print('train_loss = {}'.format(train_acc_mean))
        # print('train_PPL = {}'.format(train_acc_mean.exp()))
        self.training_acc_across_batches_per_epoch = []  # reset for next epoch

    def validation_step(self, batch, batch_idx) -> Dict:
        if self.current_epoch < 1:
            return 1

        if self.hparams.skip_val:
            return 1

        if self.hparams.hf_checkpoint:
            save_path = Path(self.hparams.save_hf)
            save_path = save_path.joinpath("checkpoint-curr_best")
            # print('the suggested save_path is {}, saving to {}'.format(filepath[:-5], save_path))

            self.model.config.save_step = self.step_count
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            print('SAVING TO checkpoint {}'.format(save_path))
            raise ValueError("just_saving")

        return self._generative_step(batch)

    def validation_epoch_end(self, outputs, prefix="val") -> Dict:

        if self.current_epoch < 1:
            logg = (0.1)
            self.log('val_rouge2', logg)
            return 1

        if self.hparams.skip_val:
            logg = (0.1)
            self.log('val_rouge2', logg)
            return 1

        self.step_count += 1
        losses = {k: torch.stack([x[k] for x in outputs]).mean() for k in self.loss_names}
        loss = losses["loss"]

        generative_metrics = {
            k: np.array([x[k] for x in outputs]).mean() for k in self.metric_names + ["gen_time", "gen_len"]
        }
        metric_val = (
            generative_metrics[self.val_metric] if self.val_metric in generative_metrics else losses[self.val_metric]
        )
        metric_tensor: torch.FloatTensor = torch.tensor(metric_val).type_as(loss)
        print('ROUGE2', metric_tensor)
        print('VAL_LOSS', loss)
        generative_metrics.update({k: v.item() for k, v in losses.items()})
        losses.update(generative_metrics)
        all_metrics = {f"{prefix}_avg_{k}": x for k, x in losses.items()}
        all_metrics["step_count"] = self.step_count
        self.metrics[prefix].append(all_metrics)  # callback writes this to self.metrics_save_path
        self.log_dict({
            "log": all_metrics,
            f"{prefix}_loss": loss,
            f"{prefix}_{self.val_metric}": metric_tensor,
        })
        preds = flatten_list([x["preds"] for x in outputs])

        return {
            "log": all_metrics,
            "preds": preds,
            f"{prefix}_loss": loss,
            f"{prefix}_{self.val_metric}": metric_tensor,
        }


    def calc_generative_metrics(self, preds, target) -> Dict:
        return calculate_rouge(preds, target)

    def _generative_step(self, batch: dict,batch_idx=None, dataloader_idx=None) -> dict:
        t0 = time.time()
        bsz = batch["input_ids"].size(0)
        prefix_prompt = self.model.get_prompt(bsz = bsz, sample_size = self.eval_beams)
        # print(prefix_prompt)
        generated_ids = self.seq2seq_model.generate(
            batch["input_ids"],
            past_key_values = prefix_prompt,
            attention_mask = batch["attention_mask"],
            use_cache = True,
            length_penalty = self.hparams.length_penalty,
            use_prefix = True,
            decoder_start_token_id = self.decoder_start_token_id,
            num_beams = self.eval_beams,
            min_length = self.eval_min_length,
            max_length = self.eval_max_length,
            no_repeat_ngram_size = 3
        )
        gen_time = (time.time() - t0) / batch["input_ids"].shape[0]
        preds: List[str] = self.ids_to_clean_text(generated_ids)
        target: List[str] = self.ids_to_clean_text(batch["labels"])
        loss_tensors = self._step(batch)
        base_metrics = {name: loss for name, loss in zip(self.loss_names, loss_tensors)}
        rouge: Dict = self.calc_generative_metrics(preds, target)
        summ_len = np.mean(lmap(len, generated_ids))
        base_metrics.update(gen_time = gen_time, gen_len = summ_len, preds = preds, target = target, **rouge)

        return base_metrics

    def test_step(self, batch, batch_idx):
        return self._generative_step(batch)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs, prefix="test")

    def get_dataset(self, type_path) -> Seq2SeqDataset:
        n_obs = self.n_obs[type_path]
        max_target_length = self.target_lens[type_path]
        dataset = self.dataset_class(
            self.tokenizer,
            type_path = type_path,
            n_obs = n_obs,
            max_target_length = max_target_length,
            **self.dataset_kwargs,
        )
        return dataset

    def get_dataloader(self, type_path: str, batch_size: int, shuffle: bool = False) -> DataLoader:
        dataset = self.get_dataset(type_path)

        if self.hparams.sortish_sampler and type_path != "test":
            sampler = dataset.make_sortish_sampler(batch_size, distributed = self.hparams.gpus > 1)
            return DataLoader(
                dataset,
                batch_size = batch_size,
                collate_fn = dataset.collate_fn,
                shuffle = False,
                num_workers = 4,
                sampler = sampler,
            )

        elif self.hparams.max_tokens_per_batch is not None and type_path != "test":
            batch_sampler = dataset.make_dynamic_sampler(
                self.hparams.max_tokens_per_batch, distributed = self.hparams.gpus > 1
            )
            return DataLoader(
                dataset,
                batch_sampler = batch_sampler,
                collate_fn = dataset.collate_fn,
                # shuffle=False,
                num_workers = self.num_workers,
                # batch_size=None,
            )
        else:
            return DataLoader(
                dataset,
                batch_size = batch_size,
                collate_fn = dataset.collate_fn,
                shuffle = shuffle,
                num_workers = self.num_workers,
                sampler = None,
            )

    def train_dataloader(self) -> DataLoader:
        dataloader = self.get_dataloader("train", batch_size = self.hparams.train_batch_size, shuffle = True)
        return dataloader

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader("val", batch_size = self.hparams.eval_batch_size)

    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader("test", batch_size = self.hparams.eval_batch_size)

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        PrefixTransformer.add_model_specific_args(parser, root_dir)
        add_generic_args(parser, root_dir)
        parser.add_argument(
            "--max_source_length",
            default=512, #1024
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--max_target_length",
            default=60, #56
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--val_dir",
            default = "",
            type = str,
            help = "The maximum total input sequence length after tokenization. Sequences longer "
                   "than this will be truncated, sequences shorter will be padded.",
        )

        parser.add_argument("--skip_train", type = bool, default = False)

        parser.add_argument(
            "--val_max_target_length",
            default=60,  #142 # these defaults are optimized for CNNDM. For xsum, see README.md.
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--test_max_target_length",
            default=100, #142
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument("--freeze_encoder", action="store_true")
        parser.add_argument("--freeze_embeds", action="store_true")
        parser.add_argument("--sortish_sampler", action="store_true", default=False)
        parser.add_argument("--max_tokens_per_batch", type=int, default=None)
        parser.add_argument("--logger_name", type=str, choices=["default", "wandb", "wandb_shared"], default="default")
        parser.add_argument("--n_train", type=int, default=-1, required=False, help="# examples. -1 means use all.")
        parser.add_argument("--n_val", type=int, default=-1, required=False, help="# examples. -1 means use all.")
        parser.add_argument("--n_test", type=int, default=-1, required=False, help="# examples. -1 means use all.")
        parser.add_argument(
            "--task_mode", type=str, default="GEC", required=False, help="# examples. -1 means use all."
        )
        parser.add_argument("--label_smoothing", type=float, default=0.0, required=False)
        parser.add_argument("--src_lang", type=str, default="", required=False)
        parser.add_argument("--save_hf", type=str, default="", required=False)

        parser.add_argument("--tgt_lang", type=str, default="", required=False)
        parser.add_argument("--eval_beams", type=int, default=6, required=False)
        parser.add_argument("--eval_min_length", type=int, default=10, required=False)
        parser.add_argument("--skip_val", type = bool, default = False, required = False)

        parser.add_argument(
            "--val_metric", type=str, default=None, required=False
        )
        parser.add_argument("--eval_max_gen_length", type=int, default=60, help="never generate more than n tokens")
        parser.add_argument("--length_penalty", type=float, default=1.0, help="never generate more than n tokens")
        parser.add_argument("--save_top_k", type=int, default=1, required=False, help="How many checkpoints to save")
        parser.add_argument("--wb_project", type = str,default="")
        parser.add_argument("--git", type =bool,default=True)
        parser.add_argument("--dev", type =bool,default=False)
        parser.add_argument("--freeze_base", type = bool, default = False)
        parser.add_argument("--wb_name", type = str,default="")
        parser.add_argument("--id", type = str, default = "")
        parser.add_argument(
            "--DART",
            default=True, #56
            type=bool,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--early_stopping_patience",
            type=int,
            default=-1,
            required=False,
            help="-1 means never early stop. early_stopping_patience is measured in validation checks, not epochs. So val_check_interval will effect it.",
        )
        parser.add_argument(
            "--T5_preamble",
            type = bool,
            default = False,
            required = False,
            help = "-1 means never early stop. early_stopping_patience is measured in validation checks, not epochs. So val_check_interval will effect it.",
        )

        parser.add_argument(
            "--restart_with_embed",
            type = bool,
            default = False,
            required = False,
            help = "-1 means never early stop. early_stopping_patience is measured in validation checks, not epochs. So val_check_interval will effect it.",
        )



        return parser


def eval(args, model=None):

    model = PrefixSummarizationModule(args)


    print('the length penalty is {}'.format(args.length_penalty))

    with torch.no_grad():
        model.eval()
        model = model.cuda()

        data_loader = model.test_dataloader()
        print('DATALOADER_LEN',len(data_loader))
        out_lst = []
        for batch_idx, batch in enumerate(data_loader):

            batch = model.transfer_batch_to_device(batch, model.device)
            out = model.test_step(batch, batch_idx)
            out_lst.append(out)
            if batch_idx % 50 == 0:
                print(model.test_epoch_end(out_lst))
                print(out['preds'])
        result = model.test_epoch_end(out_lst)

    for k, v in result.items():
        if k != 'preds':
            print('FINAL_RESULTS')
            print(k, v)

    out_path = os.path.join(args.output_dir, 'test_beam_{}'.format(args.length_penalty))
    print('writing the test results to ', out_path)
    with open(out_path, 'w') as f:
        for preds in result['preds']:
            print(preds, file = f)

def main(args, model=None):
    if os.path.exists(args.output_dir):
        raise ValueError("--Previous Experiment, delete folder if want to overwrite")

    Path(args.output_dir).mkdir(exist_ok = True)

    # for n, p in model.named_parameters():
    #     print(n, p.requires_grad)
    if model is None:
        model = PrefixSummarizationModule(args)

    pickle_save(model.hparams, model.output_dir / "hparams.pkl")
    dataset = Path(args.data_dir).name



    # batch = pickle_load('/Users/jordi/Desktop/Master/prefix_tuning/transformers/GEC/b1.pkl')
    # model._generative_step(batch)
    print(dataset)

    if (
            args.logger_name == "default"
            or args.fast_dev_run
            or str(args.output_dir).startswith("/tmp")
            or str(args.output_dir).startswith("/var")
    ):
        logger = True  # don't pollute wandb logs unnecessarily
    elif args.logger_name == "wandb":
        from pytorch_lightning.loggers import WandbLogger
        if args.id is not None:
            id_ = args.id
        else:
            id_ = wandb.util.generate_id()
            print('ID', id_)
        logger = WandbLogger(id = id_, name = args.wb_name, project = args.wb_project, entity = 'jordiclive')
        # logger.log_hyperparams(model.hparams.git_sha)


    if args.early_stopping_patience >= 0:
        es_callback = get_early_stopping_callback(model.val_metric, args.early_stopping_patience)
    else:
        es_callback = False


    trainer: pl.Trainer = generic_train(
        model,
        args,
        logging_callback = Seq2SeqLoggingCallback(),
        logger = logger,
    )
    pickle_save(model.hparams, model.output_dir / "hparams.pkl")



    if not args.do_predict:
        return model



    if args.test_checkpoint is not None:
        checkpoints = [args.test_checkpoint]
        model.hparams.test_checkpoint = checkpoints[-1]
        trainer.resume_from_checkpoint = checkpoints[-1]

        if args.do_predict and args.skip_train:
            checkpoint = checkpoints[-1]
            print(checkpoint)

            trainer.test(model, ckpt_path = checkpoint)
            return model

    trainer.test()
    return model





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = PrefixSummarizationModule.add_model_specific_args(parser, os.getcwd())
    args = parser.parse_args()
    pl.seed_everything(args.seed)


    model = main(args)
