#!/usr/bin/env python

import argparse
import glob
import logging
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import regex as re

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from callbacks import Seq2SeqLoggingCallback
from transformers import MBartTokenizer, T5ForConditionalGeneration
from transformers.modeling_bart import shift_tokens_right
from utils_conditional import (
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
    pickle_load,
    use_task_specific_params,
)

# import wandb

import os

# os.system(f'rm -rf /Users/jordi/Desktop/data/X_SUM/o')



from lightning_base import add_generic_args, generic_train, PrefixTransformer  # noqa



logger = logging.getLogger(__name__)


class PrefixSummarizationModule(PrefixTransformer):
    mode = "summarization"
    loss_names = ["loss"]
    metric_names = ['sari']
    default_val_metric = "sari"

    def __init__(self, hparams, **kwargs):
        if hparams.sortish_sampler and hparams.gpus > 1:
            hparams.replace_sampler_ddp = False
        elif hparams.max_tokens_per_batch is not None:
            if hparams.gpus > 1:
                raise NotImplementedError("Dynamic Batch size does not work for multi-gpu training")
            if hparams.sortish_sampler:
                raise ValueError("--sortish_sampler and --max_tokens_per_batch may not be used simultaneously")
        super().__init__(hparams, num_labels=None, mode=self.mode, **kwargs)
        use_task_specific_params(self.model, "summarization")
        # if self.hparams.git:
        #     save_git_info(self.hparams.output_dir)
        self.metrics_save_path = Path(self.output_dir) / "metrics.json"
        self.hparams_save_path = Path(self.output_dir) / "hparams.pkl"
        pickle_save(self.hparams, self.hparams_save_path)
        self.step_count = 0
        self.metrics = defaultdict(list)
        self.model_type = self.config.model_type
        self.vocab_size = self.config.tgt_vocab_size if self.model_type == "fsmt" else self.config.vocab_size
        self.val_output = self.hparams.val_output
        os.system(f'rm {self.val_output}')
        self.dataset_kwargs: dict = dict(
            data_dir=self.hparams.data_dir,
            max_source_length=self.hparams.max_source_length,
            prefix=self.model.config.prefix or "",
        )
        n_observations_per_split = {
            "train": self.hparams.n_train,
            "val_turk3": self.hparams.n_val,
            "val_turk5":self.hparams.n_val,
            "val_turk6": self.hparams.n_val,
            "val_turk7": self.hparams.n_val,
            "val_turk8": self.hparams.n_val,

            "val_asset3": self.hparams.n_val,
            "val_asset5":self.hparams.n_val,
            "val_asset6": self.hparams.n_val,
            "val_asset7": self.hparams.n_val,
            "val_asset8": self.hparams.n_val,

            "test": self.hparams.n_test,
        }
        self.n_obs = {k: v if v >= 0 else None for k, v in n_observations_per_split.items()}

        self.target_lens = {
            "train": self.hparams.max_target_length,
            "val_turk3": self.hparams.val_max_target_length,
            "val_turk5":self.hparams.val_max_target_length,
            "val_turk6": self.hparams.val_max_target_length,
            "val_turk7": self.hparams.val_max_target_length,
            "val_turk8": self.hparams.val_max_target_length,

            "val_asset3": self.hparams.val_max_target_length,
            "val_asset5":self.hparams.val_max_target_length,
            "val_asset6": self.hparams.val_max_target_length,
            "val_asset7": self.hparams.val_max_target_length,
            "val_asset8": self.hparams.val_max_target_length,

            "test": self.hparams.test_max_target_length,
        }
        # assert self.target_lens["train"] <= self.target_lens["val"], f"target_lens: {self.target_lens}"
        # assert self.target_lens["train"] <= self.target_lens["test"], f"target_lens: {self.target_lens}"
        # if self.hparams.freeze_embeds:
        #     self.freeze_embeds()

        freeze_params(self.seq2seq_model)
        assert_all_frozen(self.seq2seq_model)
        print('FREEZING ENTIRE seq2seq model.')
        # if self.hparams.freeze_encoder:
        #     freeze_params(self.model.get_encoder())
        #     assert_all_frozen(self.model.get_encoder())



        # if self.hparams.git:
        #     self.hparams.git_sha = get_git_info()["repo_sha"]
        self.num_workers = hparams.num_workers
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

        self.eval_min_length = self.hparams.eval_min_length
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
        if isinstance(self.model, T5ForConditionalGeneration):
            decoder_input_ids = self.model._shift_right(tgt_ids)
        else:
            decoder_input_ids = shift_tokens_right(tgt_ids, pad_token_id)
        conditional_info = {'lev': batch['levs'], 'dep': batch['deps'], 'leng': batch['lengs'], 'word': batch['words']}
        outputs = self(src_ids, attention_mask=src_mask, decoder_input_ids=decoder_input_ids, use_cache=False,
                       use_prefix=True,conditional_info=conditional_info)

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

        # print('hi', loss_tensors[0].item())
        self.training_acc_across_batches_at_curr_epoch.append(loss_tensors[0].item())
        self.log_dict(logs)
        loss = loss_tensors[0]
        return {"loss": loss}

    def on_epoch_end(self):
        train_acc_mean = np.mean(self.training_acc_across_batches_at_curr_epoch)
        self.log_dict({'train_loss': train_acc_mean})
        print('train_loss = {}'.format(train_acc_mean))
        # print('train_PPL = {}'.format(train_acc_mean.exp()))
        self.training_acc_across_batches_per_epoch = []  # reset for next epoch

    def validation_step(self, batch, batch_idx,dataloader_idx) -> Dict:
        # if self.current_epoch < 1:
        #     return 1

        return self._generative_step(batch, batch_idx, dataloader_idx)

    def validation_epoch_end(self, outputs, prefix="val") -> Dict:
        # if self.current_epoch < 1:
        #     logg = (0.1)
        #     self.log('val_sari', logg)
        #     return 1
        val_outputs_folder = "val_outputs"
        os.system("mkdir -p " + os.path.join(self.hparams.output_dir, val_outputs_folder))
        self.step_count += 1


        for output in outputs:
            dataset_idx = output[0]['dataloader_idx']

            if dataset_idx == 0:
                dataset_name = 'asset3'
            elif dataset_idx == 1:
                dataset_name = 'asset5'
            elif dataset_idx == 2:
                dataset_name = 'asset6'
            elif dataset_idx == 3:
                dataset_name = 'asset7'
            elif dataset_idx == 4:
                dataset_name = 'asset8'
            elif dataset_idx == 5:
                dataset_name = 'turk3'
            elif dataset_idx == 6:
                dataset_name = 'turk5'
            elif dataset_idx == 7:
                dataset_name = 'turk6'
            elif dataset_idx == 8:
                dataset_name = 'turk7'
            elif dataset_idx == 9:
                dataset_name = 'turk8'

            output_test_predictions_file = os.path.join(self.hparams.output_dir, val_outputs_folder,
                                                        dataset_name + "step"+
                                                        str(self.step_count) + ".txt")

            # write predictions and targets for later rouge evaluation.
            with open(output_test_predictions_file, "w") as p_writer:
                for output_batch in output:
                    p_writer.writelines(s + "\n" for s in output_batch["preds"])
                p_writer.close()


            if dataset_idx == 0:
                losses = {k: torch.stack([x[k] for x in output]).mean() for k in self.loss_names}
                loss = losses["loss"]

                generative_metrics = {
                    k: np.array([x[k] for x in output]).mean() for k in ["gen_time", "gen_len"]
                }
                cmd = f"""easse evaluate -t asset_test -m 'bleu,sari,fkgl' -q < {output_test_predictions_file}"""
                output = subprocess.check_output(cmd, shell = True).decode("utf-8")
                print(output)
                sari3 = float(re.search("'sari': ([^,]+)", output).group(1))

                metric_val = (
                    sari3
                )
                print('SARI_asset3', sari3)
                print('VAL_LOSS',loss)

            elif dataset_idx == 1:
                cmd = f"""easse evaluate -t asset_test -m 'bleu,sari,fkgl' -q < {output_test_predictions_file}"""
                output = subprocess.check_output(cmd, shell = True).decode("utf-8")
                print(output)
                sari5 = float(re.search("'sari': ([^,]+)", output).group(1))
                print('sari_asset5',sari5)

            elif dataset_idx == 2:
                cmd = f"""easse evaluate -t asset_test -m 'bleu,sari,fkgl' -q < {output_test_predictions_file}"""
                output = subprocess.check_output(cmd, shell = True).decode("utf-8")
                print(output)
                sari6 = float(re.search("'sari': ([^,]+)", output).group(1))
                print('sari_asset6', sari6)
            elif dataset_idx == 3:
                cmd = f"""easse evaluate -t asset_test -m 'bleu,sari,fkgl' -q < {output_test_predictions_file}"""
                output = subprocess.check_output(cmd, shell = True).decode("utf-8")
                print(output)
                sari7 = float(re.search("'sari': ([^,]+)", output).group(1))
                print('sari_asset7', sari7)
            elif dataset_idx == 4:
                cmd = f"""easse evaluate -t asset_test -m 'bleu,sari,fkgl' -q < {output_test_predictions_file}"""
                output = subprocess.check_output(cmd, shell = True).decode("utf-8")
                print(output)
                sari8 = float(re.search("'sari': ([^,]+)", output).group(1))
                print('sari_asset8', sari8)
            elif dataset_idx == 5:
                cmd = f"""easse evaluate -t turkcorpus_test -m 'bleu,sari,fkgl' -q < {output_test_predictions_file}"""
                output = subprocess.check_output(cmd, shell = True).decode("utf-8")
                sari_turk3 = float(re.search("'sari': ([^,]+)", output).group(1))
                print(output)
                print('SARI_TURK3',sari_turk3)
            elif dataset_idx == 6:
                cmd = f"""easse evaluate -t turkcorpus_test -m 'bleu,sari,fkgl' -q < {output_test_predictions_file}"""
                output = subprocess.check_output(cmd, shell = True).decode("utf-8")
                sari_turk5 = float(re.search("'sari': ([^,]+)", output).group(1))
                print(output)
                print('SARI_TURK5', sari_turk5)
            elif dataset_idx == 7:
                cmd = f"""easse evaluate -t turkcorpus_test -m 'bleu,sari,fkgl' -q < {output_test_predictions_file}"""
                output = subprocess.check_output(cmd, shell = True).decode("utf-8")
                sari_turk6 = float(re.search("'sari': ([^,]+)", output).group(1))
                print(output)
                print('SARI_TURK6', sari_turk6)
            elif dataset_idx == 8:
                cmd = f"""easse evaluate -t turkcorpus_test -m 'bleu,sari,fkgl' -q < {output_test_predictions_file}"""
                output = subprocess.check_output(cmd, shell = True).decode("utf-8")
                sari_turk7 = float(re.search("'sari': ([^,]+)", output).group(1))
                print(output)
                print('SARI_TURK7', sari_turk7)
            elif dataset_idx == 9:
                cmd = f"""easse evaluate -t turkcorpus_test -m 'bleu,sari,fkgl' -q < {output_test_predictions_file}"""
                output = subprocess.check_output(cmd, shell = True).decode("utf-8")
                sari_turk8 = float(re.search("'sari': ([^,]+)", output).group(1))
                print(output)
                print('SARI_TURK7', sari_turk8)

        self.log('VAL_LOSS',loss)
        self.log('SARI_ASSET3', sari3)
        self.log('SARI_ASSET5', sari5)
        self.log('SARI_ASSET6', sari6)
        self.log('SARI_ASSET7', sari7)
        self.log('SARI_ASSET8', sari8)

        self.log('SARI_TURK3', sari_turk3)
        self.log('SARI_TURK5', sari_turk5)
        self.log('SARI_TURK6', sari_turk6)
        self.log('SARI_TURK7', sari_turk7)
        self.log('SARI_TURK8', sari_turk8)

        metric_tensor: torch.FloatTensor = torch.tensor(metric_val).type_as(loss)
        generative_metrics.update({k: v.item() for k, v in losses.items()})
        losses.update(generative_metrics)
        all_metrics = {f"{prefix}_avg_{k}": x for k, x in losses.items()}
        all_metrics["step_count"] = self.step_count
        self.metrics[prefix].append(all_metrics)
        os.system(f'rm {self.val_output}')
        if prefix == 'val':
            self.log_dict({
                "log": all_metrics,
                f"{prefix}_loss": loss,
                f"{prefix}_{self.val_metric}": metric_tensor,
                "sari_turk": sari_turk3,
            })
        # preds = flatten_list([x["preds"] for x in output])
        return {
            "log": all_metrics,
            f"{prefix}_loss": loss,
            f"{prefix}_{self.val_metric}": metric_tensor,
            "sari_turk": sari_turk3
        }

    def calc_generative_metrics(self, preds, target) -> Dict:
        return calculate_rouge(preds, target)
        # return calculate_bleu(preds, target)

    def _generative_step(self, batch: dict,batch_idx=None, dataloader_idx=None) -> dict:
        t0 = time.time()

        bsz = batch["input_ids"].size(0)
        conditional_info = {'lev': batch['levs'], 'dep': batch['deps'], 'leng': batch['lengs'], 'word': batch['words']}
        prefix_prompt = self.model.get_prompt(bsz=bsz, sample_size=self.eval_beams, conditional_info=conditional_info)
        # print(prefix_prompt)
        generated_ids = self.seq2seq_model.generate(
            batch["input_ids"],
            past_key_values=prefix_prompt,
            attention_mask=batch["attention_mask"],
            use_cache=True,
            length_penalty=self.hparams.length_penalty,
            use_prefix=True,
            decoder_start_token_id=self.decoder_start_token_id,
            num_beams=self.eval_beams,
            min_length=self.eval_min_length,
            max_length=self.eval_max_length,
            no_repeat_ngram_size = 3
        )
        gen_time = (time.time() - t0) / batch["input_ids"].shape[0]
        preds: List[str] = self.ids_to_clean_text(generated_ids)
        loss_tensors = self._step(batch)
        base_metrics = {name: loss for name, loss in zip(self.loss_names, loss_tensors)}


        summ_len = np.mean(lmap(len, generated_ids))
        base_metrics.update(gen_time=gen_time, gen_len=summ_len, preds=preds)

        if dataloader_idx is not None:
            base_metrics.update(batch_idx = batch_idx, dataloader_idx = dataloader_idx)

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
            type_path=type_path,
            n_obs=n_obs,
            max_target_length=max_target_length,
            **self.dataset_kwargs,
        )
        return dataset

    def get_dataloader(self, type_path: str, batch_size: int, shuffle: bool = False) -> DataLoader:
        dataset = self.get_dataset(type_path)

        if self.hparams.sortish_sampler and type_path != "test":
            sampler = dataset.make_sortish_sampler(batch_size, distributed=self.hparams.gpus > 1)
            return DataLoader(
                dataset,
                batch_size=batch_size,
                collate_fn=dataset.collate_fn,
                shuffle=False,
                num_workers=self.num_workers,
                sampler=sampler,
            )

        elif self.hparams.max_tokens_per_batch is not None and type_path != "test":
            batch_sampler = dataset.make_dynamic_sampler(
                self.hparams.max_tokens_per_batch, distributed=self.hparams.gpus > 1
            )
            return DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                collate_fn=dataset.collate_fn,
                # shuffle=False,
                num_workers=self.num_workers,
                # batch_size=None,
            )
        else:
            return DataLoader(
                dataset,
                batch_size=batch_size,
                collate_fn=dataset.collate_fn,
                shuffle=shuffle,
                num_workers=self.num_workers,
                sampler=None,
            )

    def train_dataloader(self) -> DataLoader:
        dataloader = self.get_dataloader("train", batch_size=self.hparams.train_batch_size, shuffle=True)
        return dataloader

    def val_dataloader(self) -> DataLoader:
        asset3_dataloader = self.get_dataloader("val_asset3", batch_size = self.hparams.eval_batch_size,shuffle=False)
        asset5_dataloader = self.get_dataloader("val_asset5", batch_size = self.hparams.eval_batch_size, shuffle = False)
        asset6_dataloader = self.get_dataloader("val_asset6", batch_size = self.hparams.eval_batch_size,shuffle = False)
        asset7_dataloader = self.get_dataloader("val_asset6", batch_size = self.hparams.eval_batch_size,shuffle = False)
        asset8_dataloader = self.get_dataloader("val_asset8", batch_size = self.hparams.eval_batch_size,shuffle = False)

        turk3_dataloader = self.get_dataloader("val_turk3", batch_size = self.hparams.eval_batch_size,shuffle = False)
        turk5_dataloader = self.get_dataloader("val_turk5", batch_size = self.hparams.eval_batch_size, shuffle = False)
        turk6_dataloader = self.get_dataloader("val_turk6", batch_size = self.hparams.eval_batch_size,shuffle = False)
        turk7_dataloader = self.get_dataloader("val_turk7", batch_size = self.hparams.eval_batch_size,shuffle = False)
        turk8_dataloader = self.get_dataloader("val_turk8", batch_size = self.hparams.eval_batch_size,shuffle = False)

        return [asset3_dataloader,asset5_dataloader,asset6_dataloader,asset7_dataloader,asset8_dataloader,turk3_dataloader,turk5_dataloader,turk6_dataloader,turk7_dataloader,turk8_dataloader]
    def test_dataloader(self) -> DataLoader:

        return self.get_dataloader("test", batch_size=self.hparams.eval_batch_size,shuffle=False)

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
            "--val_output",
            default = "",
            type = str,
            help = "The maximum total input sequence length after tokenization. Sequences longer "
                   "than this will be truncated, sequences shorter will be padded.",
        )
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
        parser.add_argument("--n_val", type=int, default=500, required=False, help="# examples. -1 means use all.")
        parser.add_argument("--n_test", type=int, default=-1, required=False, help="# examples. -1 means use all.")
        parser.add_argument(
            "--task_mode", type=str, default="summarization", required=False, help="# examples. -1 means use all."
        )
        parser.add_argument("--label_smoothing", type=float, default=0.0, required=False)
        parser.add_argument("--src_lang", type=str, default="", required=False)
        parser.add_argument("--tgt_lang", type=str, default="", required=False)
        parser.add_argument("--eval_beams", type=int, default=6, required=False)
        parser.add_argument("--eval_min_length", type=int, default=10, required=False)

        parser.add_argument(
            "--val_metric", type=str, default=None, required=False, choices=["bleu", "rouge2", "loss", None]
        )
        parser.add_argument("--eval_max_gen_length", type=int, default=60, help="never generate more than n tokens")
        parser.add_argument("--length_penalty", type=float, default=1.0, help="never generate more than n tokens")
        parser.add_argument("--save_top_k", type=int, default=1, required=False, help="How many checkpoints to save")
        parser.add_argument("--wb_project", type = str,default="")
        parser.add_argument("--git", type =bool,default=True)
        parser.add_argument("--dev", type =bool,default=False)


        parser.add_argument("--wb_name", type = str,default="")
        parser.add_argument("--id", type = str, default = "")
        parser.add_argument(
            "--early_stopping_patience",
            type=int,
            default=-1,
            required=False,
            help="-1 means never early stop. early_stopping_patience is measured in validation checks, not epochs. So val_check_interval will effect it.",
        )
        return parser


def eval(args, model=None):
    if model is None:
        if "summarization" in args.task_mode:
            if args.tuning_mode == 'prefixtune':
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


    if model is None:
        if "summarization" in args.task_mode:
            if args.tuning_mode == 'prefixtune':
                model = PrefixSummarizationModule(args)
    pickle_save(args,  os.path.join(args.output_dir, "args.pkl"))
    dataset = Path(args.data_dir).name

    # batch = pickle_load('/Users/jordi/Desktop/Master/prefix_tuning/transformers/GEC/b1.pkl')
    # model._generative_step(batch)
    # print(dataset)
    pickle_save(model.hparams, model.output_dir / "hparams.pkl")

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
        early_stopping_callback = es_callback,
        logger = logger,
    )
    pickle_save(model.hparams, model.output_dir / "hparams.pkl")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = PrefixSummarizationModule.add_model_specific_args(parser, os.getcwd())

    args = parser.parse_args()
    pl.seed_everything(args.seed)

    model = main(args)
    args.do_train = False
    eval(args,model)