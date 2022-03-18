# from transformers import Trainer
import torch
from partial_embed import PartiallyFixedEmbedding
from torch import nn
from transformers import PretrainedBartModel


class ControlPrefixes(PretrainedBartModel):
    """Control Prefixes model"""

    def __init__(self, config, preseqlen=5):
        super().__init__(config)
        print("under the PrefixTuning model")

        self.match_n_layer = config.num_layers  # 6
        self.match_n_head = config.num_attention_heads  # 12
        self.n_embd = config.d_model  # 768 512
        self.match_n_embd = self.n_embd // self.match_n_head  # 64

        if hasattr(config, "new_token_len"):
            self.new_token_len = config.new_token_len
        else:
            self.new_token_len = 3

        self.es = PartiallyFixedEmbedding(torch.rand(1, 1024), self.new_token_len)

        if hasattr(config, "preseqlen"):
            self.preseqlen = config.preseqlen
        elif self.optim_prefix:
            self.preseqlen = preseqlen

        if hasattr(config, "_my_arg_task_mode"):
            self.task_mode = config._my_arg_task_mode  # GEC
        else:
            self.task_mode = "underspecified"
            assert False, "the task is underspecified"

        self.format_mode = "cat"

        if hasattr(config, "prefix_dropout"):
            self.prefix_dropout = config.prefix_dropout
            self.dropout = nn.Dropout(self.prefix_dropout)
        else:
            self.prefix_dropout = 0.0

        if hasattr(config, "mid_dim"):
            self.mid_dim = config.mid_dim
        else:
            self.mid_dim = 800
        self.use_encoder_prefix = True
        self.use_cross_prefix = True

        if hasattr(config, "m_prefix_len"):
            self.m_prefix_len = config.m_prefix_len
            print("M_Prefix_LEN")
        else:
            self.m_prefix_len = 0

        if hasattr(config, "DART"):
            self.DART = config.DART

        if self.m_prefix_len > 0:
            self.get_prompt = self.get_prompt_multiple_prefix

        self.input_tokens = torch.arange(self.preseqlen).long()
        self.wte = nn.Embedding(self.preseqlen, self.n_embd)
        self.control_trans = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),  # 1024 x 800
            nn.Tanh(),  # 800      x      12 * 2 * 1024
            nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd),
        )

        if self.use_encoder_prefix:
            self.wte_enc = nn.Embedding(self.preseqlen, self.n_embd)
            self.control_trans_enc = nn.Sequential(
                nn.Linear(self.n_embd, self.mid_dim),
                nn.Tanh(),
                nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd),
            )

        if self.use_cross_prefix:
            self.wte2 = nn.Embedding(self.preseqlen, self.n_embd)
            self.control_trans2 = nn.Sequential(
                nn.Linear(self.n_embd, self.mid_dim),
                nn.Tanh(),
                nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd),
            )

        total_param = 0
        for name, param in self.named_parameters():
            # print(param.shape)
            total_param += param.numel()
        print("Base Total Param is {}".format(total_param))

        if self.DART:
            self.categories = [
                "sources",
            ]  # specify the control prefixes for each category

            self.new_token_len = [6]
        else:

            self.categories = [
                "sources",
                "cats",
            ]  # specify the control prefixes for each category

            self.new_token_len = [
                6,
                17,
            ]  #  specify the number of labels in each category

        if self.new_token_len:
            self.M_Prefixes = torch.nn.ModuleDict()

        if hasattr(config, "unseen"):
            self.unseen = config.unseen
            print("M_Prefix_LEN")
        else:
            self.unseen = None

        for i, j in enumerate(self.categories):
            self.M_Prefixes[f"wte_{j}"] = nn.Embedding(
                self.m_prefix_len * self.new_token_len[i], self.n_embd
            )
            self.M_Prefixes[f"wte_enc_{j}"] = nn.Embedding(
                self.m_prefix_len * self.new_token_len[i], self.n_embd
            )
            self.M_Prefixes[f"wte_2_{j}"] = nn.Embedding(
                self.m_prefix_len * self.new_token_len[i], self.n_embd
            )
        if self.unseen is True:
            print("MAKING UNSEEN ZERO")
            self.M_Prefixes[f"wte_cats"].weight[-2:] = torch.zeros(
                self.M_Prefixes[f"wte_cats"].weight[-2:].shape
            ).to(self.device)
            self.M_Prefixes[f"wte_enc_cats"].weight[-2:] = torch.zeros(
                self.M_Prefixes[f"wte_cats"].weight[-2:].shape
            ).to(self.device)
            self.M_Prefixes[f"wte_2_cats"].weight[-2:] = torch.zeros(
                self.M_Prefixes[f"wte_cats"].weight[-2:].shape
            ).to(self.device)

        total_param = 0
        for name, param in self.named_parameters():
            print(name, param.shape)
            total_param += param.numel()
        print("total param is {}".format(total_param))

    def get_encoder_output(self, model, temp_input):
        return model.model.encoder.forward_with_encoder_past(temp_input).past_key_values

    def get_prompt_multiple_prefix(self, conditional_info, bsz=None, sample_size=1):

        input_tokens = (
            self.input_tokens.unsqueeze(0).expand(bsz, -1).to(self.device)
        )  # 8 x 200 rows of tensor([[  0,   1,   2,  ..., 197, 198, 199],
        temp_control = self.wte(
            input_tokens
        )  # self.wte = Embedding(200, 768) ,  # temp_control is 8, 200, 1024. so 8 repeats of the embedding matrix
        past_key_values = self.control_trans(temp_control)  # 8 x preseqlen x 24576

        temp_control2 = self.wte2(input_tokens)
        past_key_values2 = self.control_trans2(temp_control2)

        temp_control_enc = self.wte_enc(input_tokens)
        past_key_values_enc = self.control_trans_enc(temp_control_enc)

        for category_idx, category in enumerate(self.categories):
            self.seed_multiple = (
                torch.arange(self.m_prefix_len * self.new_token_len[category_idx])
                .long()
                .unsqueeze(0)
                .expand(bsz, -1)
                .to(self.device)
            )

            temp_control = self.M_Prefixes[f"wte_{category}"](
                self.seed_multiple
            )  # self.wte = Embedding(200, 768) ,  # temp_control is 8, 200, 1024. so 8 repeats of the embedding matrix

            past_key_values_multiple = self.control_trans(
                temp_control
            )  # bsz 3*seqlen ,

            idxmap = {
                i: ((i) * self.m_prefix_len, ((i + 1) * self.m_prefix_len))
                for i in range(self.new_token_len[category_idx])
            }
            cond = list(map(idxmap.get, conditional_info[category].tolist()))

            past_key_values_multiple = torch.stack(
                [
                    past_key_values_multiple[i, j[0] : j[1], :]
                    for i, j in enumerate(cond)
                ]
            )
            past_key_values = torch.cat(
                [past_key_values_multiple, past_key_values], dim=1
            )

            temp_control_2 = self.M_Prefixes[f"wte_2_{category}"](
                self.seed_multiple
            )  # self.wte = Embedding(200, 768) ,  # temp_control is 8, 200, 1024. so 8 repeats of the embedding matrix

            past_key_values_multiple_2 = self.control_trans2(temp_control_2)

            past_key_values_multiple_2 = torch.stack(
                [
                    past_key_values_multiple_2[i, j[0] : j[1], :]
                    for i, j in enumerate(cond)
                ]
            )

            past_key_values2 = torch.cat(
                [past_key_values_multiple_2, past_key_values2], dim=1
            )

            temp_control_3 = self.M_Prefixes[f"wte_enc_{category}"](
                self.seed_multiple
            )  # self.wte = Embedding(200, 768) ,  # temp_control is 8, 200, 1024. so 8 repeats of the embedding matrix
            past_key_values_multiple_3 = self.control_trans_enc(temp_control_3)
            past_key_values_multiple_3 = torch.stack(
                [
                    past_key_values_multiple_3[i, j[0] : j[1], :]
                    for i, j in enumerate(cond)
                ]
            )
            past_key_values_enc = torch.cat(
                [past_key_values_multiple_3, past_key_values_enc], dim=1
            )

        if sample_size > 1:
            past_key_values = torch.cat(sample_size * [past_key_values])

        bsz, seqlen, _ = past_key_values.shape

        past_key_values = past_key_values.view(
            bsz, seqlen, self.match_n_layer * 2, self.match_n_head, self.match_n_embd
        )  # 16, 200, 12, 12, 64 (12*64 = 768, for bart base)
        past_key_values = self.dropout(past_key_values)  # no dropout
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)

        if sample_size > 1:
            past_key_values2 = torch.cat(sample_size * [past_key_values2])

        past_key_values2 = past_key_values2.view(
            bsz, seqlen, self.match_n_layer * 2, self.match_n_head, self.match_n_embd
        )
        past_key_values2 = self.dropout(past_key_values2)
        past_key_values2 = past_key_values2.permute([2, 0, 3, 1, 4]).split(2)

        bsz_enc, seqlen, _ = past_key_values_enc.shape
        past_key_values_enc = past_key_values_enc.view(
            bsz_enc,
            seqlen,
            self.match_n_layer * 2,
            self.match_n_head,
            self.match_n_embd,
        )
        past_key_values_enc = self.dropout(past_key_values_enc)
        past_key_values_enc = past_key_values_enc.permute([2, 0, 3, 1, 4]).split(2)
        result = []
        for i, key_val in enumerate(past_key_values):
            temp_dict = {
                "self": {
                    "prev_key": key_val[0].contiguous(),
                    "prev_value": key_val[1].contiguous(),
                }
            }
            if self.use_cross_prefix:
                key_val2 = past_key_values2[i]
                temp_dict["encoder_decoder"] = {
                    "prev_key": key_val2[0].contiguous(),
                    "prev_value": key_val2[1].contiguous(),
                }
            if self.use_encoder_prefix:
                key_val_enc = past_key_values_enc[i]
                temp_dict["encoder"] = {
                    "prev_key": key_val_enc[0].contiguous(),
                    "prev_value": key_val_enc[1].contiguous(),
                }
            result.append(temp_dict)

        return result

    def forward(
        self,
        input_ids=None,
        frozen_model=None,
        past_key_values=None,
        conditional_info=None,
        **kwargs,
    ):

        # {"input_ids": batch, "labels": labels, 'src_attn': src_attn, 'tgt_attn':tgt_attn, 'src':src}

        bsz = input_ids.shape[0]

        past_key_values_prompt = self.get_prompt_multiple_prefix(
            conditional_info, bsz=bsz
        )

        if past_key_values is not None:
            assert False, "Past key values"
        else:
            past_key_values = past_key_values_prompt

        if frozen_model is None:
            assert False, "Didn't specify frozen model"

        output = frozen_model(
            input_ids=input_ids, past_key_values=past_key_values, **kwargs
        )

        return output
