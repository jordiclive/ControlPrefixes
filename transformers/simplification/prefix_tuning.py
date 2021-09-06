import torch
from transformers import PretrainedBartModel
from torch import nn

from partial_embed import PartiallyFixedEmbedding


class PrefixTuning(PretrainedBartModel):
    """Classification Head for  transformer encoders"""
    def __init__(self, config, preseqlen=5):
        super().__init__(config)
        print('under the PrefixTuning model')

        self.match_n_layer = config.decoder_layers  # 6
        self.match_n_head = config.decoder_attention_heads # 12
        self.n_embd = config.d_model # 768
        self.match_n_embd = self.n_embd // self.match_n_head # 64
        if hasattr(config, 'new_token_len'):
            self.es = PartiallyFixedEmbedding(torch.rand(1,1024),config.new_token_len)
        else:
            self.es = PartiallyFixedEmbedding(torch.rand(1,1024),3)

        self.init_random = False

        if hasattr(config, 'preseqlen'):
            self.preseqlen = config.preseqlen
        elif self.optim_prefix:
            self.preseqlen = preseqlen

        if hasattr(config, '_my_arg_task_mode'):
            self.task_mode = config._my_arg_task_mode # GEC
        else:
            self.task_mode = 'underspecified'
            assert False, 'the task is underspecified'

        self.format_mode = 'cat'

        if hasattr(config, 'prefix_dropout'):
            self.prefix_dropout = config.prefix_dropout
            self.dropout = nn.Dropout(self.prefix_dropout)
        else:
            self.prefix_dropout = 0.0

        if hasattr(config, 'mid_dim'):
            self.mid_dim = config.mid_dim
        else:
            self.mid_dim = 512

        self.input_tokens = torch.arange(self.preseqlen).long()
        self.wte = nn.Embedding(self.preseqlen, self.n_embd)
        self.control_trans = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd))


        self.use_encoder_prefix = True
        self.use_cross_prefix = True
        #todo change to GELU or something else
        if self.use_encoder_prefix:
            self.wte_enc = nn.Embedding(self.preseqlen, self.n_embd)
            self.control_trans_enc = nn.Sequential(
                nn.Linear(self.n_embd, self.mid_dim),
                nn.Tanh(),
                nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd))

        if self.use_cross_prefix:
            self.wte2 = nn.Embedding(self.preseqlen, self.n_embd)
            self.control_trans2 = nn.Sequential(
                nn.Linear(self.n_embd, self.mid_dim),
                nn.Tanh(),
                nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd))


        total_param = 0
        for name, param in self.named_parameters():
            print(param.shape)
            total_param += param.numel()
        print('total param is {}'.format(total_param))




    def get_encoder_output(self, gpt2, temp_input):
        return gpt2.model.encoder.forward_with_encoder_past(temp_input).past_key_values


    def get_prompt(self, control_code=None, gpt2=None, bsz=None, sample_size=1):
        old_bsz = bsz # 16
        bsz = bsz * sample_size  #sample size is 1
        input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1).to(self.device)  #16 x 200 rows of tensor([[  0,   1,   2,  ..., 197, 198, 199],
        temp_control = self.wte(input_tokens) # self.wte = Embedding(200, 768) , so randomized. # temp_control is 16, 200, 768
        past_key_values = self.control_trans(temp_control) #bsz, seqlen, 2*layer*emb  # Sequential( # past key values.shape is 16 x 200 x 9216 (

  # (0): Linear(in_features=768, out_features=800, bias=True)    #nn.Sequential(
        #                     nn.Linear(self.n_embd, self.mid_dim),
        #                     nn.Tanh(),
        #                     nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd))
  # (1): Tanh()
  # (2): Linear(in_features=800, out_features=9216, bias=True)
        bsz, seqlen, _ = past_key_values.shape # 16 x 200 # 16 x 200 x 9216 -> 16 x 200 * 12 * 768
        past_key_values = past_key_values.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                               self.match_n_embd) # 16, 200, 12, 12, 64 (12*64 = 768, for bart base)
        past_key_values = self.dropout(past_key_values) # no dropout
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2) # splitting into 6 lots of 2 # [2, 16, 12, 200, 64] query/key for each layer
        # for self attention layer


        if self.use_cross_prefix:
            temp_control2 = self.wte2(input_tokens)
            past_key_values2 = self.control_trans2(temp_control2)  # bsz, seqlen, layer*emb
            bsz, seqlen, _ = past_key_values2.shape
            past_key_values2 = past_key_values2.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                                   self.match_n_embd)
            past_key_values2 = self.dropout(past_key_values2)
            past_key_values2 = past_key_values2.permute([2, 0, 3, 1, 4]).split(2)
        # for cross attention layer

        if self.use_encoder_prefix:
            input_tokens_enc = self.input_tokens.unsqueeze(0).expand(old_bsz, -1).to(self.device)
            temp_control_enc = self.wte_enc(input_tokens_enc)
            past_key_values_enc = self.control_trans_enc(temp_control_enc)  # bsz, seqlen, layer*emb
            bsz_enc, seqlen, _ = past_key_values_enc.shape
            past_key_values_enc = past_key_values_enc.view(bsz_enc, seqlen, self.match_n_layer * 2, self.match_n_head,
                                                     self.match_n_embd)
            past_key_values_enc = self.dropout(past_key_values_enc)
            past_key_values_enc = past_key_values_enc.permute([2, 0, 3, 1, 4]).split(2)
        # for encoder layer
        # this seems really inefficient + bart bespoke code. # 16 x 200 of zeros
        result = []
        for i, key_val in enumerate(past_key_values):
            temp_dict = {'self': {"prev_key": key_val[0].contiguous(), # [16, 12, 200, 64] # for each head there are 12 heads. so result will 6 dictionaries for each layer
                                  "prev_value": key_val[1].contiguous(),
                                  "prev_key_padding_mask": torch.zeros(bsz, seqlen).to(key_val.device).bool() #bsz, preseqlen
                                 },
                        }
            if self.use_cross_prefix:
                key_val2 = past_key_values2[i]
                temp_dict['encoder_decoder'] = {"prev_key": key_val2[0].contiguous(),
                                                "prev_value": key_val2[1].contiguous(),
                                                "prev_key_padding_mask": torch.zeros(bsz, seqlen).to(key_val2.device).bool()
                                                }
            if self.use_encoder_prefix:
                key_val_enc = past_key_values_enc[i]
                temp_dict['encoder'] = {"prev_key": key_val_enc[0].contiguous(),
                                        "prev_value": key_val_enc[1].contiguous(),
                                        "prev_key_padding_mask": torch.zeros(bsz_enc, seqlen).to(key_val_enc.device).bool()
                                        }
            result.append(temp_dict)

        return result



    def forward(self,
                input_ids=None,
                frozen_model=None,
                past_key_values=None,
                # attention_mask=None,
                # token_type_ids=None,
                # position_ids=None,
                # head_mask=None,
                # inputs_embeds=None,
                # encoder_hidden_states=None,
                # encoder_attention_mask=None,
                # labels=None,
                # use_cache=None,
                # output_attentions=None,
                # output_hidden_states=None,
                # return_dict=None,
                src=None,
                tgt=None,
                src_attn=None,
                tgt_attn=None,
                **kwargs,
                ):

        #{"input_ids": batch, "labels": labels, 'src_attn': src_attn, 'tgt_attn':tgt_attn, 'src':src}

        bsz = input_ids.shape[0]

        past_key_values_prompt = self.get_prompt(bsz=bsz)

        if past_key_values is not None:
            assert False, "Past key values"
        else:
            past_key_values = past_key_values_prompt

        if frozen_model is None:
            assert False, "Didn't specify frozen model"


        output = frozen_model(input_ids=input_ids,
                              past_key_values=past_key_values, **kwargs)


        return output