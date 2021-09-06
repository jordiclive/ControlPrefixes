import torch
from transformers import PretrainedBartModel
from torch import nn




class PrefixTuning(PretrainedBartModel):
    """Classification Head for  transformer encoders"""
    def __init__(self, config, preseqlen=5):
        super().__init__(config)
        print('under the PrefixTuning model')

        self.match_n_layer = config.decoder_layers  # 6
        self.match_n_head = config.decoder_attention_heads  # 12
        self.n_embd = config.d_model  # 768
        self.match_n_embd = self.n_embd // self.match_n_head  # 64
        self.init_random = False
        self.CEFR_single_block = True

        if hasattr(config, 'preseqlen'):
            self.preseqlen = config.preseqlen
        elif self.optim_prefix:
            self.preseqlen = preseqlen

        if hasattr(config, '_my_arg_task_mode'):
            self.task_mode = config._my_arg_task_mode  # summarization
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
        self.use_encoder_prefix = True
        self.use_cross_prefix = True

        self.format_mode = 'cat'

        total_param = 0
        for name, param in self.named_parameters():
            # print(param.shape)
            total_param += param.numel()
        print('Base Total Param is {}'.format(total_param))
        self.CEFR = 3
        if hasattr(config, 'cefr_length'):
            self.cefr_length = config.cefr_length
        else:
            self.cefr_length = 5
        if hasattr(config, 'cefr_mid_dim'):
            self.cefr_mid_dim = config.cefr_mid_dim
        else:
            self.cefr_mid_dim = 256
        print('CEFR_length', self.cefr_length)
        print('CEFR_mid_dim', self.cefr_mid_dim)



        if self.cefr_length > 0:
            self.get_prompt = self.get_prompt_multiple_prefix

        self.input_tokens = torch.arange(self.preseqlen).long()
        self.wte = nn.Embedding(self.preseqlen, self.n_embd)
        self.control_trans = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim), # 1024 x 800
            nn.Tanh(), #800      x      12 * 2 * 1024
            nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd))


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
            #print(param.shape)
            total_param += param.numel()
        print('Base Total Param is {}'.format(total_param))



        if hasattr(config, 'm_prefix_mid_dim'):
            self.m_prefix_mid_dim = config.m_prefix_mid_dim
        else:
            self.m_prefix_mid_dim = 512

        self.categories = ['cefr']
        self.new_token_len = [3]

        # self.CEFR = False
        if self.new_token_len:
            self.M_Prefixes = torch.nn.ModuleDict()

        self.m_prefix_len = self.cefr_length
        if hasattr(config, 'same__intialization'):
            self.same_initialization = config.same_intialization
        else:
            self.same_initialization = False


        for i,j in enumerate(self.categories):
            self.M_Prefixes[f'wte_{j}'] = nn.Embedding(self.m_prefix_len * self.new_token_len[i], self.n_embd)
            self.M_Prefixes[f'wte_enc_{j}'] = nn.Embedding(self.m_prefix_len * self.new_token_len[i], self.n_embd)
            self.M_Prefixes[f'wte_2_{j}'] = nn.Embedding(self.m_prefix_len * self.new_token_len[i], self.n_embd)



        total_param = 0
        for name, param in self.named_parameters():

            print(name,param.shape)
            total_param += param.numel()
        print('total param is {}'.format(total_param))


    def initialize_same(self,embedding):
        with torch.no_grad():
            for i in range(self.new_token_len - 1):
                embedding.weight[(i+1)*self.m_prefix_len:(i + 2) * self.m_prefix_len] = embedding.weight[0:self.m_prefix_len]


    def get_encoder_output(self, gpt2, temp_input):
        return gpt2.model.encoder.forward_with_encoder_past(temp_input).past_key_values

    def get_prompt_multiple_prefix(self, CEFR,bsz=None, sample_size=1):
        conditional_info = {'cefr':CEFR}

        input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1).to(
            self.device)
        temp_control = self.wte(
            input_tokens)   matrix
        past_key_values = self.control_trans(
            temp_control)


        temp_control2 = self.wte2(input_tokens)
        past_key_values2 = self.control_trans2(temp_control2)

        temp_control_enc = self.wte_enc(input_tokens)
        past_key_values_enc = self.control_trans_enc(temp_control_enc)


        for category_idx,category in enumerate(self.categories):
            self.seed_multiple = torch.arange(self.m_prefix_len * self.new_token_len[category_idx]).long().unsqueeze(0).expand(bsz, -1).to(
                self.device)


            temp_control = self.M_Prefixes[f'wte_{category}'](
                self.seed_multiple)

            past_key_values_multiple = self.control_trans(temp_control)


            idxmap = {i:((i)*self.m_prefix_len,((i + 1) * self.m_prefix_len)) for i in range(self.new_token_len[category_idx])}
            cond = list(map(idxmap.get, conditional_info[category].tolist()))

            past_key_values_multiple = torch.stack([past_key_values_multiple[i, j[0]:j[1], :] for i,j in enumerate(cond)])
            past_key_values = torch.cat([past_key_values_multiple, past_key_values], dim = 1)



            temp_control_2 = self.M_Prefixes[f'wte_2_{category}'](
                        self.seed_multiple)
            past_key_values_multiple_2 = self.control_trans2(temp_control_2)

            past_key_values_multiple_2 = torch.stack(
                        [past_key_values_multiple_2[i, j[0]:j[1], :] for i, j in enumerate(cond)])

            past_key_values2 = torch.cat([past_key_values_multiple_2, past_key_values2], dim = 1)

            temp_control_3 = self.M_Prefixes[f'wte_enc_{category}'](
                self.seed_multiple)
            past_key_values_multiple_3 = self.control_trans_enc(temp_control_3)
            past_key_values_multiple_3 = torch.stack(
                [past_key_values_multiple_3[i, j[0]:j[1], :] for i, j in enumerate(cond)])
            past_key_values_enc = torch.cat([past_key_values_multiple_3, past_key_values_enc], dim = 1)



        if sample_size > 1:
            past_key_values = torch.cat(sample_size*[past_key_values])

        bsz, seqlen, _ = past_key_values.shape

        past_key_values = past_key_values.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                               self.match_n_embd)  # 16, 200, 12, 12, 64 (12*64 = 768, for bart base)
        past_key_values = self.dropout(past_key_values)  # no dropout
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(
            2)

        if sample_size > 1:
            past_key_values2 = torch.cat(sample_size * [past_key_values2])

        bsz, seqlen, _ = past_key_values2.shape
        past_key_values2 = past_key_values2.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                                 self.match_n_embd)
        past_key_values2 = self.dropout(past_key_values2)
        past_key_values2 = past_key_values2.permute([2, 0, 3, 1, 4]).split(2)


        bsz_enc, seqlen, _ = past_key_values_enc.shape
        past_key_values_enc = past_key_values_enc.view(bsz_enc, seqlen, self.match_n_layer * 2, self.match_n_head,
                                                       self.match_n_embd)
        past_key_values_enc = self.dropout(past_key_values_enc)
        past_key_values_enc = past_key_values_enc.permute([2, 0, 3, 1, 4]).split(2)
        result = []
        for i, key_val in enumerate(past_key_values):
            temp_dict = {'self': {"prev_key": key_val[0].contiguous(),
                                  "prev_value": key_val[1].contiguous(),
                                  "prev_key_padding_mask": torch.zeros(bsz, seqlen).to(key_val.device).bool()
                                  },
                         }
            if self.use_cross_prefix:
                key_val2 = past_key_values2[i]
                temp_dict['encoder_decoder'] = {"prev_key": key_val2[0].contiguous(),
                                                "prev_value": key_val2[1].contiguous(),
                                                "prev_key_padding_mask": torch.zeros(bsz, seqlen).to(
                                                    key_val2.device).bool()
                                                }
            if self.use_encoder_prefix:
                key_val_enc = past_key_values_enc[i]
                temp_dict['encoder'] = {"prev_key": key_val_enc[0].contiguous(),
                                        "prev_value": key_val_enc[1].contiguous(),
                                        "prev_key_padding_mask": torch.zeros(bsz_enc, seqlen).to(
                                            key_val_enc.device).bool()
                                        }
            result.append(temp_dict)

        return result




    def forward(self,
                input_ids=None,
                frozen_model=None,
                past_key_values=None,
                CEFR = None,
                **kwargs,
                ):


        bsz = input_ids.shape[0]


        past_key_values_prompt = self.get_prompt_multiple_prefix(CEFR,bsz=bsz )


        if past_key_values is not None:
            assert False, "Past key values"
        else:
            past_key_values = past_key_values_prompt

        if frozen_model is None:
            assert False, "Didn't specify frozen model"


        output = frozen_model(input_ids=input_ids,
                              past_key_values=past_key_values, **kwargs)


        return output


