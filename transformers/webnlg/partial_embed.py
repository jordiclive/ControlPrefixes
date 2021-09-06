import torch
import torch.nn as nn

class PartiallyFixedEmbedding(torch.nn.Module):
    def __init__(self, fixed_weights, num_to_learn,padding_idx=1):
        super().__init__()
        self.num_fixed = fixed_weights.size(0)
        self.num_to_learn = num_to_learn
        weight = torch.empty(self.num_fixed+num_to_learn, fixed_weights.size(1))
        weight[:self.num_fixed] = fixed_weights
        self.trainable_weight = torch.nn.Parameter(torch.empty(num_to_learn, fixed_weights.size(1)))
        torch.nn.init.kaiming_uniform_(self.trainable_weight)
        weight[self.num_fixed:] = self.trainable_weight
        self.register_buffer('weight', weight)
        self.padding_idx = padding_idx


    def forward(self, inp):
        self.weight.detach_()
        self.weight[self.num_fixed:] = self.trainable_weight
        return torch.nn.functional.embedding(
            inp, self.weight, self.padding_idx, None, 2.0, False, False)



def make_new_embeddings_learnable(model,tokenizer_len,num_to_learn):
    print('fixed_embeds',tokenizer_len-num_to_learn)
    # fixed_weights = model.shared.weight[:tokenizer_len-num_to_learn]
    fixed_weights = model.shared.weight[:32100]
    new_embed_layer = PartiallyFixedEmbedding(fixed_weights,num_to_learn)
    model.decoder.embed_tokens = new_embed_layer
    model.encoder.embed_tokens = new_embed_layer
    model.shared = new_embed_layer
    # model.model.encoder.embed_tokens = new_embed_layer
    # model.model.decoder.embed_tokens = new_embed_layer
    # model.model.shared = new_embed_layer

    # model.tie_weights()

# def create_tokens():
#     source_cats = ['e2e', 'webnlg', 'WikiTableQuestions_lily', 'WikiSQL_decl_sents', 'WikiTableQuestions_mturk', 'WikiSQL_lily']
#     all_tokens = []
#     for i in source_cats:
#         all_tokens.append('<'+i+'>')
#     return all_tokens



