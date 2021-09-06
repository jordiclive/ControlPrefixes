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



def make_new_embeddings_learnable(model,num_to_learn):
    fixed_weights = model.model.encoder.embed_tokens.weight[:50265]
    new_embed_layer = PartiallyFixedEmbedding(fixed_weights,num_to_learn)
    model.model.encoder.embed_tokens = new_embed_layer
    model.model.decoder.embed_tokens = new_embed_layer
    model.model.shared = new_embed_layer
    # model.tie_weights()

def create_tokens():
    all_tokens = []
    Lev = ['1.0', '0.95', '0.9', '0.85', '0.8', '0.75', '0.7', '0.65', '0.6', '0.55', '0.5', '0.45', '0.4', '0.35', '0.3', '0.25', '0.2', '0.0', '0.15', '0.1', '0.05']
    word =['1.0', '0.95', '1.05', '0.9', '1.1', '0.85', '1.15', '0.8', '1.2', '0.75', '1.25', '0.7', '1.3', '1.35', '0.65', '1.4', '1.45', '0.6', '1.5', '1.55', '0.55', '1.65', '1.6', '1.7', '0.5', '2.0', '1.8', '1.85', '1.9', '1.75', '0.45', '1.95', '0.4', '0.35']
    leng =['1.0', '0.95', '0.9', '1.05', '0.85', '0.8', '0.75', '0.7', '0.6', '0.65', '0.55', '0.5', '2.0', '0.05', '0.45', '0.4', '1.1', '0.35', '0.3', '0.1', '0.25', '0.15', '1.15', '0.2', '1.2', '1.25', '1.3', '1.35', '1.4', '1.45', '1.5', '0.0', '1.55', '1.6', '1.65', '1.7', '1.75', '1.8', '1.9', '1.85', '1.95']
    dep = ['1.0', '0.8', '0.85', '0.65', '0.5', '0.0', '0.6', '0.75', '2.0', '0.55', '0.7', '1.35', '1.2', '0.4', '1.25', '0.35', '0.45', '1.15', '0.9', '1.5', '1.1', '0.25', '0.2', '0.15', '1.4', '0.3', '1.65', '0.1', '1.6', '1.75', '1.3', '1.8', '1.45', '1.55', '1.85', '0.05', '0.95', '1.7', '1.05', '1.9']

    for i in Lev:
        all_tokens.append(f"<REPLACEONLYLEVENSHTEIN_{i}>")
    for i in word:
        all_tokens.append(f"<WORDRANKRATIO_{i}>")
    for i in leng:
        all_tokens.append(f"<LENGTHRATIO_{i}>")
    for i in dep:
        all_tokens.append(f"<DEPENDENCYTREEDEPTHRATIO_{i}>")
    return all_tokens

