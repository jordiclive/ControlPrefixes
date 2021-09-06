from typing import NamedTuple

import torch
import pytest
import pickle
from pathlib import Path

from src.config import ConveRTModelConfig, ConveRTTrainConfig
from src.model_components import (
    SubwordEmbedding,
    SelfAttention,
    circulant_mask,
    FeedForward1,
    SharedInnerBlock,
    MultiheadAttention,
    TransformerLayers,
    FeedForward2,
)

SEQ_LEN = 60
relative_attention = 48


@pytest.fixture
def model_config():
    return ConveRTModelConfig()


@pytest.fixture
def train_config():
    return ConveRTTrainConfig(train_batch_size = 64, split_size = 8, learning_rate = 2e-5)


def test_circulant_t():
    assert circulant_mask(50, 47).sum().item() == 2494
    try:
        circulant_mask(47, 50)
        circulant_mask(47, 47)
        circulant_mask(47, 45)
    except ExceptionType:
        self.fail("ciculant_t Failed")


def test_SubwordEmbedding(train_config, model_config):
    embedding = SubwordEmbedding(model_config)
    input_token_ids = torch.randint(high = model_config.vocab_size, size = (train_config.train_batch_size, SEQ_LEN))
    positional_input = torch.randint(high = model_config.vocab_size, size = (train_config.train_batch_size, SEQ_LEN))

    embedding_output = embedding(input_ids = input_token_ids, position_ids = positional_input)

    assert embedding_output.size() == (train_config.train_batch_size, SEQ_LEN, model_config.num_embed_hidden,)


def test_SelfAttention(model_config, train_config):
    attention = SelfAttention(model_config, relative_attention)

    query = torch.rand(train_config.train_batch_size, SEQ_LEN, model_config.num_embed_hidden)
    attn_mask = torch.ones(query.size()[:-1], dtype = torch.float)
    output = attention(query, attn_mask)
    assert output.size() == (train_config.train_batch_size, SEQ_LEN, model_config.num_embed_hidden,)


def test_FeedForward1(train_config, model_config):
    ff1 = FeedForward1(model_config.num_embed_hidden, model_config.feed_forward1_hidden, model_config.dropout_rate)
    embed = torch.rand(train_config.train_batch_size, SEQ_LEN, model_config.num_embed_hidden)
    output = ff1(embed)
    assert output.size() == embed.size()


def test_SharedInnerBlock(train_config, model_config):
    from random import randrange

    SIB = SharedInnerBlock(model_config, model_config.relative_attns[randrange(6)])
    embed = torch.rand(train_config.train_batch_size, SEQ_LEN, model_config.num_embed_hidden)
    attn_mask = torch.ones(embed.size()[:-1], dtype = torch.float)
    out1 = SIB(embed, attn_mask)
    assert out1.size() == embed.size()


def test_MultiheadAttention(train_config, model_config):
    MHA = MultiheadAttention(model_config)
    embed = torch.rand(train_config.train_batch_size, SEQ_LEN, model_config.num_embed_hidden)
    attn_mask = torch.ones(embed.size()[:-1], dtype = torch.float)

    assert model_config.num_embed_hidden % MHA.num_attention_heads == 0

    assert MHA(embed, attn_mask).size() == (
        train_config.train_batch_size,
        SEQ_LEN,
        model_config.num_embed_hidden * model_config.num_attention_heads,
    )


def test_TransformerLayers(model_config):
    TL = TransformerLayers(model_config)

    path = str(Path(__file__).parents[1].resolve() / "data" / "batch_context.pickle")
    with open(path, "rb") as input_file:
        encoder_input = pickle.load(input_file)
    print(type(encoder_input))
    embedding = SubwordEmbedding(model_config)
    emb_output = embedding(encoder_input.input_ids, encoder_input.position_ids)

    assert TL(encoder_input).size() == emb_output.size()[:-1] + (
        model_config.num_embed_hidden * model_config.num_attention_heads,
    )


def test_FeedForward2(model_config, train_config):
    embed = torch.rand(
        train_config.train_batch_size, SEQ_LEN, model_config.num_embed_hidden * model_config.num_attention_heads
    )
    attn_mask = torch.ones(embed.size()[:-1], dtype = torch.float)

    FF2 = FeedForward2(model_config)
    assert FF2(embed, attn_mask).size() == (train_config.train_batch_size, model_config.num_embed_hidden)


if __name__ == "__main__":
    pytest.main()
