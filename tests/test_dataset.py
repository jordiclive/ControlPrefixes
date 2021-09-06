import pytest
from sentencepiece import SentencePieceProcessor


from src.config import ConveRTTrainConfig
from src.dataset import load_instances_from_reddit_json



@pytest.fixture
def config():
    return ConveRTTrainConfig()


@pytest.fixture
def tokenizer() -> SentencePieceProcessor:
    tokenizer = SentencePieceProcessor()
    tokenizer.Load(config.sp_model_path)
    return tokenizer


def test_load_instances_from_reddit_json(config):
    instances = load_instances_from_reddit_json(config.dataset_path)
    assert len(instances) == 1000


if __name__ == "__main__":
    pytest.main()
