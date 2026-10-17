import pytest

from legal_agent import embeddings


class FakeCollection:
    def __init__(self, metadata):
        self.metadata = metadata


class FakeEmbedder:
    name = "qwen3-0.6b"


@pytest.fixture
def fake_active(monkeypatch):
    monkeypatch.setattr(embeddings, "get_embedder", lambda name=None: FakeEmbedder())


def test_mismatch_raises(fake_active):
    with pytest.raises(RuntimeError, match="e5-small"):
        embeddings.get_embedder_for_collection(FakeCollection({"embedder": "e5-small"}))


def test_match_returns_embedder(fake_active):
    emb = embeddings.get_embedder_for_collection(FakeCollection({"embedder": "qwen3-0.6b"}))
    assert emb.name == "qwen3-0.6b"


def test_missing_metadata_skips_guard(fake_active):
    assert embeddings.get_embedder_for_collection(FakeCollection(None)).name == "qwen3-0.6b"
