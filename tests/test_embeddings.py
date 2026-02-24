from __future__ import annotations

import pytest

from hayfind import embeddings
from hayfind.embeddings import (
    MODEL_NAME,
    OPENAI_MODEL_NAME,
    FallbackEmbedder,
    GeminiEmbedder,
    OpenAIEmbedder,
    get_embedder,
)


class _FakeEmbedding:
    def __init__(self, values: list[float]) -> None:
        self.values = values


class _FakeEmbedResponse:
    def __init__(self, embeddings: list[_FakeEmbedding]) -> None:
        self.embeddings = embeddings


class _FakeModels:
    def embed_content(self, *, model: str, contents: list[str] | str) -> _FakeEmbedResponse:
        assert model == MODEL_NAME
        if isinstance(contents, list):
            return _FakeEmbedResponse(
                embeddings=[
                    _FakeEmbedding([0.1, 0.2, 0.3]),
                    _FakeEmbedding([0.4, 0.5, 0.6]),
                ]
            )
        return _FakeEmbedResponse(embeddings=[_FakeEmbedding([0.7, 0.8, 0.9])])


class _FakeClient:
    def __init__(self) -> None:
        self.models = _FakeModels()


def test_embedder_normalizes_sdk_response_to_plain_vectors() -> None:
    embedder = GeminiEmbedder(client=_FakeClient())

    docs = embedder.embed_documents(["doc one", "doc two"])
    query = embedder.embed_query("find me")

    assert docs == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    assert query == [0.7, 0.8, 0.9]


class _FakeOpenAIEmbeddings:
    def create(self, *, model: str, input: list[str]) -> dict:
        assert model == OPENAI_MODEL_NAME
        if len(input) == 2:
            return {"data": [{"embedding": [1, 2, 3]}, {"embedding": [4, 5, 6]}]}
        return {"data": [{"embedding": [7, 8, 9]}]}


class _FakeOpenAIClient:
    def __init__(self) -> None:
        self.embeddings = _FakeOpenAIEmbeddings()


def test_openai_embedder_normalizes_response_to_plain_vectors() -> None:
    embedder = OpenAIEmbedder(client=_FakeOpenAIClient())

    docs = embedder.embed_documents(["doc one", "doc two"])
    query = embedder.embed_query("find me")

    assert docs == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    assert query == [7.0, 8.0, 9.0]


def test_get_embedder_selects_provider_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HAYFIND_EMBED_PROVIDER", "openai")
    monkeypatch.delenv("HAYFIND_EMBED_FALLBACK_PROVIDER", raising=False)

    class _Chosen:
        pass

    monkeypatch.setattr(embeddings, "OpenAIEmbedder", lambda: _Chosen())
    embedder = get_embedder()

    assert isinstance(embedder, _Chosen)


def test_get_embedder_rejects_local_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HAYFIND_EMBED_PROVIDER", "local")
    monkeypatch.delenv("HAYFIND_EMBED_FALLBACK_PROVIDER", raising=False)

    with pytest.raises(RuntimeError, match="not implemented"):
        get_embedder()


def test_get_embedder_builds_fallback_for_indexing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HAYFIND_EMBED_PROVIDER", "gemini")
    monkeypatch.setenv("HAYFIND_EMBED_FALLBACK_PROVIDER", "openai")
    monkeypatch.setattr(embeddings, "GeminiEmbedder", lambda: _PrimaryBatchEmbedder())
    monkeypatch.setattr(embeddings, "OpenAIEmbedder", lambda: _FallbackBatchEmbedder())

    embedder = get_embedder(for_indexing=True)

    assert isinstance(embedder, FallbackEmbedder)


class _PrimaryBatchEmbedder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        if len(self.calls) == 2:
            raise RuntimeError("429 quota exhausted")
        return [[float(len(t))] for t in texts]

    def embed_query(self, query: str) -> list[float]:
        return [float(len(query))]


class _FallbackBatchEmbedder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[999.0] for _ in texts]

    def embed_query(self, query: str) -> list[float]:
        return [999.0]


def test_fallback_switches_and_continues_remaining_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HAYFIND_EMBED_BATCH_SIZE", "2")

    primary = _PrimaryBatchEmbedder()
    fallback = _FallbackBatchEmbedder()
    embedder = FallbackEmbedder(
        primary=primary,
        primary_provider="gemini",
        fallback=fallback,
        fallback_provider="openai",
    )

    vectors = embedder.embed_documents(["aa", "bbb", "cccc", "ddddd", "eeeeee"])

    assert vectors == [[2.0], [3.0], [999.0], [999.0], [999.0]]
    assert primary.calls == [["aa", "bbb"], ["cccc", "ddddd"]]
    assert fallback.calls == [["cccc", "ddddd"], ["eeeeee"]]
