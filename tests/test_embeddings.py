from __future__ import annotations

from hayfind.embeddings import MODEL_NAME, GeminiEmbedder


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
