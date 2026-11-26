import pytest

from rag_pipeline.chunking.chunk import Chunk
from rag_pipeline.reranking.cross_encoder import CrossEncoderReranker
from rag_pipeline.vectorstore.qdrant import SearchHit


class FakeCrossEncoder:
    """Score passages by keyword so no model is downloaded."""

    def __init__(self):
        self.calls = []

    def predict(self, pairs, **kwargs):
        self.calls.append((pairs, kwargs))

        return [
            float(passage.lower().count(query.lower()))
            for query, passage in pairs
        ]


def make_hit(text, score):
    return SearchHit(
        chunk=Chunk(
            document_id="doc-001",
            text=text,
            index=0,
            start=0,
            end=len(text),
        ),
        score=score,
        source="notes.txt",
        pages=[],
    )


def test_rerank_orders_by_cross_encoder_score():
    reranker = CrossEncoderReranker(model=FakeCrossEncoder())

    hits = [
        make_hit("Coffee tastes good.", 0.9),
        make_hit("Robots and more robots.", 0.4),
        make_hit("A robot navigates.", 0.6),
    ]

    reranked = reranker.rerank("robot", hits)

    assert [hit.chunk.text for hit in reranked] == [
        "Robots and more robots.",
        "A robot navigates.",
        "Coffee tastes good.",
    ]
    assert [hit.rerank_score for hit in reranked] == [2.0, 1.0, 0.0]

    # The original vector similarity is preserved.
    assert reranked[0].score == pytest.approx(0.4)


def test_rerank_sends_query_passage_pairs():
    model = FakeCrossEncoder()
    reranker = CrossEncoderReranker(model=model)

    reranker.rerank("robot", [make_hit("A robot.", 0.5)])

    pairs, options = model.calls[0]

    assert pairs == [("robot", "A robot.")]
    assert options["show_progress_bar"] is False


def test_rerank_empty_hits_skips_model():
    model = FakeCrossEncoder()
    reranker = CrossEncoderReranker(model=model)

    assert reranker.rerank("robot", []) == []
    assert model.calls == []
