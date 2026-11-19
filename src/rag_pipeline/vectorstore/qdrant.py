from dataclasses import dataclass
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

from rag_pipeline.chunking.chunk import Chunk
from rag_pipeline.ingestion import IngestionResult


@dataclass
class SearchHit:
    """A retrieved passage and its similarity score."""

    chunk: Chunk
    score: float
    source: str
    pages: list[int]


class QdrantVectorStore:
    """Store and search document embeddings using Qdrant."""

    def __init__(
        self,
        collection_name: str = "documents",
        vector_size: int = 384,
        client: QdrantClient | None = None,
        path: str | Path | None = None,
    ):
        if vector_size <= 0:
            raise ValueError("Vector size must be positive")

        if client is not None and path is not None:
            raise ValueError("Provide either client or path, not both")

        self.collection_name = collection_name
        self.vector_size = vector_size

        if client is not None:
            self.client = client
        elif path is not None:
            self.client = QdrantClient(path=str(path))
        else:
            self.client = QdrantClient(":memory:")

        if not self.client.collection_exists(
            collection_name=self.collection_name
        ):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE,
                ),
            )

    def add(
        self,
        result: IngestionResult,
        vectors: list[list[float]],
    ) -> int:
        """Store chunks with their embeddings and source metadata."""

        if len(result.chunks) != len(vectors):
            raise ValueError(
                "Each chunk must have exactly one embedding"
            )

        points = []

        for chunk, vector in zip(
            result.chunks, vectors, strict=True
        ):
            if len(vector) != self.vector_size:
                raise ValueError(
                    f"Expected {self.vector_size} dimensions, "
                    f"received {len(vector)}"
                )

            if chunk.document_id != result.document.id:
                raise ValueError(
                    "Chunk does not belong to the supplied document"
                )

            # Deterministic IDs prevent duplicate records
            # when the same chunks are indexed again.
            point_id = str(
                uuid5(
                    NAMESPACE_URL,
                    (
                        f"{chunk.document_id}:"
                        f"{chunk.index}:"
                        f"{chunk.start}:"
                        f"{chunk.end}"
                    ),
                )
            )

            # Identify original PDF pages intersecting this chunk.
            page_spans = result.document.metadata.get(
                "page_spans", []
            )

            pages = [
                span["page"]
                for span in page_spans
                if chunk.start < span["end"]
                and chunk.end > span["start"]
            ]

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "document_id": chunk.document_id,
                        "text": chunk.text,
                        "index": chunk.index,
                        "start": chunk.start,
                        "end": chunk.end,
                        "source": result.document.source,
                        "pages": pages,
                    },
                )
            )

        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )

        return len(points)

    def search(
        self,
        query_vector: list[float],
        limit: int = 5,
    ) -> list[SearchHit]:
        """Retrieve the closest matching document chunks."""

        if len(query_vector) != self.vector_size:
            raise ValueError(
                f"Query vector must have {self.vector_size} dimensions"
            )

        if limit <= 0:
            raise ValueError("Search limit must be positive")

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
        )

        hits = []

        for point in response.points:
            payload = point.payload or {}

            chunk = Chunk(
                document_id=payload["document_id"],
                text=payload["text"],
                index=payload["index"],
                start=payload["start"],
                end=payload["end"],
            )

            hits.append(
                SearchHit(
                    chunk=chunk,
                    score=point.score,
                    source=payload["source"],
                    pages=payload.get("pages", []),
                )
            )

        return hits