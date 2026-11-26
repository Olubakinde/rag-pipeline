from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated
from zipfile import BadZipFile

from docx.opc.exceptions import PackageNotFoundError
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from pypdf.errors import PdfReadError

from rag_pipeline.config import Settings
from rag_pipeline.factory import open_pipeline
from rag_pipeline.retrieval import RAGPipeline

ALLOWED_EXTENSIONS = {".txt", ".pdf", ".docx"}


class IndexResponse(BaseModel):
    document_id: str
    filename: str
    chunks_indexed: int


class DocumentSummary(BaseModel):
    document_id: str
    source: str
    chunk_count: int


class DocumentListResponse(BaseModel):
    documents: list[DocumentSummary]


class DeleteResponse(BaseModel):
    document_id: str
    chunks_deleted: int


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    limit: int = Field(default=5, ge=1, le=50)
    min_score: float | None = Field(default=None, ge=-1, le=1)
    sources: list[str] | None = None
    diversity: float = Field(default=0.0, ge=0, le=1)


class SearchResult(BaseModel):
    text: str
    score: float
    source: str
    document_id: str
    chunk_index: int
    start: int
    end: int
    pages: list[int]
    rerank_score: float | None = None


class SearchResponse(BaseModel):
    results: list[SearchResult]


def create_app(
    pipeline: RAGPipeline | None = None,
    settings: Settings | None = None,
) -> FastAPI:
    """Create the API with an injectable retrieval pipeline."""

    if settings is None:
        settings = Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if pipeline is not None:
            # Tests can supply a lightweight fake embedding model.
            app.state.pipeline = pipeline
            yield
            return

        with open_pipeline(settings) as configured_pipeline:
            app.state.pipeline = configured_pipeline
            yield

    app = FastAPI(
        title="RAG Pipeline API",
        description="Document ingestion and semantic retrieval API",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post(
        "/documents",
        response_model=IndexResponse,
        status_code=201,
    )
    def upload_document(file: Annotated[UploadFile, File()]):
        """Upload, parse, chunk, embed, and index a document."""

        filename = Path(
            (file.filename or "").replace("\\", "/")
        ).name

        extension = Path(filename).suffix.lower()

        if extension not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=415,
                detail="Supported formats: TXT, PDF, DOCX",
            )

        with TemporaryDirectory() as temporary_directory:
            file_path = Path(temporary_directory) / filename

            total_bytes = 0

            # Stream to disk without loading the full upload into memory.
            with file_path.open("wb") as output:
                while True:
                    block = file.file.read(1024 * 1024)

                    if not block:
                        break

                    total_bytes += len(block)

                    if total_bytes > settings.max_upload_bytes:
                        raise HTTPException(
                            status_code=413,
                            detail=(
                                "File exceeds the "
                                f"{settings.max_upload_mb} MB limit"
                            ),
                        )

                    output.write(block)

            if total_bytes == 0:
                raise HTTPException(
                    status_code=400,
                    detail="Uploaded file is empty",
                )

            try:
                indexed = app.state.pipeline.index(file_path)
            except (
                ValueError,
                UnicodeError,
                PdfReadError,
                PackageNotFoundError,
                BadZipFile,
            ) as exc:
                raise HTTPException(
                    status_code=422,
                    detail="The document could not be parsed",
                ) from exc

        return IndexResponse(
            document_id=indexed.document_id,
            filename=filename,
            chunks_indexed=indexed.chunk_count,
        )

    @app.get(
        "/documents",
        response_model=DocumentListResponse,
    )
    def list_documents():
        """List every indexed document."""

        return DocumentListResponse(
            documents=[
                DocumentSummary(
                    document_id=document.document_id,
                    source=document.source,
                    chunk_count=document.chunk_count,
                )
                for document in app.state.pipeline.list_documents()
            ]
        )

    @app.delete(
        "/documents/{document_id}",
        response_model=DeleteResponse,
    )
    def delete_document(document_id: str):
        """Remove a document and all of its chunks from the index."""

        deleted = app.state.pipeline.delete_document(document_id)

        if deleted == 0:
            raise HTTPException(
                status_code=404,
                detail="Document not found",
            )

        return DeleteResponse(
            document_id=document_id,
            chunks_deleted=deleted,
        )

    @app.post(
        "/search",
        response_model=SearchResponse,
    )
    def search_documents(request: SearchRequest):
        """Search indexed documents by semantic similarity."""

        if not request.query.strip():
            raise HTTPException(
                status_code=422,
                detail="Search query cannot be blank",
            )

        hits = app.state.pipeline.search(
            query=request.query,
            limit=request.limit,
            min_score=request.min_score,
            sources=request.sources,
            diversity=request.diversity,
        )

        return SearchResponse(
            results=[
                SearchResult(
                    text=hit.chunk.text,
                    score=hit.score,
                    source=hit.source,
                    document_id=hit.chunk.document_id,
                    chunk_index=hit.chunk.index,
                    start=hit.chunk.start,
                    end=hit.chunk.end,
                    pages=hit.pages,
                    rerank_score=hit.rerank_score,
                )
                for hit in hits
            ]
        )

    return app


app = create_app()