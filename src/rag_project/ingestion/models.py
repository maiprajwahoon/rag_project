"""Data models for document ingestion and chunking."""

from pydantic import BaseModel, Field


class Document(BaseModel):
    """Represents an ingested markdown source document.

    Attributes:
        source_path: Path of the document relative to the docs root directory.
        title: Extracted title from the first level-1 heading, or filename if none.
        text: Full markdown text content with YAML front matter removed.
    """

    source_path: str = Field(
        ...,
        description="Path relative to the documents root directory.",
    )
    title: str = Field(
        ...,
        description="Document title from first # heading or filename.",
    )
    text: str = Field(
        ...,
        description="Markdown text with front matter stripped.",
    )


class Chunk(BaseModel):
    """Represents a text chunk produced from a document.

    Attributes:
        chunk_id: Stable deterministic ID formatted as f"{source_path}::{chunk_index:04d}".
        source_path: Source document path with forward slashes.
        title: Title of the source document.
        heading_path: Breadcrumb heading hierarchy string (e.g. "Tutorial > Query Parameters").
        chunk_index: 0-based sequential index of the chunk within the document.
        text: Text content of the chunk.
        token_count: Number of tokens according to the configured tokenizer.
    """

    chunk_id: str = Field(
        ...,
        description="Deterministic ID: f'{source_path}::{chunk_index:04d}'.",
    )
    source_path: str = Field(
        ...,
        description="Relative source document path with forward slashes.",
    )
    title: str = Field(
        ...,
        description="Document title.",
    )
    heading_path: str = Field(
        ...,
        description="Hierarchical heading path, e.g. 'Tutorial > Query Parameters'.",
    )
    chunk_index: int = Field(
        ...,
        description="0-based sequential index of the chunk in the document.",
    )
    text: str = Field(
        ...,
        description="Text content of the chunk.",
    )
    token_count: int = Field(
        ...,
        description="Token count of this chunk.",
    )
    anchor: str | None = Field(
        default=None,
        description="Optional HTML anchor identifier stripped from the heading.",
    )
