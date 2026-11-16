import json
from pathlib import Path

from langchain_core.documents import Document
from pydantic import BaseModel, ConfigDict, Field

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
KNOWLEDGE_DIR = Path(__file__).resolve().parents[1] / "knowledge"


class KnowledgeEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_id: str = Field(min_length=1)
    path: str = Field(min_length=1)
    version: int = Field(gt=0)
    effective_date: str = Field(min_length=1)
    applicability: str = Field(min_length=1)
    content_type: str = Field(min_length=1)


class KnowledgeManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    documents: list[KnowledgeEntry]


def load_knowledge_documents(
    knowledge_dir: Path | None = None,
) -> list[Document]:
    """读取知识清单和 Markdown 原文，保留可追溯元数据。"""

    root = (knowledge_dir or KNOWLEDGE_DIR).resolve()
    manifest_path = root / "manifest.json"

    if not manifest_path.is_file():
        raise FileNotFoundError(f"找不到知识清单：{manifest_path}")

    raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = KnowledgeManifest.model_validate(raw_manifest)

    doc_ids = [entry.doc_id for entry in manifest.documents]
    if len(doc_ids) != len(set(doc_ids)):
        raise ValueError("知识清单中存在重复的 doc_id")

    documents: list[Document] = []

    for entry in manifest.documents:
        source_path = (root / entry.path).resolve()

        if not source_path.is_relative_to(root):
            raise ValueError(f"知识文件路径越界：{entry.path}")

        if not source_path.is_file():
            raise FileNotFoundError(f"找不到知识文件：{source_path}")

        content = source_path.read_text(encoding="utf-8").strip()
        if not content:
            raise ValueError(f"知识文件为空：{source_path}")

        documents.append(
            Document(
                page_content=content,
                metadata={
                    "doc_id": entry.doc_id,
                    "source": source_path.relative_to(root).as_posix(),
                    "version": entry.version,
                    "effective_date": entry.effective_date,
                    "applicability": entry.applicability,
                    "content_type": entry.content_type,
                },
            )
        )

    return documents

def split_knowledge_documents(
    documents: list[Document],
    chunk_size: int = 600,
    chunk_overlap: int = 80,
) -> list[Document]:
    """按 Markdown 标题分段，再限制块大小；保留来源和版本信息。"""

    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap 必须在 0 和 chunk_size 之间")

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "document_title"),
            ("##", "section"),
        ],
        strip_headers=False,
    )
    size_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )

    chunks: list[Document] = []

    for document in documents:
        sections = header_splitter.split_text(document.page_content)
        section_chunks = size_splitter.split_documents(sections)

        for chunk_index, chunk in enumerate(section_chunks):
            content = chunk.page_content.strip()
            if not content:
                continue

            metadata = {
                **document.metadata,
                **chunk.metadata,
                "chunk_index": chunk_index,
                "chunk_id": (
                    f"{document.metadata['doc_id']}"
                    f":v{document.metadata['version']}"
                    f":{chunk_index:04d}"
                ),
            }

            chunks.append(
                Document(
                    page_content=content,
                    metadata=metadata,
                )
            )

    return chunks