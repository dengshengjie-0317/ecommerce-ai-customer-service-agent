from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from retrieval.schemas import RetrievalHit


class MilvusDenseRetriever:
    """保存知识块及其向量，并进行余弦相似度检索。"""

    def __init__(
        self,
        db_path: str,
        encoder: Any,
        collection_name: str = "ecommerce_bge_m3_v1",
    ):
        from pymilvus import DataType, MilvusClient

        path = Path(db_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)

        self.encoder = encoder
        self.collection_name = collection_name
        self.client = MilvusClient(uri=str(path))

        if not self.client.has_collection(collection_name):
            schema = MilvusClient.create_schema(
                auto_id=False,
                enable_dynamic_field=False,
            )
            schema.add_field(
                field_name="chunk_id",
                datatype=DataType.VARCHAR,
                is_primary=True,
                max_length=512,
            )
            schema.add_field(
                field_name="text",
                datatype=DataType.VARCHAR,
                max_length=65535,
            )
            schema.add_field(
                field_name="metadata",
                datatype=DataType.JSON,
            )
            schema.add_field(
                field_name="vector",
                datatype=DataType.FLOAT_VECTOR,
                dim=encoder.dimension,
            )

            index_params = MilvusClient.prepare_index_params()
            index_params.add_index(
                field_name="vector",
                index_type="FLAT",
                metric_type="COSINE",
            )

            self.client.create_collection(
                collection_name=collection_name,
                schema=schema,
                index_params=index_params,
                consistency_level="Strong",
            )

        self.client.load_collection(collection_name)

    def upsert_documents(self, documents: list[Document]) -> int:
        if not documents:
            return 0

        chunk_ids = [
            document.metadata.get("chunk_id")
            for document in documents
        ]
        if any(not chunk_id for chunk_id in chunk_ids):
            raise ValueError("每个文本块必须包含 chunk_id")
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("同一批次的 chunk_id 不能重复")

        vectors = self.encoder.embed_documents(
            [document.page_content for document in documents]
        )

        rows = [
            {
                "chunk_id": document.metadata["chunk_id"],
                "text": document.page_content,
                "metadata": dict(document.metadata),
                "vector": vector,
            }
            for document, vector in zip(documents, vectors, strict=True)
        ]

        self.client.upsert(
            collection_name=self.collection_name,
            data=rows,
        )
        self.client.flush(self.collection_name)
        return len(rows)

    def search(self, query: str, top_k: int = 5) -> list[RetrievalHit]:
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")
        if not query.strip():
            return []

        vector = self.encoder.embed_query(query)

        results = self.client.search(
            collection_name=self.collection_name,
            data=[vector],
            anns_field="vector",
            limit=top_k,
            search_params={"metric_type": "COSINE"},
            output_fields=["chunk_id", "text", "metadata"],
        )

        return [
            RetrievalHit(
                document=Document(
                    page_content=hit["entity"]["text"],
                    metadata=hit["entity"]["metadata"],
                ),
                score=float(hit["distance"]),
                retriever="dense",
            )
            for hit in results[0]
        ]

    def close(self) -> None:
        self.client.close()