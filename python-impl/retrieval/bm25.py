import re
import unicodedata

import jieba
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from retrieval.schemas import RetrievalHit


TOKEN_PATTERN = re.compile(
    r"[a-z0-9]+(?:[-_][a-z0-9]+)*|[\u4e00-\u9fff]+"
)


def tokenize(text: str) -> list[str]:
    """中文分词，并保留 EB-100、USB-C 等完整型号。"""

    normalized = unicodedata.normalize("NFKC", text).lower()
    tokens: list[str] = []

    for segment in TOKEN_PATTERN.findall(normalized):
        if re.fullmatch(r"[\u4e00-\u9fff]+", segment):
            tokens.extend(
                word.strip()
                for word in jieba.lcut(segment)
                if word.strip()
            )
        else:
            tokens.append(segment)

    return tokens


class BM25Retriever:
    def __init__(self, documents: list[Document]):
        if not documents:
            raise ValueError("BM25 需要至少一个文本块")

        chunk_ids = [
            document.metadata.get("chunk_id")
            for document in documents
        ]

        if any(not chunk_id for chunk_id in chunk_ids):
            raise ValueError("每个文本块必须包含 chunk_id")

        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("文本块的 chunk_id 不能重复")

        self.documents = list(documents)
        tokenized_documents = [
            tokenize(document.page_content)
            for document in self.documents
        ]

        if not all(tokenized_documents):
            raise ValueError("存在无法分词的空文本块")

        self.index = BM25Okapi(tokenized_documents)

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalHit]:
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self.index.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(self.documents)),
            key=lambda index: float(scores[index]),
            reverse=True,
        )

        return [
            RetrievalHit(
                document=self.documents[index],
                score=float(scores[index]),
                retriever="bm25",
            )
            for index in ranked_indices
            if scores[index] > 0
        ][:top_k]