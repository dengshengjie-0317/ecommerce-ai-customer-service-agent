from pathlib import Path
from threading import Lock


class BGEM3Encoder:
    """使用本地 BGE-M3 模型生成 1024 维 Dense 向量。"""

    dimension = 1024

    def __init__(
        self,
        model_path: str,
        batch_size: int = 1,
        max_length: int = 1024,
    ):
        # 延迟导入：只有使用编码器时才加载推理依赖。
        import torch
        from FlagEmbedding import BGEM3FlagModel

        path = Path(model_path).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"模型目录不存在：{path}")
        if batch_size <= 0 or max_length <= 0:
            raise ValueError("batch_size 和 max_length 必须大于 0")

        torch.set_num_threads(3)

        self.batch_size = batch_size
        self.max_length = max_length
        self._lock = Lock()
        self._model = BGEM3FlagModel(
            str(path),
            use_fp16=False,
            devices="cpu",
            normalize_embeddings=True,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        import numpy as np

        if not texts:
            return []
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("待编码文本必须是非空字符串")

        # 同一模型实例串行推理，控制 CPU 和内存开销。
        with self._lock:
            result = self._model.encode(
                texts,
                batch_size=self.batch_size,
                max_length=self.max_length,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )

        vectors = np.asarray(result["dense_vecs"], dtype=np.float32)
        if vectors.shape != (len(texts), self.dimension):
            raise ValueError(f"向量形状异常：{vectors.shape}")
        if not np.isfinite(vectors).all():
            raise ValueError("向量包含无效数值")

        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]