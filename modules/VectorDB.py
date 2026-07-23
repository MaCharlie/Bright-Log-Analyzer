"""
向量数据库模块：负责日志块的 embedding 存储与语义检索。

架构要点：
    - 多实例注册表：按 (chroma_path, collection_name) 区分生产库与评估库
    - Embedding 模型全局共享：避免重复加载 SentenceTransformer
    - 混合检索：向量相似度 + ChromaDB 元数据过滤（start_time、service）

数据流（写入）：
    LogPreprocessing 输出的 log_blocks
        → encode(text_for_embedding)
        → ChromaDB collection.add(ids, embeddings, documents, metadatas)

数据流（检索）：
    query 文本
        → encode(query)
        → collection.query(query_embeddings, where=..., n_results=top_k)
        → 返回 [{id, text, score, metadata}, ...]
"""

import chromadb
import shutil
from pathlib import Path

from sentence_transformers import SentenceTransformer

import config


class VectorDB(object):
    # 实例注册表：key = (chroma_path, collection_name)，value = VectorDB 实例
    # 这样生产库 ./chroma_data 与评估库 ./chroma_data_eval 可以并存且互不干扰
    # 多个库对应多个实例
    _instances = {}

    # Embedding 模型全局单例：所有 VectorDB 实例共用同一份模型权重
    _embed_model = None

    def __new__(cls, chroma_path=None, collection_name=None):
        """
        按 (路径, 集合名) 创建或复用 VectorDB 实例。

        不传参数时使用 config 中的生产库默认值；
        评估脚本传入 eval_chroma_data_path / eval_collection_name 以隔离评估数据。
        """
        chroma_path = chroma_path or config.chroma_data_path
        collection_name = collection_name or config.collection_name
        key = (chroma_path, collection_name)

        if key not in cls._instances:
            instance = super().__new__(cls)
            instance.chroma_path = chroma_path
            instance.collection_name = collection_name

            """初始化向量数据库"""
            # PersistentClient：数据持久化到本地目录，重启后自动加载
            instance.client = chromadb.PersistentClient(path=chroma_path)
            instance.collection = instance.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},  # 使用余弦距离衡量向量相似度
            )

            """初始化embedding模型"""
            # 首次创建任意实例时加载 embedding 模型，后续实例直接复用
            if cls._embed_model is None:
                cls._embed_model = SentenceTransformer(
                    config.embed_model_name, config.embed_cache_path
                )
            instance.embed_model = cls._embed_model
            cls._instances[key] = instance

        return cls._instances[key]



    def store_log_blocks(self, log_blocks: list):
        """
        批量写入预处理后的日志块。
        本函数几乎没有变动。
        由于存在多个VectorDB实例，本函数降级为实例方法（即与不同数据库对应）

        Args:
            log_blocks: LogPreprocessing.process_log_file() 的返回值，
                        每个元素含 metadata、text_for_embedding、raw_logs
        """
        # 分别存储log_id, 文档, 元数据
        ids, documents, metadatas = [], [], []

        for block in log_blocks:
            # 组合 request_id + start_time + service hash 作为唯一 ID，避免重复写入冲突
            log_id = (
                f"{block['metadata']['request_id']}-"
                f"{block['metadata']['start_time']}-"
                f"{hash(block['metadata']['services'][0])}"
            )
            ids.append(log_id)
            # text_for_embedding：同一 reqId 下所有 message 用 | 拼接，作为向量化的文本
            documents.append(block["text_for_embedding"])
            # 元数据用于检索时的 where 过滤，不参与 embedding
            metadatas.append({
                "request_id": block["metadata"]["request_id"],
                "start_time": block["metadata"]["start_time"],
                "service": block["metadata"]["services"][0],
            })

        # 批量 encode 比逐条 encode 更高效（SentenceTransformer 内部有 batch 优化）
        embeddings = self.embed_model.encode(documents).tolist()
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def search_logs(self, query: str, filters: dict = None, top_k: int = 5):
        """
        混合检索：语义相似度 + 可选元数据过滤。
        降级为实例方法。

        Args:
            query: 自然语言查询
            filters: 可选过滤，如 {"service": "PaymentService", "start_time": "2023-10-25 14:30:19,010"}
            top_k: 返回最相似的 K 条日志块

        Returns:
            按相似度排序的结果列表，score 为余弦距离（越小越相似）
        """
        # 1. 生成查询向量，用于与库中 embeddings 做近邻搜索
        query_embedding = self.embed_model.encode([query]).tolist()[0]

        # 2. 构建混合查询条件（仅支持部分字段的精确/范围过滤）
        where_conditions = {}
        if filters:
            if "start_time" in filters:
                where_conditions["start_time"] = {"$gte": filters["start_time"]}
            if "service" in filters:
                where_conditions["service"] = {"$eq": filters["service"]}

        # 查询参数
        query_kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        # 有过滤条件时才传 where，避免空 dict 引发 ChromaDB 报错
        if where_conditions:
            query_kwargs["where"] = where_conditions

        # 4. 整理结果
        results = self.collection.query(**query_kwargs)

        output = []
        # 无匹配结果时 ChromaDB 返回空列表，需提前返回
        if not results["ids"] or not results["ids"][0]:
            return output

        for i in range(len(results["ids"][0])):
            output.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "score": results["distances"][0][i],
                "metadata": results["metadatas"][0][i],
            })

        return output
