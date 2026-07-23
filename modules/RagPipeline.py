"""
RAG 编排模块：将「检索」与「生成」串联为一条完整链路。

整体流程：
    用户问题 query
        → VectorDB.search_logs()   # 语义检索 + 元数据过滤
        → [可选] Reranker          # 对 Top-K 结果重新排序
        → build_log_qa_messages()  # 拼装 LLM 对话消息
        → RagLLM.chat()            # 调用本地 Ollama 生成回答

本模块被以下两处共用，保证 API 与离线评估使用同一套逻辑：
    - apps/appLog.py 的 /answer 端点
    - eval/run_ragas_eval.py 评估脚本
"""

from typing import Optional

import config
from modules import VectorDB
from modules.RagLLM import RagLLM
from modules.Reranker import Reranker

# 系统提示词：约束 LLM 只能依据检索到的日志上下文作答，减少幻觉
SYSTEM_PROMPT = """You are a log analysis assistant. Answer questions using ONLY the provided log contexts.
Rules:
1. Base every claim on the retrieved log contexts.
2. Mention request_id, service name, and log level when available.
3. If the logs do not contain enough information, reply exactly: 日志中未找到相关信息
4. Keep answers concise and factual."""


def build_log_qa_messages(query: str, contexts: list) -> list:
    """
    将用户问题与检索到的日志块拼装为 Chat Completions 格式的 messages。

    Args:
        query: 用户的自然语言问题
        contexts: 检索到的日志块文本列表（每个元素对应一个 reqId 聚合块）

    Returns:
        OpenAI Chat 格式的 messages 列表，包含 system 与 user 两条消息
    """
    # 若无检索结果，显式告知 LLM「没有上下文」，避免其凭空编造
    if not contexts:
        context_text = "(no log contexts retrieved)"
    else:
        # 为每条上下文编号，便于 LLM 在回答中引用
        context_text = "\n\n".join(
            f"[{index + 1}] {context}" for index, context in enumerate(contexts)
        )

    user_content = (
        f"Question: {query}\n\n"
        f"Log contexts:\n{context_text}\n\n"
        "Answer the question using only the log contexts above."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def retrieve_and_answer(
    query: str,
    top_k: int = 5,
    filters: Optional[dict] = None,
    rerank: bool = False,
    vector_db: Optional[VectorDB.VectorDB] = None,
    model: Optional[str] = None,
) -> dict:
    """
    RAG 主入口：检索日志上下文 → 生成回答。

    Args:
        query: 用户问题
        top_k: 检索返回的日志块数量
        filters: 元数据过滤条件，支持 start_time（>=）和 service（=）
        rerank: 是否启用 CrossEncoder 重排序（默认关闭）
        vector_db: 可选，传入评估专用的 VectorDB 实例；不传则使用生产库
        model: 可选，指定生成模型；不传则使用 config.model_name

    Returns:
        dict 包含三个字段：
            - response: LLM 生成的回答文本
            - retrieved_contexts: 用于生成的日志块文本列表（ragas 评估需要）
            - results: 完整检索结果（含 id、score、metadata）
    """
    # 1. 检索：默认连生产库，评估脚本会传入 eval 专用实例
    db = vector_db or VectorDB.VectorDB()
    results = db.search_logs(query=query, filters=filters, top_k=top_k)

    # 2. 可选重排序：用 CrossEncoder 对 (query, context) 对打分，提升相关块排名
    if rerank and results:
        reranker = Reranker()
        ranked = reranker.rerank_results(query, [result["text"] for result in results])
        # 建立 text → 原始 result 的映射，重排后还原完整 metadata
        text_to_result = {result["text"]: result for result in results}
        results = [text_to_result[text] for text, _score in ranked if text in text_to_result]

    # 3. 提取纯文本上下文，供 LLM 与 ragas 使用
    contexts = [result["text"] for result in results]

    # 4. 生成：拼装 prompt 并调用 Ollama
    messages = build_log_qa_messages(query, contexts)
    answer = RagLLM().chat(messages, model=model or config.model_name)

    return {
        "response": answer,
        "retrieved_contexts": contexts,
        "results": results,
    }
