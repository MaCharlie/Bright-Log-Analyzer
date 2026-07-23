"""
Flask Web 服务入口：暴露日志 RAG 系统的 HTTP API。

端点一览：
    GET  /        健康检查
    POST /ingest  日志文件入库（预处理 → 向量化 → ChromaDB）
    POST /search  语义检索（仅返回相关日志块，不生成回答）
    POST /answer  完整 RAG 链路（检索 → 可选重排 → LLM 生成回答）
"""

from flask import Flask, request, jsonify

import config
from modules import LogPreprocessing, VectorDB
from modules.RagPipeline import retrieve_and_answer
from modules.Reranker import Reranker

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/ingest', methods=['POST'])
def ingest_logs():
    """
    日志入库端点。

    请求体：{"file_path": "/path/to/logs.log"}
    流程：读取日志文件 → 按 reqId 分组预处理 → 向量化写入 ChromaDB
    """
    # todo: try catch 结构处理异常
    file_path = request.json['file_path']
    log_preprocessor = LogPreprocessing.LogPreprocessing()
    vector_db = VectorDB.VectorDB()

    # 1. 日志文件预处理：解析、分组、提取 metadata、生成 text_for_embedding
    log_blocks = log_preprocessor.process_log_file(file_path)

    # 2. 批量 embedding 并写入向量库
    vector_db.store_log_blocks(log_blocks)
    return jsonify({
        'status': 'success',
        'message': '储存成功',
        'blocks_ingested': len(log_blocks)
    })


@app.route('/search', methods=['POST'])
def search():
    """
    语义检索端点（不含 LLM 生成）。

    请求体：{"query": "...", "top_k": 5, "filters": {"service": "UserService"}}
    返回：按相似度排序的日志块列表（含 text、score、metadata）
    """
    data = request.json
    vector_db = VectorDB.VectorDB()

    results = vector_db.search_logs(
        query=data['query'],
        filters=data.get('filters', None),
        top_k=data.get('top_k', 5)
    )

    # 可选：启用 CrossEncoder 重排序（当前默认关闭，评估时可通过 --rerank 对比效果）
    # reranker = Reranker()
    # reranked_results = reranker.rerank_results(data['query'],
    #                                   [r['text'] for r in results])

    return jsonify({
        "results": results,
        "status": "success",
        "message": "查询成功"
    })


@app.route('/answer', methods=['POST'])
def answer():
    """
    完整 RAG 问答端点。

    请求体：
        {
            "query": "req-123 发生了什么错误？",
            "top_k": 5,                          // 可选，默认 5
            "filters": {"service": "UserService"}, // 可选
            "rerank": false                       // 可选，是否启用重排序
        }

    返回：
        {
            "response": "LLM 生成的回答",
            "retrieved_contexts": ["日志块1", "日志块2", ...],
            "results": [{id, text, score, metadata}, ...],
            "status": "success"
        }

    与 eval/run_ragas_eval.py 共用 retrieve_and_answer()，保证线上与评估逻辑一致。
    """
    data = request.json
    result = retrieve_and_answer(
        query=data['query'],
        top_k=data.get('top_k', 5),
        filters=data.get('filters'),
        rerank=data.get('rerank', False),
    )
    return jsonify({
        "response": result["response"],
        "retrieved_contexts": result["retrieved_contexts"],
        "results": result["results"],
        "status": "success",
        "message": "回答成功",
    })


if __name__ == '__main__':
    app.run(host=config.app_host, port=config.app_port, debug=True)
