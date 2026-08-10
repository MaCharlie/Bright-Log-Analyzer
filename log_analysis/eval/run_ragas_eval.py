#!/usr/bin/env python3
"""
LoggingFlask 离线 Ragas 评估脚本。

评估流程（三阶段）：
    Phase 1 - 准备数据
        清空 eval 向量库 → 导入 sample_logs.log → 写入 eval ChromaDB

    Phase 2 - 运行 RAG Pipeline
        遍历 golden_set.json 中的每条样本
        → retrieve_and_answer() 得到 retrieved_contexts + response
        → 与 reference 一起组装为 ragas 所需的数据行

    Phase 3 - Ragas 打分
        用 4 个核心指标评估检索与生成质量
        → 导出 eval_report.json / eval_report.csv / baseline_summary.md

用法：
    python eval/run_ragas_eval.py \\
        --golden eval/data/golden_set.json \\
        --log-file eval/data/sample_logs.log \\
        --top-k 5 \\
        --output eval/results/

前置条件：
    - Ollama 已启动，且已 pull qwen2:72b（生成）和 qwen2.5:7b（评判）
    - pip install -r requirements-min.txt（含 ragas、datasets）
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# 将项目根目录加入 sys.path，使脚本能直接 import modules.*
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from modules import LogPreprocessing, VectorDB
from modules.RagPipeline import retrieve_and_answer


def load_golden_set(path: Path) -> list:
    """
    加载 Golden Set 测试集。

    每条样本包含：
        - user_input: 模拟用户查询
        - reference: 标准答案（人工标注）
        - reference_contexts: 期望检索到的日志块（可选，用于 Non-LLM 快检）
        - filters: 元数据过滤条件（可选）
    """
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ingest_eval_logs(log_file: Path) -> int:
    """
    将评估语料导入隔离的 eval 向量库。

    每次评估前调用 reset_eval_store() 清空旧数据，保证结果可复现。

    Returns:
        写入的日志块数量
    """
    VectorDB.VectorDB.reset_eval_store()
    eval_db = VectorDB.VectorDB(
        chroma_path=config.eval_chroma_data_path,
        collection_name=config.eval_collection_name,
    )
    log_blocks = LogPreprocessing.LogPreprocessing().process_log_file(str(log_file))
    eval_db.store_log_blocks(log_blocks)
    return len(log_blocks)


def run_pipeline(golden_set: list, top_k: int, rerank: bool, generation_model: str) -> list:
    """
    对 Golden Set 中每条样本运行完整 RAG 链路，收集 ragas 所需字段。

    Args:
        golden_set: 测试样本列表
        top_k: 检索返回条数
        rerank: 是否启用 CrossEncoder 重排序
        generation_model: Ollama 生成模型名

    Returns:
        ragas 数据行列表，每行含 user_input / reference / retrieved_contexts / response
    """
    eval_db = VectorDB.VectorDB(
        chroma_path=config.eval_chroma_data_path,
        collection_name=config.eval_collection_name,
    )
    rows = []

    for index, sample in enumerate(golden_set, start=1):
        print(f"[pipeline] {index}/{len(golden_set)}: {sample['user_input']}")
        result = retrieve_and_answer(
            query=sample["user_input"],
            top_k=top_k,
            filters=sample.get("filters"),
            rerank=rerank,
            vector_db=eval_db,
            model=generation_model,
        )
        rows.append({
            "user_input": sample["user_input"],
            "reference": sample["reference"],
            # ragas 要求 retrieved_contexts 为非空 list；无结果时填 [""] 占位
            "retrieved_contexts": result["retrieved_contexts"] or [""],
            "response": result["response"],
        })

    return rows


def build_ragas_llm():
    """
    构建 ragas 评判用的 LLM。

    通过 Langchain 的 ChatOpenAI 连接本地 Ollama，
    再用 LangchainLLMWrapper 适配 ragas 的 LLM 接口。
    评判模型使用较小的 qwen2.5:7b，与生成用的 qwen2:72b 分离。
    """
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    llm = ChatOpenAI(
        model=config.eval_judge_model,
        base_url=config.base_url,
        api_key=config.ollama_api_key,
        temperature=0,  # 评判任务需要确定性输出
    )
    return LangchainLLMWrapper(llm)


def build_ragas_embeddings():
    """
    构建 ragas 用的 Embedding 模型。

    answer_relevancy 指标需要通过 embedding 计算语义相似度，
    这里复用与检索相同的 SentenceTransformer 模型，保持一致性。
    """
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    embeddings = HuggingFaceEmbeddings(model_name=config.embed_model_name)
    return LangchainEmbeddingsWrapper(embeddings)


def evaluate_rows(rows: list):
    """
    调用 ragas.evaluate() 对数据行进行四指标评估。

    指标说明：
        - context_recall:    检索是否覆盖了 reference 中的关键信息
        - context_precision: 检索结果中相关块是否排在前面
        - faithfulness:      生成回答是否 grounded 于 retrieved_contexts（防幻觉）
        - answer_relevancy:  生成回答是否切题

    Returns:
        ragas EvaluationResult 对象，可调用 .to_pandas() 转为 DataFrame
    """
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    dataset = Dataset.from_list(rows)
    evaluator_llm = build_ragas_llm()
    evaluator_embeddings = build_ragas_embeddings()

    metrics = [
        context_recall,
        context_precision,
        faithfulness,
        answer_relevancy,
    ]

    # 为每个 metric 注入 LLM 和 embedding（ragas 内部用它们做评判）
    for metric in metrics:
        metric.llm = evaluator_llm
        if hasattr(metric, "embeddings"):
            metric.embeddings = evaluator_embeddings

    return evaluate(
        dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )


def summarize_bad_cases(rows: list, scores_df, threshold: float = 0.6) -> list:
    """
    从评估结果中筛选低分样本（bad cases），便于针对性优化。

    Args:
        rows: 原始 RAG pipeline 输出
        scores_df: ragas 评估结果的 DataFrame
        threshold: 低于此分数的指标视为 bad case（默认 0.6）

    Returns:
        bad case 列表，每条含 user_input、reference、response、low_metrics
    """
    metric_columns = [
        column for column in scores_df.columns
        if column not in {"user_input", "reference", "retrieved_contexts", "response"}
    ]
    bad_cases = []

    for index, row in scores_df.iterrows():
        low_metrics = {
            metric: float(row[metric])
            for metric in metric_columns
            if metric in row and row[metric] is not None and float(row[metric]) < threshold
        }
        if not low_metrics:
            continue

        source = rows[index]
        bad_cases.append({
            "user_input": source["user_input"],
            "reference": source["reference"],
            "response": source["response"],
            "retrieved_contexts": source["retrieved_contexts"],
            "low_metrics": low_metrics,
        })

    return bad_cases


def export_report(output_dir: Path, rows: list, ragas_result, args, blocks_ingested: int):
    """
    导出评估报告到三种格式：
        - eval_report.json:  完整结构化报告（含 metadata、平均分、逐条分数、bad cases）
        - eval_report.csv:   逐条分数表格，便于 Excel 分析
        - baseline_summary.md: 人类可读的 baseline 摘要
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    scores_df = ragas_result.to_pandas()

    metric_columns = [
        column for column in scores_df.columns
        if column not in {"user_input", "reference", "retrieved_contexts", "response"}
    ]
    averages = {
        metric: float(scores_df[metric].mean())
        for metric in metric_columns
        if metric in scores_df
    }

    bad_cases = summarize_bad_cases(rows, scores_df)
    timestamp = datetime.now(timezone.utc).isoformat()

    report = {
        "metadata": {
            "timestamp": timestamp,
            "top_k": args.top_k,
            "rerank": args.rerank,
            "generation_model": args.generation_model,
            "judge_model": config.eval_judge_model,
            "embedding_model": config.embed_model_name,
            "golden_set": str(args.golden),
            "log_file": str(args.log_file),
            "blocks_ingested": blocks_ingested,
        },
        "averages": averages,
        "per_row": scores_df.to_dict(orient="records"),
        "bad_cases": bad_cases,
    }

    json_path = output_dir / "eval_report.json"
    csv_path = output_dir / "eval_report.csv"
    baseline_path = output_dir / "baseline_summary.md"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    scores_df.to_csv(csv_path, index=False)

    # 生成 Markdown 摘要，方便快速浏览 baseline
    baseline_lines = [
        "# LoggingFlask Ragas Baseline",
        "",
        f"- Generated at: {timestamp}",
        f"- Golden set size: {len(rows)}",
        f"- Blocks ingested: {blocks_ingested}",
        f"- top_k: {args.top_k}",
        f"- rerank: {args.rerank}",
        "",
        "## Average Scores",
        "",
    ]
    for metric, value in averages.items():
        baseline_lines.append(f"- {metric}: {value:.4f}")

    baseline_lines.extend(["", "## Bad Cases", ""])
    if bad_cases:
        for case in bad_cases:
            baseline_lines.append(f"### {case['user_input']}")
            baseline_lines.append(f"- reference: {case['reference']}")
            baseline_lines.append(f"- response: {case['response']}")
            baseline_lines.append(f"- low_metrics: {case['low_metrics']}")
            baseline_lines.append("")
    else:
        baseline_lines.append("- None below threshold 0.6")

    baseline_path.write_text("\n".join(baseline_lines), encoding="utf-8")

    return report, json_path, csv_path, baseline_path


def parse_args():
    """解析命令行参数，均提供合理默认值指向 eval/data/ 下的标准文件。"""
    parser = argparse.ArgumentParser(description="Run Ragas evaluation for LoggingFlask")
    parser.add_argument(
        "--golden",
        type=Path,
        default=PROJECT_ROOT / "eval/data/golden_set.json",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=PROJECT_ROOT / "eval/data/sample_logs.log",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "eval/results",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument(
        "--generation-model",
        default=config.model_name,
        help="Ollama model used by retrieve_and_answer",
    )
    return parser.parse_args()


def main():
    """
    评估主流程：ingest → pipeline → ragas evaluate → export report
    """
    args = parse_args()

    if not args.golden.exists():
        raise FileNotFoundError(f"Golden set not found: {args.golden}")
    if not args.log_file.exists():
        raise FileNotFoundError(f"Log file not found: {args.log_file}")

    # Phase 1: 准备评估数据
    golden_set = load_golden_set(args.golden)
    print(f"[ingest] loading {args.log_file}")
    blocks_ingested = ingest_eval_logs(args.log_file)
    print(f"[ingest] stored {blocks_ingested} log blocks")

    # Phase 2: 运行 RAG Pipeline，收集 ragas 输入
    rows = run_pipeline(
        golden_set=golden_set,
        top_k=args.top_k,
        rerank=args.rerank,
        generation_model=args.generation_model,
    )

    # Phase 3: Ragas 评估 + 导出报告
    print("[ragas] evaluating...")
    ragas_result = evaluate_rows(rows)
    report, json_path, csv_path, baseline_path = export_report(
        args.output,
        rows,
        ragas_result,
        args,
        blocks_ingested,
    )

    print("[ragas] averages:")
    for metric, value in report["averages"].items():
        print(f"  - {metric}: {value:.4f}")

    print(f"[ragas] report written to {json_path}")
    print(f"[ragas] csv written to {csv_path}")
    print(f"[ragas] baseline summary written to {baseline_path}")
    print(f"[ragas] bad cases: {len(report['bad_cases'])}")


if __name__ == "__main__":
    main()
