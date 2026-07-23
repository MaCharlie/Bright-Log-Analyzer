# =============================================================================
# LoggingFlask 全局配置
# =============================================================================

# --- 运行信息 ---
app_host = "0.0.0.0"   # 监听所有网卡，便于容器/远程访问
app_port = 5000

# --- Embedding 与向量库（生产环境） ---
# embed_model_path = "D:/cs/projects/RAG/mooc/RAG_full_stack_course_notebooks/llm_app/gte-large-zh/"
# chroma_data_path = "D:\cs\projects\RAG\LoggingFlask\chroma_data"

embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"  # 轻量英文 embedding 模型
embed_cache_path = "./model/embeddings"                     # 模型权重本地缓存目录
chroma_data_path = "./chroma_data"                          # 生产向量库持久化路径
collection_name = "server_logs"                               # ChromaDB 集合名

# --- 评估专用向量库（与生产库隔离，每次 eval 前会清空重建） ---
eval_chroma_data_path = "./chroma_data_eval"
eval_collection_name = "server_logs_eval"
eval_judge_model = "qwen2.5:7b"  # ragas 评判用的较小模型，降低成本

# --- LLM（Ollama 本地推理） ---
base_url = "http://localhost:11434/v1/"  # Ollama OpenAI 兼容 API 地址
model_name = "qwen2:72b"                   # RAG 生成使用的大模型
ollama_api_key = "ollama"                  # Ollama 不校验 key，填任意非空值即可

# --- 重排序模型 ---
reranker_model = "BAAI/bge-reranker-small"  # CrossEncoder，用于检索结果精排
