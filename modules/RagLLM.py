"""
大语言模型调用模块：通过 Ollama 的 OpenAI 兼容 API 进行推理。

Ollama 在本地 11434 端口暴露 /v1/ 接口，本模块用 OpenAI SDK 连接它，
无需 cloud API Key，适合本地开发与 ragas 评估。

提供两种调用方式：
    - chat()：推荐，使用 Chat Completions + system/user 消息（RagPipeline 使用）
    - __call__()：兼容旧版 Completions API 的单 prompt 调用
"""

from typing import Any, List, Optional

from openai import OpenAI

import config


class RagLLM(object):
    _instance = None

    def __new__(cls):
        """单例模式：全局共享一个 OpenAI client，避免重复建立 HTTP 连接。"""
        if not cls._instance:
            cls._instance = super().__new__(cls)
            # base_url 指向本地 Ollama；api_key 填任意非空字符串即可（Ollama 不校验）
            cls._instance.client = OpenAI(
                base_url=config.base_url,
                api_key=config.ollama_api_key,
            )
        return cls._instance

    def chat(self, messages: List[dict], model: Optional[str] = None, **kwargs: Any) -> str:
        """
        基于 Chat Completions 的多轮对话推理（RAG 生成推荐使用）。

        Args:
            messages: OpenAI 格式消息列表，如 [{"role": "system", ...}, {"role": "user", ...}]
            model: 模型名，默认 config.model_name（如 qwen2:72b）
            **kwargs: temperature、top_p、max_tokens 等生成参数

        Returns:
            LLM 生成的文本内容（assistant 的 message.content）
        """
        completion = self.client.chat.completions.create(
            model=model or config.model_name,
            messages=messages,
            temperature=kwargs.get("temperature", 0.1),   # 低温度 → 回答更稳定、更少随机性
            top_p=kwargs.get("top_p", 0.9),
            max_tokens=kwargs.get("max_tokens", 4096),
        )
        return completion.choices[0].message.content

    def __call__(self, prompt: str, **kwargs: Any):
        """
        兼容旧版 Completions API 的单 prompt 调用。

        保留此方法是为了不破坏可能存在的旧代码；新功能请使用 chat()。
        """
        completion = self.client.completions.create(
            model=kwargs.get("model", config.model_name),
            prompt=prompt,
            temperature=kwargs.get("temperature", 0.1),
            top_p=kwargs.get("top_p", 0.9),
            max_tokens=kwargs.get("max_tokens", 4096),
            stream=kwargs.get("stream", False),
        )
        if kwargs.get("stream", False):
            return completion
        return completion.choices[0].text
