"""
领域异常。

设计意图（给后续 Agent 自纠错用）：
- QuerySyntaxError：查询语句本身有问题（Insights 语法错误等）→ Agent 应改写 Pipeline 再试
- LogBackendError：超时、权限、网络、服务端失败等 → 不一定是改语句能解决的
"""


class LogBackendError(Exception):
    """CloudWatch / 日志后端在执行查询时发生的通用失败。"""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class QuerySyntaxError(LogBackendError):
    """
    查询字符串无法被 CloudWatch Logs Insights 接受，或查询以 Failed 结束且可判定为语句问题。

    Attributes:
        message: AWS / 后端返回的原始错误说明（务必保留原文，便于 LLM 纠错）
        query: 当时提交的 Pipeline 语句
    """

    def __init__(self, message: str, query: str) -> None:
        self.query = query
        # 拼进 Exception 字符串，方便日志打印时一眼看到语句
        super().__init__(f"{message} | query={query!r}")
