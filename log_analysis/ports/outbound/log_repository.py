"""
Outbound repository interface definition

core（LogQueryService / Agent）only depends on this interfaces
implemented by CloudWatchLogRepository or ChromaLogRepository(MVP stage)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from log_analysis.domain.models import LogEvent, LogQuery


class LogRepository(ABC):

    @abstractmethod # methods that must be implemented
    def query_logs(self, query: LogQuery) -> list[LogEvent]:
        """
        按 query 检索日志事件列表。

        实现方约定：
        - 语法类错误 → 抛 QuerySyntaxError（带上原始错误与 query 语句）
        - 超时 / 后端故障 → 抛 LogBackendError
        - 无匹配结果 → 返回空列表（不是异常）
        """
        raise NotImplementedError
