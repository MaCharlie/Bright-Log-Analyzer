"""
领域模型：日志查询入参与单条日志事件。

与 Java 推送侧约定的 JSON Schema（写入 CloudWatch @message 的内容）见：
    .cursor/plans/phases/02-phase1-log-schema-cw-adapter.md

本仓库只「读」这些字段；不负责 put_log_events。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator


class LogQuery(BaseModel):
    """
    Input model for LogQuery

    Phase 1 fields：
        - log_group / start_time / end_time：--> StartQuery
        - raw_pipeline：Insights pipeline statements

    Phase 2 fields：
        combine structured fields to form pipeline statements
    """

    log_group: str = Field(
        ...,
        description="CloudWatch Log Group name，eg. /loganalysis/app-logs",
    )
    start_time: datetime = Field(
        ...,
        description="start time for query (datetime format with timezone)",
    )
    end_time: datetime = Field(
        ...,
        description="end time for query",
    )
    # ---- for Phase 1 ----
    raw_pipeline: Optional[str] = Field(
        default=None,
        description="CloudWatch Logs Insights pipeline grammar",
    )

    # ---- for Phase 2 LogQueryService  ----
    level: Optional[str] = None
    service: Optional[str] = None
    keyword: Optional[str] = None
    request_id: Optional[str] = None
    limit: int = 20

    @model_validator(mode="after")
    def _end_after_start(self) -> LogQuery:
        if self.end_time <= self.start_time:
            raise ValueError("end_time must be greater than start_time")
        return self


class LogEvent(BaseModel):
    """
    structured log event object, parsed from CW Insights result row

    Insights 返回的是 field/value 列表；若 @message 是 JSON 且查询里选了字段，
    level/service 等可能已作为独立列展开。解析不到的字段保持 None，不臆造。
    """

    timestamp: Optional[datetime] = Field(
        default=None,
        description="优先取 @timestamp；解析失败时可为 None",
    )
    level: Optional[str] = None
    service: Optional[str] = None
    request_id: Optional[str] = None
    message: Optional[str] = Field(
        default=None,
        description="业务正文；也可能来自 @message 整段 JSON 字符串",
    )
    # 保留整行原始字段映射，便于调试与后续扩展（error_code 等）
    raw: dict[str, Any] = Field(default_factory=dict)
