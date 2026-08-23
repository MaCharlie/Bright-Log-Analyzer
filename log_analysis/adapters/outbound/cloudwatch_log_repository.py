"""
CloudWatch Logs Insights 出站适配器。

Call Sequence:
    1. start_query(...)  → to get queryId
    2. loop get_query_results(queryId)，until status turns into Complete / Failed / Cancelled
    3. timeout -> stop_query，raise LogBackendError
    4. parse results field/value into list of LogEvent


"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any, Mapping, Optional

import boto3
from botocore.client import BaseClient
from botocore.exceptions import ClientError

from log_analysis.config import Settings, get_settings
from log_analysis.domain.errors import LogBackendError, QuerySyntaxError
from log_analysis.domain.models import LogEvent, LogQuery
from log_analysis.ports.outbound.log_repository import LogRepository

# when query fails, these codes often signify invalid query parameter or invalid syntax
_SYNTAX_ERROR_CODES = frozenset(
    {
        "InvalidParameterException",
        "MalformedQueryException",
        "InvalidParameter",
    }
)

# get_query_results termination status: defined by cw threads
_TERMINAL_STATUSES = frozenset({"Complete", "Failed", "Cancelled", "Timeout"})


class CloudWatchLogRepository(LogRepository):


    def __init__(
        self,
        settings: Optional[Settings] = None,
        client: Optional[BaseClient] = None,
        *,
        # 可注入 clock/sleep，便于单测控制「超时」而不真的 sleep 60 秒
        sleep_fn=time.sleep,
        monotonic_fn=time.monotonic,
    ) -> None:
        self._settings = settings or get_settings()
        session = boto3.Session(profile_name=self._settings.profile_name)
        self._client = client or session.client(
            "logs",
            region_name=self._settings.aws_region,
        )
        self._sleep = sleep_fn
        self._monotonic = monotonic_fn

    # ------------------------------------------------------------------
    # LogRepository 实现
    # ------------------------------------------------------------------

    def query_logs(self, query: LogQuery) -> list[LogEvent]:
        """

        core process.
        execute Insights query and return LogEvent list

        Phase 1 constraint：the input param must be cw pipeline statement
        """
        pipeline = (query.raw_pipeline or "").strip()
        if not pipeline:
            raise QuerySyntaxError(
                "raw_pipeline is required in Phase 1; "
                "pass a CloudWatch Logs Insights query string "
                "(structured LogQuery fields are assembled in Phase 2).",
                query=pipeline,
            )

        log_group = query.log_group or self._settings.cw_log_group
        start_epoch = _to_epoch_seconds(query.start_time) # convert datetime to epoch seconds(for cw insights)
        end_epoch = _to_epoch_seconds(query.end_time)

        # 1. start_query to get query_id
        query_id = self._start_query(
            log_group=log_group,
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            pipeline=pipeline,
        )
        # 2. poll to get query results until termination or timeout
        raw_response = self._poll_until_done(query_id=query_id, pipeline=pipeline)

        # 3. object transform: parse raw results to LogEvent list
        return _parse_insight_results(raw_response.get("results") or [])


    def _start_query(
        self,
        *,
        log_group: str,
        start_epoch: int,
        end_epoch: int,
        pipeline: str,
    ) -> str:
        """
        call boto3 client's StartQuery
        :return query_id
        """
        try:
            response = self._client.start_query(
                logGroupName=log_group,
                startTime=start_epoch,
                endTime=end_epoch,
                queryString=pipeline,
            )
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            msg = exc.response.get("Error", {}).get("Message", str(exc))

            """ core process: when syntax error, reserve the original msg for agent to self-correct
            grammar check by CW Insights will happen in this stage
            """
            if code in _SYNTAX_ERROR_CODES:
                raise QuerySyntaxError(message=msg, query=pipeline) from exc
            raise LogBackendError(f"start_query failed [{code}]: {msg}") from exc

        query_id = response.get("queryId")
        if not query_id:
            raise LogBackendError("start_query returned empty queryId")
        return query_id

    def _poll_until_done(self, *, query_id: str, pipeline: str) -> dict[str, Any]:
        """
        Poll to GetQueryResults，until finished or main threads timeout

        """
        deadline = self._monotonic() + self._settings.cw_query_timeout_sec
        last: dict[str, Any] = {}

        while True:
            try:
                last = self._client.get_query_results(queryId=query_id)
            except ClientError as exc:
                code = exc.response.get("Error", {}).get("Code", "")
                msg = exc.response.get("Error", {}).get("Message", str(exc))

                # # not necessary.
                # if code in _SYNTAX_ERROR_CODES:
                #     raise QuerySyntaxError(message=msg, query=pipeline) from exc
                raise LogBackendError(f"get_query_results failed [{code}]: {msg}") from exc

            status = last.get("status") or ""
            if status in _TERMINAL_STATUSES:
                return self._handle_terminal(status=status, response=last, pipeline=pipeline)

            # (main threads) timeout：try to stop_query，then raise LogBackendError
            if self._monotonic() >= deadline:
                self._safe_stop_query(query_id)
                raise LogBackendError(
                    f"CloudWatch Insights query timed out after "
                    f"{self._settings.cw_query_timeout_sec}s (queryId={query_id})"
                )

            # sleep for 1 second, to prevent over-frequently calling
            self._sleep(self._settings.cw_poll_interval_sec)

    def _handle_terminal(
        self,
        *,
        status: str,
        response: Mapping[str, Any],
        pipeline: str,
    ) -> dict[str, Any]:
        """
        handle the terminal process according to the defined termination status
        """
        if status == "Complete":
            return dict(response)

        # Failed / Cancelled / Timeout(cw threads timeout)
        detail_parts = [f"status={status}"]
        # statistics 里有时有 recordsScanned 等，对排错有用
        stats = response.get("statistics")
        if stats:
            detail_parts.append(f"statistics={stats}")
        # 部分 SDK 响应可能带 message 字段（并非总有）
        if response.get("message"):
            detail_parts.append(str(response["message"]))

        detail = "; ".join(detail_parts)
        if status == "Failed":
            # 对 Agent 来说「改语句再试」是合理默认；若实际是权限问题，
            # 错误原文里通常能看出来，仍通过 message 透出。
            raise QuerySyntaxError(
                message=f"CloudWatch Insights query failed ({detail})",
                query=pipeline,
            )
        raise LogBackendError(f"CloudWatch Insights query ended ({detail})")

    def _safe_stop_query(self, query_id: str) -> None:
        """超时后尽力取消查询，避免继续计费扫描；失败只忽略。"""
        try:
            self._client.stop_query(queryId=query_id)
        except ClientError:
            pass


# ----------------------------------------------------------------------
# 纯函数：时间与结果解析（便于单测、无 AWS 依赖）
# ----------------------------------------------------------------------


def _to_epoch_seconds(dt: datetime) -> int:
    """
    parse datetime into Unix epoch that StartQuery needs
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def _parse_insight_results(rows: list[list[dict[str, str]]]) -> list[LogEvent]:
    """
    parse GetQueryResults results (CW Insights original JSON format) into list of LogEvent。

    CW Insights results like：
        [{"field": "@timestamp", "value": "..."}, {"field": "level", "value": "ERROR"}, ...]

    若查询未 fields 展开 JSON 键、但 @message 是约定中的单行 JSON，
    会用 JSON 内的 level/service/request_id/message 补全空缺（不覆盖已展开列）。
    """
    events: list[LogEvent] = []
    for row in rows:
        field_map: dict[str, Any] = {}
        for cell in row:
            name = cell.get("field")
            if not name:
                continue
            field_map[name] = cell.get("value")

        json_fields = _try_parse_message_json(field_map.get("@message"))

        def pick(*keys: str) -> Optional[str]:
            for key in keys:
                if field_map.get(key) is not None:
                    return _as_optional_str(field_map.get(key))
            for key in keys:
                if json_fields.get(key) is not None:
                    return _as_optional_str(json_fields.get(key))
            return None

        events.append(
            LogEvent(
                timestamp=_parse_timestamp(field_map.get("@timestamp")),
                level=pick("level"),
                service=pick("service"),
                request_id=pick("request_id", "requestId"),
                message=pick("message") or (
                    str(field_map["@message"])
                    if field_map.get("@message") is not None
                    else None
                ),
                raw=field_map,
            )
        )
    return events


def _try_parse_message_json(raw_message: Any) -> dict[str, Any]:
    """尝试把 @message 解析为 dict；失败则返回空 dict。"""
    if raw_message is None:
        return {}
    try:
        payload = json.loads(str(raw_message))
    except (TypeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _parse_timestamp(value: Any) -> Optional[datetime]:
    """解析 Insights 的 @timestamp（常见为 'YYYY-MM-DD HH:MM:SS.mmm'）。"""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value

    text = str(value).strip()
    # Insights 常见格式；带 Z 或不带毫秒都试一下
    for fmt in (
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S%z",
    ):
        try:
            dt = datetime.strptime(text, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue

    # 有时是 epoch 毫秒/秒字符串
    try:
        num = float(text)
        if num > 1e12:  # 毫秒
            num /= 1000.0
        return datetime.fromtimestamp(num, tz=timezone.utc)
    except (TypeError, ValueError):
        return None


def _as_optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(value)
