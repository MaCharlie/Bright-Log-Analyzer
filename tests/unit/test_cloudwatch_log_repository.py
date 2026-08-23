"""
CloudWatchLogRepository 单元测试（botocore Stubber，不访问真实 AWS）。

在项目根目录执行（PYTHONPATH=. 让解释器能 import 顶层包 log_analysis）：
    PYTHONPATH=. pytest tests/unit/test_cloudwatch_log_repository.py -v -m "not live"

可选 live 用例（需本机 AWS 凭证；默认用 -m \"not live\" 跳过）：
    PYTHONPATH=. pytest tests/unit/test_cloudwatch_log_repository.py -v -m live
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import boto3
import pytest
from botocore.stub import Stubber

from log_analysis.adapters.outbound.cloudwatch_log_repository import (
    CloudWatchLogRepository,
    _parse_insight_results,
)
from log_analysis.config import Settings
from log_analysis.domain.errors import LogBackendError, QuerySyntaxError
from log_analysis.domain.models import LogQuery

# ---------------------------------------------------------------------------
# 公共夹具
# ---------------------------------------------------------------------------

START = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
END = datetime(2024, 1, 1, 1, 0, 0, tzinfo=timezone.utc)
PIPELINE = 'fields @timestamp, level, service, message | filter level = "ERROR" | limit 5'
LOG_GROUP = "/loganalysis/app-logs"
QUERY_ID = "query-id-test-001"


def _settings(**overrides) -> Settings:
    base = dict(
        aws_region="ap-northeast-1",
        cw_log_group=LOG_GROUP,
        cw_query_timeout_sec=60,
        cw_poll_interval_sec=0,  # 单测不真睡
    )
    base.update(overrides)
    return Settings(**base)


def _query(**overrides) -> LogQuery:
    data = dict(
        log_group=LOG_GROUP,
        start_time=START,
        end_time=END,
        raw_pipeline=PIPELINE,
    )
    data.update(overrides)
    return LogQuery(**data)


def _result_row(**fields: str) -> list[dict[str, str]]:
    return [{"field": k, "value": v} for k, v in fields.items()]


@pytest.fixture
def logs_client():
    session = boto3.Session(profile_name="log-analyzer-dev")
    client = session.client("logs", region_name="ap-northeast-1")
    with Stubber(client) as stubber:
        yield client, stubber


# ---------------------------------------------------------------------------
# 成功路径
# ---------------------------------------------------------------------------


def test_query_logs_completes_on_first_poll(logs_client):
    """轮询一次即为 Complete，应解析出 LogEvent。"""
    client, stubber = logs_client
    stubber.add_response(
        "start_query",
        {"queryId": QUERY_ID},
        {
            "logGroupName": LOG_GROUP,
            "startTime": int(START.timestamp()),
            "endTime": int(END.timestamp()),
            "queryString": PIPELINE,
        },
    )
    stubber.add_response(
        "get_query_results",
        {
            "status": "Complete",
            "results": [
                _result_row(
                    **{
                        "@timestamp": "2024-01-01 00:30:00.000",
                        "level": "ERROR",
                        "service": "OrderService",
                        "request_id": "req-123",
                        "message": "Timeout while calling PaymentGateway",
                    }
                )
            ],
        },
        {"queryId": QUERY_ID},
    )

    repo = CloudWatchLogRepository(settings=_settings(), client=client)
    events = repo.query_logs(_query())

    assert len(events) == 1
    assert events[0].level == "ERROR"
    assert events[0].service == "OrderService"
    assert events[0].request_id == "req-123"
    assert "Timeout" in (events[0].message or "")
    assert events[0].raw["level"] == "ERROR"
    stubber.assert_no_pending_responses()


def test_query_logs_running_then_complete(logs_client):
    """多次 Running 后 Complete；sleep 被调用。"""
    client, stubber = logs_client
    stubber.add_response(
        "start_query",
        {"queryId": QUERY_ID},
        {
            "logGroupName": LOG_GROUP,
            "startTime": int(START.timestamp()),
            "endTime": int(END.timestamp()),
            "queryString": PIPELINE,
        },
    )
    stubber.add_response(
        "get_query_results",
        {"status": "Running", "results": []},
        {"queryId": QUERY_ID},
    )
    stubber.add_response(
        "get_query_results",
        {"status": "Running", "results": []},
        {"queryId": QUERY_ID},
    )
    stubber.add_response(
        "get_query_results",
        {
            "status": "Complete",
            "results": [
                _result_row(
                    **{
                        "@timestamp": "2024-01-01 00:10:00.000",
                        "level": "INFO",
                        "service": "UserService",
                        "message": "ok",
                    }
                )
            ],
        },
        {"queryId": QUERY_ID},
    )

    sleep_fn = MagicMock()
    repo = CloudWatchLogRepository(
        settings=_settings(cw_poll_interval_sec=1.0),
        client=client,
        sleep_fn=sleep_fn,
    )
    events = repo.query_logs(_query())

    assert len(events) == 1
    assert events[0].service == "UserService"
    # Running → sleep → Running → sleep → Complete，共 2 次 sleep
    assert sleep_fn.call_count == 2
    stubber.assert_no_pending_responses()


# ---------------------------------------------------------------------------
# 超时
# ---------------------------------------------------------------------------


def test_query_logs_timeout_stops_query_and_raises(logs_client):
    """超时仍未 Complete：应 stop_query，并抛 LogBackendError。"""
    client, stubber = logs_client
    stubber.add_response(
        "start_query",
        {"queryId": QUERY_ID},
        {
            "logGroupName": LOG_GROUP,
            "startTime": int(START.timestamp()),
            "endTime": int(END.timestamp()),
            "queryString": PIPELINE,
        },
    )
    stubber.add_response(
        "get_query_results",
        {"status": "Running", "results": []},
        {"queryId": QUERY_ID},
    )
    stubber.add_response(
        "stop_query",
        {"success": True},
        {"queryId": QUERY_ID},
    )

    # 第 1 次 monotonic：deadline = 0 + timeout
    # 第 2 次 monotonic：轮询后判断 → 立刻超时
    ticks = iter([0.0, 1000.0])

    def fake_monotonic():
        return next(ticks)

    repo = CloudWatchLogRepository(
        settings=_settings(cw_query_timeout_sec=5, cw_poll_interval_sec=0),
        client=client,
        sleep_fn=lambda _: None,
        monotonic_fn=fake_monotonic,
    )

    with pytest.raises(LogBackendError, match="timed out"):
        repo.query_logs(_query())

    stubber.assert_no_pending_responses()


# ---------------------------------------------------------------------------
# 语法 / 参数错误
# ---------------------------------------------------------------------------


def test_start_query_invalid_parameter_raises_syntax_error(logs_client):
    """start_query 抛 InvalidParameterException → QuerySyntaxError，且含 AWS 原文。"""
    client, stubber = logs_client
    aws_msg = "Query syntax is invalid at character 42"
    stubber.add_client_error(
        "start_query",
        service_error_code="InvalidParameterException",
        service_message=aws_msg,
        http_status_code=400,
        expected_params={
            "logGroupName": LOG_GROUP,
            "startTime": int(START.timestamp()),
            "endTime": int(END.timestamp()),
            "queryString": PIPELINE,
        },
    )

    repo = CloudWatchLogRepository(settings=_settings(), client=client)

    with pytest.raises(QuerySyntaxError) as exc_info:
        repo.query_logs(_query())

    assert aws_msg in str(exc_info.value)
    assert exc_info.value.query == PIPELINE
    stubber.assert_no_pending_responses()


def test_failed_status_maps_to_query_syntax_error(logs_client):
    """status=Failed → QuerySyntaxError（保留 status 信息）。"""
    client, stubber = logs_client
    stubber.add_response(
        "start_query",
        {"queryId": QUERY_ID},
        {
            "logGroupName": LOG_GROUP,
            "startTime": int(START.timestamp()),
            "endTime": int(END.timestamp()),
            "queryString": PIPELINE,
        },
    )
    stubber.add_response(
        "get_query_results",
        {
            "status": "Failed",
            "results": [],
            "statistics": {"recordsMatched": 0.0, "recordsScanned": 0.0},
        },
        {"queryId": QUERY_ID},
    )

    repo = CloudWatchLogRepository(settings=_settings(), client=client)

    with pytest.raises(QuerySyntaxError) as exc_info:
        repo.query_logs(_query())

    assert "Failed" in str(exc_info.value)
    assert exc_info.value.query == PIPELINE
    stubber.assert_no_pending_responses()


def test_missing_raw_pipeline_raises_syntax_error(logs_client):
    """Phase 1 未提供 raw_pipeline 时应明确失败。"""
    client, stubber = logs_client
    repo = CloudWatchLogRepository(settings=_settings(), client=client)

    with pytest.raises(QuerySyntaxError, match="raw_pipeline is required"):
        repo.query_logs(_query(raw_pipeline=None))

    stubber.assert_no_pending_responses()


# ---------------------------------------------------------------------------
# 解析辅助
# ---------------------------------------------------------------------------


def test_parse_insight_results_from_json_message():
    """仅有 @message JSON 时，应从 JSON 补全 level/service/message 等。"""
    rows = [
        _result_row(
            **{
                "@timestamp": "2024-01-01 00:00:01.000",
                "@message": (
                    '{"level":"ERROR","service":"OrderService",'
                    '"request_id":"req-1","message":"boom"}'
                ),
            }
        )
    ]
    events = _parse_insight_results(rows)
    assert events[0].message == "boom"
    assert events[0].level == "ERROR"
    assert events[0].service == "OrderService"
    assert events[0].request_id == "req-1"


# ---------------------------------------------------------------------------
# 可选：真实 AWS（默认不跑）
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_live_describe_or_query():
    """
    需要本机已配置凭证。仅做连通性探测：describe-log-groups。
    """

    logRepo = CloudWatchLogRepository()
    client = logRepo._client

    groups = client.describe_log_groups(limit=5)
    print()

    for group in groups["logGroups"]:
        print(group)

    assert "logGroups" in groups
