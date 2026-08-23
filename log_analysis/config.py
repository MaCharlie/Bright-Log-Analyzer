"""
应用配置（pydantic-settings）。

本 Phase 只关心 CloudWatch 查询所需项：区域、默认 Log Group、轮询超时与间隔。
凭证不写在这里——boto3 会自动读环境变量 / ~/.aws/credentials。

环境变量示例（可放在项目根 .env，勿提交密钥）：
    AWS_DEFAULT_REGION=ap-northeast-1
    CW_LOG_GROUP=/loganalysis/app-logs
    CW_QUERY_TIMEOUT_SEC=60
    CW_POLL_INTERVAL_SEC=1
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    load configuration from .env file

    Priorities: explicit calling > .env > default values(in this setting)
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # 忽略与本 Settings 无关的环境变量，避免启动失败
    )

    # CloudWatch API 所在区域；须与 Log Group 实际所在 Region 一致
    aws_region: str = "ap-northeast-1"

    profile_name: str = "log-analyzer-dev"

    # 默认查询的 Log Group；也可在每次 LogQuery 里显式传入覆盖
    cw_log_group: str = "/apps/light-log-analyzer/dev"

    # Insights 异步查询：轮询总超时（秒）。超时后会 stop_query 并抛 LogBackendError
    cw_query_timeout_sec: int = 60

    # 两次 get_query_results 之间的休眠秒数
    cw_poll_interval_sec: float = 1.0


@lru_cache
def get_settings() -> Settings:
    """进程内单例，避免反复解析环境变量。测试里可 Settings() 直接 new，或 clear 缓存。"""
    return Settings()
