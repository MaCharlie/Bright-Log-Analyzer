from collections import defaultdict
import re

class LogPreprocessing(object):
    def __init__(self):
        super().__init__()


    def parse_logs(self, file_path: str):
        """
        将日志行按requestId进行分组
        :param file_path:
        :return:
        """
        # 按requestId分组存储日志行
        log_blocks = defaultdict(list)

        # 正则表达式，格式为[reqId:xxx]
        request_id_pattern = re.compile(r'\[reqId:(\S+)\]')
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # 搜寻search该模式。如果是match则为匹配
                req_id_match = request_id_pattern.search(line)
                if req_id_match:
                    req_id = req_id_match.group(1)
                    log_blocks[req_id].append(line)
                else:
                    # 没有reqId的日志，单独放入一个分类
                    log_blocks["NO_REQ_ID"].append(line)

        # 返回一个dict, key为requestId, value为分组后的日志行
        return log_blocks


    def clean_and_extract_metadata(self, log_line: str):
        """
        单行日志提取元数据
        """
        # 根据该模式进行解码
        pattern = re.compile(
            r'\[(?P<timestamp>.+?)\] '
            r'\[(?P<level>\w+)\] '
            r'(?:\[reqId:\S+\] )?'
            r'(?P<service>\w+): '
            r'(?P<message>.*)'
        )

        match = pattern.match(log_line)
        if not match:
            return None

        log_data = match.groupdict()

        return {
            "raw_log": log_line,
            "message": log_data["message"],
            "timestamp": log_data["timestamp"],
            "level": log_data["level"],
            "service": log_data["service"]
        }


    def build_log_block(self, request_id: str, log_lines: list):
        """
        对每个reqId对应的所有日志, 组装日志块, 生成用于embedding的文本.
        """

        # List<Map<>>, 每个reqId对应着一个dict格式
        cleaned_logs = []
        for line in log_lines:
            cleaned = self.clean_and_extract_metadata(line)
            if cleaned:
                cleaned_logs.append(cleaned)

        if not cleaned_logs:
            return None

        # 同一个reqId的日志按照时间戳排序
        cleaned_logs.sort(key=lambda log: log["timestamp"])
        # 将日志的message信息通过 | 分割开
        block_text = " | ".join([log["message"] for log in cleaned_logs])

        # 提取该请求的起始和结束时间
        first_log = cleaned_logs[0]
        last_log = cleaned_logs[-1]

        # 封装入元数据
        metadata = {
            "request_id": request_id,
            "start_time": first_log["timestamp"],
            "end_time": last_log["timestamp"],
            "services": list(set(log["service"] for log in cleaned_logs)), # 全部调用类
            "log_level": list(set(log["level"] for log in cleaned_logs)), # 全部日志级别
        }

        return {
            "metadata": metadata,
            "text_for_embedding": block_text,
            "raw_logs": [log["raw_log"] for log in cleaned_logs]
        }


    def process_log_file(self, file_path: str):
        """
        文件预处理流程串联
        :param file_path:
        :return:
        """

        # 1. 日志行按requestId分组
        grouped_logs = self.parse_logs(file_path)

        blocks = []
        for req_id, log_lines in grouped_logs.items():
            # 2. 将日志块进行分装，生成用于embedding的文本
            block = self.build_log_block(req_id, log_lines)
            if block:
                blocks.append(block)

        return blocks
