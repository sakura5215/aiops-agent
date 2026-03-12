import random

from langchain_core.tools import tool
from rag.rag_service import RagSummarizeService
from utils.logger_handler import logger

rag = RagSummarizeService()

# 注：这里没有用到data/records.csv中的数据，不需要提取数据，而是直接写在这里
service_list = [
    "order-service",
    "payment-service",
    "inventory-service",
    "user-service",
    "gateway-service",
]

time_range_list = [
    "最近1小时",
    "最近24小时",
    "今天",
    "本周",
    "本月",
]

mock_alert_data = {
    "order-service": {
        "最近1小时": {
            "critical": 2,
            "warning": 4,
            "alerts": [
                "订单服务 5xx 错误率升高",
                "订单服务平均响应时间超过阈值",
                "订单服务 Pod 重启次数异常",
            ],
        },
        "今天": {
            "critical": 5,
            "warning": 9,
            "alerts": [
                "订单服务 5xx 错误率升高",
                "订单服务数据库连接池使用率过高",
                "订单服务平均响应时间超过阈值",
            ],
        },
        "本月": {
            "critical": 18,
            "warning": 37,
            "alerts": [
                "订单服务 5xx 错误率升高",
                "订单服务数据库连接池使用率过高",
                "订单服务平均响应时间超过阈值",
                "订单服务 CPU 利用率高",
            ],
        },
    },
    "payment-service": {
        "最近1小时": {
            "critical": 1,
            "warning": 3,
            "alerts": [
                "支付服务交易失败率升高",
                "支付服务响应延迟升高",
            ],
        },
        "今天": {
            "critical": 3,
            "warning": 6,
            "alerts": [
                "支付服务交易失败率升高",
                "支付服务调用第三方接口超时",
            ],
        },
        "本月": {
            "critical": 11,
            "warning": 24,
            "alerts": [
                "支付服务交易失败率升高",
                "支付服务调用第三方接口超时",
                "支付服务 JVM Old 区使用率高",
            ],
        },
    },
    "inventory-service": {
        "最近1小时": {
            "critical": 0,
            "warning": 2,
            "alerts": [
                "库存服务查询延迟升高",
            ],
        },
        "今天": {
            "critical": 1,
            "warning": 5,
            "alerts": [
                "库存服务查询延迟升高",
                "库存服务缓存命中率下降",
            ],
        },
        "本月": {
            "critical": 7,
            "warning": 16,
            "alerts": [
                "库存服务查询延迟升高",
                "库存服务缓存命中率下降",
                "库存服务数据库慢查询增多",
            ],
        },
    },
}
mock_metric_data = {
    "order-service": {
        "最近1小时": {
            "cpu_usage": "83%",
            "memory_usage": "78%",
            "qps": "1260",
            "rt": "428ms",
            "error_rate": "3.8%",
            "pod_restart_count": 3,
        },
        "今天": {
            "cpu_usage": "76%",
            "memory_usage": "73%",
            "qps": "1180",
            "rt": "392ms",
            "error_rate": "2.9%",
            "pod_restart_count": 8,
        },
        "本月": {
            "cpu_usage": "68%",
            "memory_usage": "70%",
            "qps": "1025",
            "rt": "355ms",
            "error_rate": "2.1%",
            "pod_restart_count": 19,
        },
    },
    "payment-service": {
        "最近1小时": {
            "cpu_usage": "65%",
            "memory_usage": "69%",
            "qps": "730",
            "rt": "516ms",
            "error_rate": "2.7%",
            "pod_restart_count": 1,
        },
        "今天": {
            "cpu_usage": "61%",
            "memory_usage": "66%",
            "qps": "702",
            "rt": "481ms",
            "error_rate": "2.2%",
            "pod_restart_count": 2,
        },
        "本月": {
            "cpu_usage": "58%",
            "memory_usage": "63%",
            "qps": "680",
            "rt": "455ms",
            "error_rate": "1.9%",
            "pod_restart_count": 7,
        },
    },
    "inventory-service": {
        "最近1小时": {
            "cpu_usage": "49%",
            "memory_usage": "57%",
            "qps": "845",
            "rt": "287ms",
            "error_rate": "1.1%",
            "pod_restart_count": 0,
        },
        "今天": {
            "cpu_usage": "52%",
            "memory_usage": "59%",
            "qps": "812",
            "rt": "301ms",
            "error_rate": "1.3%",
            "pod_restart_count": 1,
        },
        "本月": {
            "cpu_usage": "47%",
            "memory_usage": "55%",
            "qps": "790",
            "rt": "276ms",
            "error_rate": "1.0%",
            "pod_restart_count": 2,
        },
    },
}

mock_log_data = {
    "order-service": {
        "最近1小时": {
            "top_errors": [
                "java.sql.SQLTransientConnectionException: Connection is not available",
                "HTTP 500 Internal Server Error",
                "Timeout while calling inventory-service",
            ],
            "summary": "最近1小时内订单服务出现数据库连接获取超时、下游库存服务调用超时，以及少量 500 错误，请重点排查数据库连接池和下游依赖可用性。",
        },
        "今天": {
            "top_errors": [
                "java.sql.SQLTransientConnectionException: Connection is not available",
                "Timeout while calling inventory-service",
                "RejectedExecutionException",
            ],
            "summary": "今日订单服务主要异常集中在数据库连接池紧张、线程池拒绝任务以及调用下游库存服务超时。",
        },
        "本月": {
            "top_errors": [
                "Connection pool exhausted",
                "Timeout while calling inventory-service",
                "HTTP 500 Internal Server Error",
            ],
            "summary": "本月订单服务异常主要由高峰期连接池耗尽、下游依赖波动和应用内部 500 错误构成。",
        },
    },
    "payment-service": {
        "最近1小时": {
            "top_errors": [
                "Third-party payment gateway timeout",
                "HTTP 502 Bad Gateway",
            ],
            "summary": "最近1小时支付服务主要异常为调用第三方支付接口超时，伴随少量网关层 502 错误。",
        },
        "今天": {
            "top_errors": [
                "Third-party payment gateway timeout",
                "SocketTimeoutException",
                "HTTP 502 Bad Gateway",
            ],
            "summary": "今日支付服务异常主要与第三方支付通道响应不稳定有关。",
        },
        "本月": {
            "top_errors": [
                "Third-party payment gateway timeout",
                "SocketTimeoutException",
                "CircuitBreakerOpenException",
            ],
            "summary": "本月支付服务主要问题为外部接口稳定性不足及熔断触发。",
        },
    },
    "inventory-service": {
        "最近1小时": {
            "top_errors": [
                "Redis cache miss ratio increased",
                "Slow SQL detected",
            ],
            "summary": "最近1小时库存服务以缓存命中下降和数据库慢查询为主。",
        },
        "今天": {
            "top_errors": [
                "Redis cache miss ratio increased",
                "Slow SQL detected",
                "Read timeout from MySQL",
            ],
            "summary": "今日库存服务问题主要表现为缓存效果下降及慢查询增加。",
        },
        "本月": {
            "top_errors": [
                "Slow SQL detected",
                "Redis cache miss ratio increased",
                "Read timeout from MySQL",
            ],
            "summary": "本月库存服务问题整体以数据库慢查询与缓存命中率下降为主。",
        },
    },
}

mock_topology_data = {
    "order-service": {
        "upstream": ["gateway-service"],
        "downstream": ["inventory-service", "payment-service", "user-service"],
        "middleware": ["MySQL", "Redis", "Kafka"],
    },
    "payment-service": {
        "upstream": ["gateway-service", "order-service"],
        "downstream": ["third-party-payment-gateway"],
        "middleware": ["MySQL", "Redis"],
    },
    "inventory-service": {
        "upstream": ["order-service", "gateway-service"],
        "downstream": [],
        "middleware": ["MySQL", "Redis"],
    },
    "user-service": {
        "upstream": ["gateway-service", "order-service"],
        "downstream": [],
        "middleware": ["MySQL", "Redis"],
    },
    "gateway-service": {
        "upstream": [],
        "downstream": ["order-service", "payment-service", "inventory-service", "user-service"],
        "middleware": ["Nginx"],
    },
}


# -----------------------------
# RAG 工具：保留
# -----------------------------
@tool(description="向向量存储中检索参考资料")
def rag_summarize(query: str) -> str:
    return rag.rag_summarize(query)


# -----------------------------
# 基础上下文工具
# -----------------------------
@tool(description="获取当前待分析的目标服务名称，以纯字符串形式返回")
def get_target_service() -> str:
    return random.choice(service_list)


@tool(description="获取当前分析任务的默认时间范围，以纯字符串形式返回，例如最近1小时、今天、本周、本月")
def get_time_range() -> str:
    return random.choice(time_range_list)


# -----------------------------
# 告警 / 指标 / 日志 / 拓扑工具
# -----------------------------
@tool(description="获取指定服务在指定时间范围内的告警信息，以结构化字符串形式返回；若未检索到数据则返回空字符串")
def fetch_alert_data(service_name: str, time_range: str) -> str:
    try:
        data = mock_alert_data[service_name][time_range]
        return str(data)
    except KeyError:
        logger.warning(f"[fetch_alert_data] 未检索到服务 {service_name} 在 {time_range} 的告警数据")
        return ""


@tool(description="获取指定服务在指定时间范围内的关键监控指标，以结构化字符串形式返回；若未检索到数据则返回空字符串")
def fetch_metric_data(service_name: str, time_range: str) -> str:
    try:
        data = mock_metric_data[service_name][time_range]
        return str(data)
    except KeyError:
        logger.warning(f"[fetch_metric_data] 未检索到服务 {service_name} 在 {time_range} 的监控指标数据")
        return ""


@tool(description="获取指定服务在指定时间范围内的日志摘要与高频错误信息，以结构化字符串形式返回；若未检索到数据则返回空字符串")
def fetch_log_summary(service_name: str, time_range: str) -> str:
    try:
        data = mock_log_data[service_name][time_range]
        return str(data)
    except KeyError:
        logger.warning(f"[fetch_log_summary] 未检索到服务 {service_name} 在 {time_range} 的日志摘要数据")
        return ""


@tool(description="获取指定服务的上下游依赖关系与中间件拓扑信息，以结构化字符串形式返回；若未检索到数据则返回空字符串")
def fetch_service_topology(service_name: str) -> str:
    try:
        data = mock_topology_data[service_name]
        return str(data)
    except KeyError:
        logger.warning(f"[fetch_service_topology] 未检索到服务 {service_name} 的拓扑数据")
        return ""


# -----------------------------
# 报告工具
# -----------------------------
def generate_report_data():
    """
    预留函数：
    后续如果需要从真实外部系统拉取数据，
    可以在这里统一聚合 Prometheus / Elasticsearch / 告警平台等数据。
    """
    return None


@tool(description="获取指定服务在指定时间范围内的运行报告汇总数据，以结构化字符串形式返回；若未检索到数据则返回空字符串")
def fetch_report_data(service_name: str, time_range: str) -> str:
    generate_report_data()

    try:
        alert_data = mock_alert_data.get(service_name, {}).get(time_range, {})
        metric_data = mock_metric_data.get(service_name, {}).get(time_range, {})
        log_data = mock_log_data.get(service_name, {}).get(time_range, {})
        topology_data = mock_topology_data.get(service_name, {})

        if not alert_data and not metric_data and not log_data:
            logger.warning(f"[fetch_report_data] 未检索到服务 {service_name} 在 {time_range} 的报告数据")
            return ""

        report_data = {
            "service_name": service_name,
            "time_range": time_range,
            "alert_summary": alert_data,
            "metric_summary": metric_data,
            "log_summary": log_data,
            "topology_summary": topology_data,
        }
        return str(report_data)
    except Exception as e:
        logger.error(f"[fetch_report_data] 获取服务 {service_name} 在 {time_range} 的报告数据失败：{str(e)}")
        return ""


@tool(description="无入参，无返回值，调用后触发中间件自动为报告生成场景动态注入上下文信息，为后续提示词切换提供上下文信息")
def fill_context_for_report():
    return "fill_context_for_report已调用"