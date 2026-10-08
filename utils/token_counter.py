"""精确 token 计数器：用 Qwen3 原生 tokenizer 替换 GPT-2 fallback。

背景：项目 chat 模型是 qwen3-max（DashScope 闭源 API）。Qwen3 全系（含 qwen3-max）
共享同一套 tokenizer 词表（vocab_size=151669），因此用 Qwen3 开源权重导出的
tokenizer.json 即可对 qwen3-max 输入做「精确」计数，而非 GPT-2 的近似值。

实现选择（绕开两个已知坑）：
1. 不用 transformers.AutoTokenizer——它加载前必须先读 config.json，而 hf-mirror
   对该文件偶发返回空响应（0 字节）导致 OSError；且 import 会触发 torch 缺失告警。
2. 不用 huggingface_hub 下载——它读系统代理（Clash 7890 关闭时）会下到 0 字节。
   改为 urllib 直连 + 手动写盘，避免代理干扰。

因此直接用底层 `tokenizers` 库（已随 transformers 安装）加载 tokenizer.json：
- 从 hf-mirror（国内可达）下载 tokenizer.json（约 11MB，含完整词表），
- 缓存到项目内 .cache/qwen3_tokenizer.json（已 gitignore，不入库），
- 单例 + 惰性加载：只在首次用到时下载/加载一次，循环内复用。

加载失败（无网络 / 未装 tokenizers）时回退 GPT-2 近似计数，并显式标记 degraded，
让上层可感知（真实容错，而非「假装精确」）。
"""
from __future__ import annotations

import os
import threading
import urllib.request
from functools import lru_cache
from pathlib import Path

from utils.logger_handler import logger

# Qwen3 全系共享词表；用最小开源权重 0.6B 的 tokenizer.json，与 qwen3-max 完全一致。
_QWEN3_REPO = "Qwen/Qwen3-0.6B"
# 下载源：默认 hf-mirror（国内可达，无需代理）；可用环境变量 HF_ENDPOINT 覆盖为官方源。
_DEFAULT_ENDPOINT = os.environ.get("HF_ENDPOINT") or "https://hf-mirror.com"
_TOKENIZER_URL = f"{_DEFAULT_ENDPOINT.rstrip('/')}/{_QWEN3_REPO}/resolve/main/tokenizer.json"

# 缓存路径：项目内 .cache/，随 gitignore 忽略，避免污染仓库
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CACHE_DIR = _PROJECT_ROOT / ".cache"
_CACHE_FILE = _CACHE_DIR / "qwen3_tokenizer.json"

_lock = threading.Lock()
_tokenizer = None
_load_attempted = False
_degraded = False


def _download_tokenizer() -> Path | None:
    """用 urllib 直连下载 tokenizer.json（绕开代理），返回本地路径或 None。

    已存在且非空的缓存直接复用，不重复下载。
    """
    if _CACHE_FILE.exists() and _CACHE_FILE.stat().st_size > 0:
        return _CACHE_FILE
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        # 下载到临时文件，成功后再原子改名，避免半截文件被当成有效缓存
        tmp = _CACHE_FILE.with_suffix(".json.download")
        # 构造一个无代理的 opener：Windows 下系统代理（Clash 7890 关闭）会导致下到 0 字节
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(_TOKENIZER_URL, timeout=60) as resp, open(tmp, "wb") as f:
            while True:
                chunk = resp.read(1024 * 256)
                if not chunk:
                    break
                f.write(chunk)
        if tmp.stat().st_size == 0:
            tmp.unlink(missing_ok=True)
            raise RuntimeError("下载到空文件（可能被代理/镜像截断）")
        tmp.replace(_CACHE_FILE)
        logger.info(f"[token_counter]已缓存 Qwen3 tokenizer: {_CACHE_FILE}")
        return _CACHE_FILE
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[token_counter]tokenizer 下载失败: {e}")
        return None


def _try_load_qwen3_tokenizer():
    """加载 Qwen3 tokenizer（单例、惰性）。返回 tokenizer 或 None。"""
    global _tokenizer, _load_attempted, _degraded
    with _lock:
        if _load_attempted:
            return _tokenizer
        _load_attempted = True
        try:
            from tokenizers import Tokenizer

            path = _download_tokenizer()
            if path is None:
                raise RuntimeError("tokenizer.json 不可用")
            tok = Tokenizer.from_file(str(path))
            _tokenizer = tok
            _degraded = False
            logger.info(
                f"[token_counter]已加载 Qwen3 精确 tokenizer "
                f"(vocab={tok.get_vocab_size()})"
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                f"[token_counter]Qwen3 tokenizer 加载失败，回退 GPT-2 近似计数: {e}"
            )
            _tokenizer = None
            _degraded = True
        return _tokenizer


def is_degraded() -> bool:
    """当前计数是否处于降级状态（GPT-2 近似），供上层感知与决策。"""
    return _degraded if _load_attempted else False


@lru_cache(maxsize=4096)
def count_tokens(text: str) -> int:
    """返回 text 的精确 token 数（Qwen3 词表）；降级时用 GPT-2 近似并标记 degraded。

    lru_cache 缓存计数结果：循环内同一段内容重复计数时命中缓存，避免重复 encode。
    """
    tok = _try_load_qwen3_tokenizer()
    if tok is not None:
        return len(tok.encode(text).ids)
    # GPT-2 fallback（约 0.5 字/token 的粗略近似；仅降级场景使用）
    return max(1, len(text) // 2)


def reset_for_tests() -> None:
    """测试专用：清空缓存与加载状态，便于注入/重置。"""
    global _tokenizer, _load_attempted, _degraded
    with _lock:
        _tokenizer = None
        _load_attempted = False
        _degraded = False
    count_tokens.cache_clear()
