"""token 计数，用 Qwen3 的 tokenizer 数。

项目 chat 模型是 qwen3-max，Qwen3 这一系（含 qwen3-max）共用同一套词表
（vocab_size=151669），所以拿 Qwen3 开源权重导出的 tokenizer.json 就能对
qwen3-max 的输入数出准数。

所以直接用底层的 tokenizers 库加载 tokenizer.json：
- 存到项目里 .cache/qwen3_tokenizer.json（已 gitignore），
- 只加载一次，之后复用。

没网或没装 tokenizers 时才退回 GPT-2 估数，并记个日志。
"""
from __future__ import annotations

import os
import threading
import urllib.request
from functools import lru_cache
from pathlib import Path

from utils.logger_handler import logger

# Qwen3 这一系共用词表；拿最小的 0.6B 的 tokenizer.json 用，跟 qwen3-max 完全一致。
_QWEN3_REPO = "Qwen/Qwen3-0.6B"
# 下载源：默认 hf-mirror（国内能直连），想走官方源就设环境变量 HF_ENDPOINT。
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
    """用 urllib 直连下载 tokenizer.json，返回本地路径；失败返回 None。

    已经有缓存的非空文件就直接用，不重复下。
    """
    if _CACHE_FILE.exists() and _CACHE_FILE.stat().st_size > 0:
        return _CACHE_FILE
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        # 先下到临时文件，成功再改名，免得半截文件被当成有效缓存
        tmp = _CACHE_FILE.with_suffix(".json.download")
        # 用不带代理的 opener：Windows 下系统代理（Clash 没开）会把文件下成 0 字节
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
    """加载 Qwen3 tokenizer（只加载一次）。返回 tokenizer 或 None。"""
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
    """现在是不是在退回 GPT-2 估数的状态。"""
    return _degraded if _load_attempted else False


@lru_cache(maxsize=4096)
def count_tokens(text: str) -> int:
    """返回 text 的 token 数（Qwen3 词表）；退回 GPT-2 时用估数。

    lru_cache 缓存结果：循环里同一段内容重复数时直接命中，不用反复 encode。
    """
    tok = _try_load_qwen3_tokenizer()
    if tok is not None:
        return len(tok.encode(text).ids)
    # GPT-2 兜底估法：中文一个字约半个 token，很粗
    return max(1, len(text) // 2)


def reset_for_tests() -> None:
    """测试专用：清空缓存与加载状态，便于注入/重置。"""
    global _tokenizer, _load_attempted, _degraded
    with _lock:
        _tokenizer = None
        _load_attempted = False
        _degraded = False
    count_tokens.cache_clear()
