import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from langchain_core.embeddings import Embeddings
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models.tongyi import ChatTongyi, BaseChatModel
from utils.config_handler import rag_conf

# ---------------------------------------------------------------------------
# 凭据解析：密钥只在运行时从环境读取，绝不写进代码或 config/*.yml（仓库是公开的）
#   优先级：真实环境变量 > 项目根目录 .env（已被 gitignore）
# ---------------------------------------------------------------------------
CREDENTIAL_ENV = "DASHSCOPE_API_KEY"

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_dotenv() -> None:
    """加载项目根目录的 .env；未安装 python-dotenv 时静默跳过，退化为只用环境变量。"""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(_PROJECT_ROOT / ".env", override=False)


class CredentialError(RuntimeError):
    """密钥缺失。构造模型前就抛出，避免下游出现与配置无关的 SDK 报错。"""


def resolve_api_key() -> str:
    key = os.getenv(CREDENTIAL_ENV, "").strip()
    if not key:
        raise CredentialError(
            f"未检测到 {CREDENTIAL_ENV}，无法初始化通义千问模型。请选择任意一种方式配置：\n"
            f"  1) 项目根目录新建 .env（推荐，已加入 .gitignore，不会进版本库），参考 .env.example：\n"
            f"        {CREDENTIAL_ENV}=sk-xxxxxxxx\n"
            f"  2) 设置环境变量：\n"
            f"        Windows PowerShell : $env:DASHSCOPE_API_KEY=\"sk-xxxxxxxx\"\n"
            f"        Windows CMD        : set DASHSCOPE_API_KEY=sk-xxxxxxxx\n"
            f"        macOS / Linux      : export {CREDENTIAL_ENV}=sk-xxxxxxxx"
        )
    return key


class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        pass


class ChatModelFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return ChatTongyi(
            model=rag_conf["chat_model_name"],
            dashscope_api_key=resolve_api_key(),
        )


class EmbeddingsFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return DashScopeEmbeddings(
            model=rag_conf["embedding_model_name"],
            dashscope_api_key=resolve_api_key(),
        )


_load_dotenv()

chat_model = ChatModelFactory().generator()
embed_model = EmbeddingsFactory().generator()
