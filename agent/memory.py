# agent/memory.py
import os, json
from typing import Sequence
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict
from langchain_core.chat_history import BaseChatMessageHistory


class FileChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, session_id: str, storage_dir: str = "./chat_histories"):
        self.session_id = session_id
        self.file_path = os.path.join(storage_dir, f"{session_id}.json")
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

    def add_message(self, message: BaseMessage) -> None:       # 通常只添加单条
        all_msgs = list(self.messages)
        all_msgs.append(message)
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump([message_to_dict(m) for m in all_msgs], f, ensure_ascii=False, indent=2)

    @property
    def messages(self) -> list[BaseMessage]:
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return messages_from_dict(data)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def recent_messages(self, max_messages: int) -> list[BaseMessage]:
        """滑动窗口：返回最近 max_messages 条历史消息，用于控制喂给模型的上下文长度。
        全量历史仍完整持久化在文件中，仅截取尾部喂回，避免长对话 token 膨胀。"""
        msgs = self.messages
        if max_messages <= 0 or max_messages >= len(msgs):
            return msgs
        return msgs[-max_messages:]

    def clear(self) -> None:
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump([], f)