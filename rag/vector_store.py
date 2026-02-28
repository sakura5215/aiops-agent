"""
向量库服务：通过 adapter 抽象屏蔽底层差异，按 config/vector_store.yml 的 provider
字段在 Chroma / Milvus Lite 之间切换。知识库加载沿用 MD5 去重逻辑。
"""
import os

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.config_handler import vector_conf
from model.factory import embed_model
from utils.logger_handler import logger
from utils.path_tool import get_abs_path
from utils.file_handler import pdf_loader, txt_loader, listdir_with_allowed_type, get_file_md5_hex


class VectorStoreBackend:
    """向量库后端抽象接口，Chroma / Milvus 等具体实现继承此类。"""

    def as_retriever(self, k: int):
        raise NotImplementedError

    def add_documents(self, documents: list[Document]) -> None:
        raise NotImplementedError


class ChromaBackend(VectorStoreBackend):
    """基于 langchain-chroma 的本地持久化向量库后端。"""

    def __init__(self):
        from langchain_chroma import Chroma
        self.store = Chroma(
            collection_name=vector_conf["collection_name"],
            embedding_function=embed_model,
            persist_directory=get_abs_path(vector_conf["chroma_persist_directory"]),
        )

    def as_retriever(self, k: int):
        return self.store.as_retriever(search_kwargs={"k": k})

    def add_documents(self, documents: list[Document]) -> None:
        self.store.add_documents(documents)


class MilvusBackend(VectorStoreBackend):
    """基于 Milvus Lite 的向量库后端。Milvus Lite 以单文件 .db 形式落地，无需独立部署服务端。"""

    def __init__(self):
        from langchain_milvus import Milvus
        uri = get_abs_path(vector_conf["milvus_uri"])
        os.makedirs(os.path.dirname(uri), exist_ok=True)
        self.store = Milvus(
            embedding_function=embed_model,
            collection_name=vector_conf["collection_name"],
            connection_args={"uri": uri},
        )

    def as_retriever(self, k: int):
        return self.store.as_retriever(search_kwargs={"k": k})

    def add_documents(self, documents: list[Document]) -> None:
        self.store.add_documents(documents)


def build_backend() -> VectorStoreBackend:
    provider = str(vector_conf.get("provider", "milvus")).lower()
    if provider == "chroma":
        return ChromaBackend()
    if provider == "milvus":
        return MilvusBackend()
    raise ValueError(
        f"不支持的向量库 provider: {provider}，请在 config/vector_store.yml 中配置 chroma 或 milvus"
    )


class VectorStoreService:
    def __init__(self):
        self.backend = build_backend()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=vector_conf["chunk_size"],
            chunk_overlap=vector_conf["chunk_overlap"],
            separators=vector_conf["separators"],
            length_function=len,
        )

    def get_retriever(self):
        return self.backend.as_retriever(k=vector_conf["k"])

    def load_document(self):
        """
        从数据文件夹内读取数据文件，转为向量存入向量库。
        计算文件 MD5 做去重，避免重复向量化。
        """

        def check_md5_hex(md5_for_check: str):
            md5_store = get_abs_path(vector_conf["md5_hex_store"])
            if not os.path.exists(md5_store):
                open(md5_store, "w", encoding="utf-8").close()
                return False
            with open(md5_store, "r", encoding="utf-8") as f:
                for line in f.readlines():
                    if line.strip() == md5_for_check:
                        return True
                return False

        def save_md5_hex(md5_for_check: str):
            with open(get_abs_path(vector_conf["md5_hex_store"]), "a", encoding="utf-8") as f:
                f.write(md5_for_check + "\n")

        def get_file_documents(read_path: str):
            if read_path.endswith("txt"):
                return txt_loader(read_path)
            if read_path.endswith("pdf"):
                return pdf_loader(read_path)
            return []

        allowed_files_path: list[str] = listdir_with_allowed_type(
            get_abs_path(vector_conf["data_path"]),
            tuple(vector_conf["allow_knowledge_file_type"]),
        )

        for path in allowed_files_path:
            md5_hex = get_file_md5_hex(path)

            if check_md5_hex(md5_hex):
                logger.info(f"[加载知识库]{path}内容已经存在于数据库中，跳过")
                continue

            try:
                documents: list[Document] = get_file_documents(path)

                if not documents:
                    logger.warning(f"[加载知识库]{path}内没有有效文本内容，跳过")
                    continue

                split_document: list[Document] = self.splitter.split_documents(documents)

                if not split_document:
                    logger.warning(f"[加载知识库]{path}分片后没有有效文本内容，跳过")
                    continue

                self.backend.add_documents(split_document)
                save_md5_hex(md5_hex)
                logger.info(f"[加载知识库]{path}内容成功")

            except Exception as e:
                logger.error(f"[加载知识库]{path}加载失败，{str(e)}", exc_info=True)
                continue


if __name__ == '__main__':
    vs = VectorStoreService()
    vs.load_document()
    retriever = vs.get_retriever()
    res = retriever.invoke("迷路")
    for r in res:
        print(r.page_content)
        print("-" * 20)
