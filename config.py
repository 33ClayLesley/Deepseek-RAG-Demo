from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr

class AppConfig(BaseSettings):
    llm_api_key: SecretStr
    llm_base_url: str = "https://api.deepseek.com"
    llm_model: str = "deepseek-chat"
    llm_temperature: float = 0.1

    # 这里改为 bge 中文模型，和 .env 保持一致
    embedding_model: str = "BAAI/bge-small-zh-v1.5" 
    chunk_size: int = 500
    chunk_overlap: int = 100
    retriever_fetch_k: int = 20  # 粗排召回数量
    retriever_k: int = 3         # 精排后保留数量
    chroma_persist_dir: str = "./chroma_db"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

# 单例模式 保证整个项目在from config import settings时都是得到同一个实例
settings = AppConfig()