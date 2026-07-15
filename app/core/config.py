from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "development"
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    database_url: str = "postgresql+asyncpg://sanrachna:sanrachna_secret@localhost:5432/sanrachna"
    database_url_sync: str = "postgresql://sanrachna:sanrachna_secret@localhost:5432/sanrachna"

    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "sanrachna_chunks"
    qdrant_upsert_batch_size: int = 64
    qdrant_payload_text_max_chars: int = 4000
    qdrant_payload_parent_max_chars: int = 512

    sarvam_api_key: str = ""
    sarvam_cost_per_page_inr: float = 0.5
    # auto = smart route + fallback chain (local/sarvam/mineru/vision)
    digitization_provider: str = "auto"
    mineru_enabled: bool = False
    mineru_bin: str = "mineru"
    mineru_backend: str = "pipeline"
    mineru_timeout_sec: int = 600
    vision_ocr_enabled: bool = True
    vision_ocr_model: str = "gemini-3.5-flash"
    vision_ocr_fallback_models: str = "gemini-2.5-flash,gpt-4o-mini"
    vision_ocr_timeout_sec: int = 120
    vision_ocr_max_pages: int = 20
    vision_ocr_render_scale: float = 2.0
    vision_ocr_image_detail: str = "high"
    tesseract_ocr_enabled: bool = True
    tesseract_ocr_lang: str = "eng"
    tesseract_ocr_max_pages: int = 50
    tesseract_ocr_render_scale: float = 2.0
    vision_ocr_concurrency: int = 5
    tesseract_ocr_concurrency: int = 4
    entity_extraction_concurrency: int = 1
    embed_batch_concurrency: int = 2
    image_caption_concurrency: int = 1
    ingest_cost_log_buffer: bool = True
    auto_translate_queries: bool = True

    litellm_base_url: str = "http://localhost:4000"
    litellm_api_key: str = "sk-sanrachna-master"
    litellm_embedding_model: str = "text-embedding-3-small"
    litellm_chat_model: str = "gpt-4o-mini"

    upload_dir: str = "./data/uploads"
    output_dir: str = "./data/outputs"

    chunk_size: int = 512
    chunk_overlap: int = 64
    child_chunk_size: int = 512
    parent_chunk_size: int = 2048
    sarvam_max_pages: int = 10
    enable_rerank: bool = True
    enable_cohere_rerank: bool = False
    rerank_model: str = "rerank-v3.5"
    rerank_top_n: int = 20
    cohere_api_key: str = ""
    gemini_api_key: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()
