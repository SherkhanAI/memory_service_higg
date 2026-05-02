from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://memory:memory@localhost:5432/memory"

    # OpenRouter is the unified gateway by default.
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # Direct-vendor keys (used only when a *_provider is "direct").
    google_api_key: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    jina_api_key: str = ""
    cohere_api_key: str = ""

    # Models — OpenRouter slugs by default.
    embedding_provider: str = "openrouter"
    embedding_model: str = "google/gemini-embedding-2-preview"
    embedding_dim: int = 1536

    extraction_provider: str = "openrouter"
    extraction_model: str = "openai/gpt-5.4-mini"

    reranker_provider: str = "jina"
    reranker_model: str = "jina-reranker-v3"

    memory_auth_token: str = ""
    log_level: str = "INFO"

    @property
    def has_embedding_key(self) -> bool:
        if self.embedding_provider == "openrouter":
            return bool(self.openrouter_api_key)
        return bool(self.google_api_key or self.openai_api_key)

    @property
    def has_extraction_key(self) -> bool:
        if self.extraction_provider == "openrouter":
            return bool(self.openrouter_api_key)
        return bool(self.openai_api_key or self.anthropic_api_key)

    @property
    def has_reranker_key(self) -> bool:
        if self.reranker_provider == "jina":
            return bool(self.jina_api_key)
        if self.reranker_provider == "cohere":
            return bool(self.cohere_api_key)
        if self.reranker_provider == "openrouter":
            return bool(self.openrouter_api_key)
        return False


settings = Settings()
