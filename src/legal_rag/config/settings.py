"""Application settings, env-driven, with validation and secret masking."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed, validated settings for Legal Assist RAG.

    All values can be overridden via environment variables or a .env file.
    Secrets are stored as SecretStr and never appear in repr, str, or model_dump.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── LLM ──────────────────────────────────────────────────────────────
    llm_provider: str = Field(default="groq", description="LLM provider name")
    llm_model: str = Field(
        default="llama-3.3-70b-versatile",
        description="Model ID (verified against Groq production models Nov 2026)",
    )
    llm_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="Provider API key (required for LLM calls)",
    )
    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=2048, ge=1, le=131072)
    llm_timeout_seconds: float = Field(default=30.0, gt=0.0)
    llm_max_retries: int = Field(
        default=3, ge=0, le=10, description="Max retries for transient errors"
    )
    llm_retry_min_wait: float = Field(
        default=0.5, ge=0.0, description="Min backoff wait in seconds"
    )
    llm_retry_max_wait: float = Field(
        default=8.0, ge=0.0, description="Max backoff wait in seconds"
    )

    # ── Sessions ─────────────────────────────────────────────────────────
    session_ttl_seconds: int = Field(default=3600, ge=60, description="Session TTL in seconds")
    session_max_count: int = Field(
        default=1000, ge=1, description="Max concurrent sessions before eviction"
    )

    # ── Embedding (name only; model loaded in later phases) ──────────────
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence-transformer model name (baseline; bake-off in Phase 4)",
    )

    # ── Retrieval defaults ───────────────────────────────────────────────
    retrieval_top_k: int = Field(default=5, ge=1, le=100)
    retrieval_score_threshold: float = Field(default=0.0, ge=0.0, le=1.0)

    retrieval_candidates: int = Field(default=30, ge=1, le=500)
    retrieval_per_doc_cap: int = Field(default=3, ge=1, le=50)
    abstain_threshold: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Abstain if best dense score is below this"
    )
    chunk_strategy: str = Field(default="auto", description="auto|s0|s1|s2|s3")
    chunk_size: int = Field(default=800, ge=100, le=8000)
    chunk_overlap: int = Field(default=150, ge=0, le=2000)
    chunk_max_tokens: int = Field(default=256, ge=16, le=8192)
    contextual_prefix: bool = True
    low_text_chars: int = Field(default=50, ge=1)
    vector_store_backend: str = Field(default="memory", description="memory|qdrant")
    qdrant_url: str = Field(default="", description="Qdrant server URL; empty = local mode")
    qdrant_path: Path = Field(default=Path("data/processed/qdrant"))
    qdrant_collection: str = Field(default="legal_rag")
    embedding_cache_path: Path = Field(default=Path("data/cache/embeddings.sqlite"))
    judge_model: str = Field(default="", description="Pinned judge model id (required for --judge)")
    price_in_per_1k: float = Field(default=0.0, ge=0.0, description="USD per 1k prompt tokens")
    price_out_per_1k: float = Field(default=0.0, ge=0.0, description="USD per 1k output tokens")
    prompt_version: str = Field(default="v2")
    history_turns: int = Field(default=4, ge=0, le=20)

    # ── Paths ────────────────────────────────────────────────────────────
    data_dir: Path = Field(default=Path("data"))
    processed_dir: Path = Field(default=Path("data/processed"))
    eval_dir: Path = Field(default=Path("eval"))
    manifest_path: Path = Field(default=Path("data/manifest.yaml"))
    index_path: Path = Field(default=Path("data/processed/index.json"))
    mappings_path: Path = Field(default=Path("data/mappings/ipc_to_bns.json"))
    feedback_path: Path = Field(default=Path("data/processed/feedback.jsonl"))

    # ── API & Logging ────────────────────────────────────────────────────
    api_host: str = Field(default="127.0.0.1")
    api_port: int = Field(default=8000, ge=1, le=65535)
    cors_origins: list[str] = Field(default=["http://localhost:3000"])
    log_level: str = Field(default="INFO", description="Logging level")

    # ── Limits ───────────────────────────────────────────────────────────
    max_question_length: int = Field(default=1000, ge=10, le=10000)
    rate_limit_rpm: int = Field(default=60, ge=1)
    max_body_bytes: int = Field(default=1_000_000, ge=1024)
    admin_api_key: SecretStr = Field(default=SecretStr(""), description="Admin endpoint key")

    # ── Validators ───────────────────────────────────────────────────────

    @field_validator("llm_provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        allowed = {"groq"}
        if v.lower() not in allowed:
            msg = f"Unsupported LLM provider '{v}'. Allowed: {allowed}"
            raise ValueError(msg)
        return v.lower()

    @field_validator("vector_store_backend")
    @classmethod
    def validate_backend(cls, v: str) -> str:
        if v.lower() not in {"memory", "qdrant"}:
            msg = f"Invalid vector_store_backend '{v}'. Allowed: memory, qdrant"
            raise ValueError(msg)
        return v.lower()

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        val = v.upper()
        if val not in allowed:
            msg = f"Invalid log_level '{v}'. Allowed: {allowed}"
            raise ValueError(msg)
        return val

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: str | list[str]) -> list[str]:
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    @model_validator(mode="after")
    def validate_retry_wait_bounds(self) -> Settings:
        if self.llm_retry_min_wait > self.llm_retry_max_wait:
            msg = (
                f"llm_retry_min_wait ({self.llm_retry_min_wait}) cannot exceed "
                f"llm_retry_max_wait ({self.llm_retry_max_wait})"
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def validate_api_key_present(self) -> Settings:
        """Warn at construction time if the API key is empty."""
        if not self.llm_api_key.get_secret_value():
            import warnings

            warnings.warn(
                "LLM_API_KEY is empty. LLM calls will fail. "
                "Set the LLM_API_KEY environment variable.",
                UserWarning,
                stacklevel=2,
            )
        return self

    # ── Safe representation ──────────────────────────────────────────────

    def __repr__(self) -> str:
        safe = self.model_dump()
        return f"Settings({safe})"

    def __str__(self) -> str:
        return self.__repr__()

    def model_dump(self, **kwargs):  # type: ignore[override]
        """Override to always mask secrets in dumps."""
        data = super().model_dump(**kwargs)
        for key, value in data.items():
            if isinstance(value, SecretStr):
                data[key] = "**********"
            elif key.lower().endswith(("_key", "_secret", "_token", "_password")):
                data[key] = "**********"
        return data


def get_settings(**overrides) -> Settings:
    """Create a Settings instance, optionally with overrides for testing."""
    return Settings(**overrides)
