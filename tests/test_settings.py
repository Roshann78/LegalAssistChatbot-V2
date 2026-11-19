"""Tests for Settings: validation, env override, secret masking."""

from __future__ import annotations

import warnings

import pytest

from legal_rag.config.settings import Settings


class TestSettingsDefaults:
    """Default values load correctly without any env vars."""

    def test_default_provider(self):
        s = Settings(llm_api_key="test-key-1234")
        assert s.llm_provider == "groq"

    def test_default_model(self):
        s = Settings(llm_api_key="test-key-1234")
        assert s.llm_model == "llama-3.3-70b-versatile"

    def test_default_temperature(self):
        s = Settings(llm_api_key="test-key-1234")
        assert s.llm_temperature == 0.1

    def test_default_retrieval(self):
        s = Settings(llm_api_key="test-key-1234")
        assert s.retrieval_top_k == 5
        assert s.retrieval_score_threshold == 0.0

    def test_default_api(self):
        s = Settings(llm_api_key="test-key-1234")
        assert s.api_host == "127.0.0.1"
        assert s.api_port == 8000
        assert s.cors_origins == ["http://localhost:3000"]


class TestSettingsEnvOverride:
    """Settings can be overridden via constructor (simulating env vars)."""

    def test_override_model(self):
        s = Settings(llm_api_key="k", llm_model="llama-3.1-8b-instant")
        assert s.llm_model == "llama-3.1-8b-instant"

    def test_override_temperature(self):
        s = Settings(llm_api_key="k", llm_temperature=0.7)
        assert s.llm_temperature == 0.7

    def test_override_top_k(self):
        s = Settings(llm_api_key="k", retrieval_top_k=10)
        assert s.retrieval_top_k == 10

    def test_override_cors_string(self):
        s = Settings(llm_api_key="k", cors_origins="http://a.com, http://b.com")
        assert s.cors_origins == ["http://a.com", "http://b.com"]

    def test_override_cors_list(self):
        s = Settings(llm_api_key="k", cors_origins=["http://x.com"])
        assert s.cors_origins == ["http://x.com"]

    def test_override_via_env(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "env-key-value")
        monkeypatch.setenv("LLM_MODEL", "llama-3.1-8b-instant")
        monkeypatch.setenv("RETRIEVAL_TOP_K", "20")
        s = Settings()
        assert s.llm_api_key.get_secret_value() == "env-key-value"
        assert s.llm_model == "llama-3.1-8b-instant"
        assert s.retrieval_top_k == 20

    def test_blank_env_values_use_defaults(self, monkeypatch):
        monkeypatch.setenv("ABSTAIN_THRESHOLD", "")
        monkeypatch.setenv("VECTOR_STORE_BACKEND", "")
        monkeypatch.setenv("RETRIEVAL_MODE", "")
        monkeypatch.setenv("RERANK_STRATEGY", "")
        monkeypatch.setenv("UPLOAD_MAX_BYTES", "")
        monkeypatch.setenv("UPLOAD_TTL_SECONDS", "")
        monkeypatch.setenv("API_PORT", "")
        monkeypatch.setenv("LOG_LEVEL", "")

        s = Settings(llm_api_key="k")

        assert s.abstain_threshold == 0.0
        assert s.vector_store_backend == "memory"
        assert s.retrieval_mode == "hybrid"
        assert s.rerank_strategy == "none"
        assert s.upload_max_bytes == 5_000_000
        assert s.upload_ttl_seconds == 1800
        assert s.api_port == 8000
        assert s.log_level == "INFO"


class TestSettingsValidation:
    """Invalid values are rejected with clear messages."""

    def test_invalid_provider(self):
        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            Settings(llm_api_key="k", llm_provider="openai")

    def test_temperature_too_high(self):
        with pytest.raises(ValueError):
            Settings(llm_api_key="k", llm_temperature=3.0)

    def test_temperature_negative(self):
        with pytest.raises(ValueError):
            Settings(llm_api_key="k", llm_temperature=-1.0)

    def test_top_k_zero(self):
        with pytest.raises(ValueError):
            Settings(llm_api_key="k", retrieval_top_k=0)

    def test_port_out_of_range(self):
        with pytest.raises(ValueError):
            Settings(llm_api_key="k", api_port=99999)

    def test_max_question_length_too_small(self):
        with pytest.raises(ValueError):
            Settings(llm_api_key="k", max_question_length=5)

    def test_retry_min_wait_exceeds_max_wait_raises(self):
        with pytest.raises(ValueError, match="cannot exceed"):
            Settings(llm_api_key="k", llm_retry_min_wait=10.0, llm_retry_max_wait=5.0)

    def test_default_resilience_and_sessions(self):
        s = Settings(llm_api_key="k")
        assert s.llm_max_retries == 3
        assert s.llm_retry_min_wait == 0.5
        assert s.llm_retry_max_wait == 8.0
        assert s.session_ttl_seconds == 3600
        assert s.session_max_count == 1000

    def test_empty_api_key_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Settings(llm_api_key="")
            assert any("LLM_API_KEY is empty" in str(warning.message) for warning in w)


class TestSecretMasking:
    """Secrets never leak through repr, str, or model_dump."""

    TEST_SECRET = "super-secret-" + "groq-key-12345"  # pragma: allowlist secret

    def _make_settings(self):
        return Settings(llm_api_key=self.TEST_SECRET)

    def test_repr_masks_key(self):
        s = self._make_settings()
        r = repr(s)
        assert self.TEST_SECRET not in r
        assert "**********" in r

    def test_str_masks_key(self):
        s = self._make_settings()
        assert self.TEST_SECRET not in str(s)

    def test_model_dump_masks_key(self):
        s = self._make_settings()
        d = s.model_dump()
        assert d["llm_api_key"] == "**********"
        assert self.TEST_SECRET not in str(d)

    def test_secret_not_in_dict_values(self):
        s = self._make_settings()
        d = s.model_dump()
        flat_values = str(d.values())
        assert self.TEST_SECRET not in flat_values

    def test_get_secret_value_works(self):
        s = self._make_settings()
        key_val = s.llm_api_key.get_secret_value()
        assert key_val == self.TEST_SECRET


def test_cors_origins_from_environment_comma_separated_and_json(monkeypatch):
    """Regression: CORS_ORIGINS=a,b used to crash startup (env lists were decoded as JSON)."""
    monkeypatch.setenv("CORS_ORIGINS", "http://localhost:3010, http://127.0.0.1:3010")
    assert Settings().cors_origins == ["http://localhost:3010", "http://127.0.0.1:3010"]
    monkeypatch.setenv("CORS_ORIGINS", '["https://a.example","https://b.example"]')
    assert Settings().cors_origins == ["https://a.example", "https://b.example"]
    monkeypatch.setenv("CORS_ORIGINS", "http://only.example")
    assert Settings().cors_origins == ["http://only.example"]
