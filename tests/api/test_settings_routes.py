"""
Tests for settings API routes helper functions.

Tests cover:
  - _mask_value() masking logic
  - _is_placeholder() placeholder detection
  - _read_env_file() / _write_env_values() round-trip
  - ALLOWED_ENV_KEYS validation via the endpoint
"""

import pytest
from unittest.mock import patch

from bashgym.api.settings_routes import (
    _mask_value,
    _is_placeholder,
    _read_env_file,
    _write_env_values,
    ALLOWED_ENV_KEYS,
)


# ---------------------------------------------------------------------------
# 1. _mask_value()
# ---------------------------------------------------------------------------

class TestMaskValue:
    """Masking logic for API key display."""

    def test_empty_string_returns_empty(self):
        assert _mask_value("") == ""

    def test_short_string_returns_stars(self):
        """Strings <= 4 chars get fully masked."""
        assert _mask_value("abc") == "****"
        assert _mask_value("ab") == "****"
        assert _mask_value("a") == "****"

    def test_exactly_four_chars(self):
        assert _mask_value("abcd") == "****"

    def test_exactly_five_chars_shows_last_four(self):
        """5 chars -> '****bcde' (stars + last 4)."""
        assert _mask_value("abcde") == "****bcde"

    def test_normal_key(self):
        result = _mask_value("sk-ant-api03-abc123xyz")
        assert result == "****3xyz"

    def test_placeholder_value_returns_empty(self):
        """Placeholder values are treated as unset."""
        assert _mask_value("your-key-here") == ""
        assert _mask_value("your_key_here") == ""

    def test_none_returns_empty(self):
        """None is falsy, treated like empty."""
        assert _mask_value(None) == ""


# ---------------------------------------------------------------------------
# 2. _is_placeholder()
# ---------------------------------------------------------------------------

class TestIsPlaceholder:
    """Placeholder detection for API keys."""

    def test_your_key_here(self):
        assert _is_placeholder("your-key-here") is True

    def test_your_anthropic_api_key_here(self):
        assert _is_placeholder("your-anthropic-api-key-here") is True

    def test_underscore_variant(self):
        assert _is_placeholder("your_key_here") is True

    def test_empty_string(self):
        assert _is_placeholder("") is True

    def test_real_anthropic_key(self):
        assert _is_placeholder("sk-ant-abc123") is False

    def test_real_nvidia_key(self):
        assert _is_placeholder("nvapi-real-key") is False

    def test_whitespace_only(self):
        """Whitespace-only is truthy, and after strip doesn't match any pattern.
        The function does not explicitly handle whitespace-only as placeholder."""
        # _is_placeholder checks `if not value` (fails for "   " since it's truthy)
        # then strips and checks prefixes. "   " stripped = "" which doesn't match
        # any prefix/suffix pattern, so it returns False.
        assert _is_placeholder("   ") is False

    def test_starts_with_your_dash(self):
        assert _is_placeholder("your-something") is True

    def test_starts_with_your_underscore(self):
        assert _is_placeholder("your_something") is True

    def test_ends_with_here_dash(self):
        assert _is_placeholder("put-it-here") is True

    def test_ends_with_here_underscore(self):
        assert _is_placeholder("put_it_here") is True


# ---------------------------------------------------------------------------
# 3. _read_env_file() + _write_env_values()
# ---------------------------------------------------------------------------

class TestEnvFileRoundTrip:
    """Read/write .env file operations using tmp_path."""

    def _patch_env_path(self, tmp_path):
        """Return a context manager that patches _get_project_env_path."""
        env_file = tmp_path / ".env"
        return patch(
            "bashgym.api.settings_routes._get_project_env_path",
            return_value=env_file,
        )

    def test_round_trip(self, tmp_path):
        """Write values then read them back -- they should match."""
        with self._patch_env_path(tmp_path):
            _write_env_values({"ANTHROPIC_API_KEY": "sk-test-123", "HF_TOKEN": "hf_abc"})
            result = _read_env_file()
            assert result["ANTHROPIC_API_KEY"] == "sk-test-123"
            assert result["HF_TOKEN"] == "hf_abc"

    def test_preserves_comments(self, tmp_path):
        """Comments and blank lines should survive a write cycle."""
        env_file = tmp_path / ".env"
        env_file.write_text(
            "# This is a comment\n"
            "ANTHROPIC_API_KEY=old-value\n"
            "\n"
            "# Another comment\n"
            "HF_TOKEN=hf_old\n",
            encoding="utf-8",
        )
        with self._patch_env_path(tmp_path):
            _write_env_values({"ANTHROPIC_API_KEY": "sk-new-value"})
            raw = env_file.read_text(encoding="utf-8")

        assert "# This is a comment" in raw
        assert "# Another comment" in raw
        assert "ANTHROPIC_API_KEY=sk-new-value" in raw
        # HF_TOKEN was not updated, so it should keep its old value
        assert "HF_TOKEN=hf_old" in raw

    def test_handles_quoted_values(self, tmp_path):
        """Values wrapped in quotes should be stripped on read."""
        env_file = tmp_path / ".env"
        env_file.write_text(
            'ANTHROPIC_API_KEY="sk-quoted"\n'
            "HF_TOKEN='hf-single-quoted'\n",
            encoding="utf-8",
        )
        with self._patch_env_path(tmp_path):
            result = _read_env_file()
            assert result["ANTHROPIC_API_KEY"] == "sk-quoted"
            assert result["HF_TOKEN"] == "hf-single-quoted"

    def test_missing_file_returns_empty(self, tmp_path):
        """Reading a nonexistent .env returns an empty dict."""
        with self._patch_env_path(tmp_path):
            result = _read_env_file()
            assert result == {}

    def test_write_creates_file(self, tmp_path):
        """Writing to a nonexistent .env creates it."""
        with self._patch_env_path(tmp_path):
            _write_env_values({"OPENAI_API_KEY": "sk-new"})
            assert (tmp_path / ".env").exists()
            result = _read_env_file()
            assert result["OPENAI_API_KEY"] == "sk-new"

    def test_append_new_key(self, tmp_path):
        """New keys are appended at the end."""
        env_file = tmp_path / ".env"
        env_file.write_text("ANTHROPIC_API_KEY=existing\n", encoding="utf-8")
        with self._patch_env_path(tmp_path):
            _write_env_values({"OPENAI_API_KEY": "new-key"})
            result = _read_env_file()
            assert result["ANTHROPIC_API_KEY"] == "existing"
            assert result["OPENAI_API_KEY"] == "new-key"

    def test_overwrite_existing_key(self, tmp_path):
        """Existing keys are replaced in-place."""
        env_file = tmp_path / ".env"
        env_file.write_text("ANTHROPIC_API_KEY=old\n", encoding="utf-8")
        with self._patch_env_path(tmp_path):
            _write_env_values({"ANTHROPIC_API_KEY": "new"})
            result = _read_env_file()
            assert result["ANTHROPIC_API_KEY"] == "new"


# ---------------------------------------------------------------------------
# 4. ALLOWED_ENV_KEYS validation
# ---------------------------------------------------------------------------

class TestAllowedKeys:
    """Verify the allowlist contains expected keys and rejects others."""

    def test_expected_keys_present(self):
        for key in [
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "GOOGLE_API_KEY",
            "NVIDIA_API_KEY",
            "HF_TOKEN",
        ]:
            assert key in ALLOWED_ENV_KEYS

    def test_disallowed_key_not_present(self):
        assert "DATABASE_URL" not in ALLOWED_ENV_KEYS
        assert "SECRET_KEY" not in ALLOWED_ENV_KEYS
        assert "AWS_SECRET_ACCESS_KEY" not in ALLOWED_ENV_KEYS

    def test_update_endpoint_rejects_disallowed_key(self):
        """PUT /api/settings/env should reject keys not in the allowlist."""
        from fastapi.testclient import TestClient
        from bashgym.api.routes import app

        client = TestClient(app)
        response = client.put(
            "/api/settings/env",
            json={"values": {"FORBIDDEN_KEY": "some-value"}},
        )
        assert response.status_code == 400
        assert "not in the allowlist" in response.json()["detail"]
