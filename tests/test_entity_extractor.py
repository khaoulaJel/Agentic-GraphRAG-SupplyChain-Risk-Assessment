"""Tests for retrieval.entity_extractor."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Module under test (imported after conftest stubs heavy deps)
# ---------------------------------------------------------------------------
from retrieval.entity_extractor import (
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_ENTITY_PAYLOAD,
    _call_entity_llm,
    _extract_text_from_response,
    _strip_code_fences,
    extract_entities,
)


# ---------------------------------------------------------------------------
# _strip_code_fences
# ---------------------------------------------------------------------------
class TestStripCodeFences:
    def test_plain_json_unchanged(self):
        raw = '{"companies": []}'
        assert _strip_code_fences(raw) == raw

    def test_removes_backtick_json_fence(self):
        raw = "```json\n{}\n```"
        result = _strip_code_fences(raw)
        assert result == "{}"

    def test_removes_plain_triple_backtick(self):
        raw = "```\n{}\n```"
        result = _strip_code_fences(raw)
        assert result == "{}"

    def test_strips_leading_trailing_whitespace(self):
        raw = "   {}\n  "
        assert _strip_code_fences(raw) == "{}"


# ---------------------------------------------------------------------------
# _extract_text_from_response
# ---------------------------------------------------------------------------
class TestExtractTextFromResponse:
    def test_string_passthrough(self):
        assert _extract_text_from_response("hello") == "hello"

    def test_string_content_attr(self):
        resp = MagicMock()
        resp.content = "direct string content"
        assert _extract_text_from_response(resp) == "direct string content"

    def test_list_content_with_text_attr(self):
        item = MagicMock()
        item.text = "part one"
        resp = MagicMock()
        resp.content = [item]
        assert _extract_text_from_response(resp) == "part one"

    def test_list_content_with_text_dict(self):
        resp = MagicMock()
        resp.content = [{"text": "dict part"}]
        assert _extract_text_from_response(resp) == "dict part"

    def test_text_attr_fallback(self):
        resp = MagicMock(spec=[])           # no .content
        resp.text = "fallback text"
        del resp.content                     # ensure AttributeError path not taken
        # Use a fresh object with only .text
        class OnlyText:
            text = "fallback"
        assert _extract_text_from_response(OnlyText()) == "fallback"

    def test_unsupported_raises(self):
        with pytest.raises(TypeError, match="Unsupported"):
            _extract_text_from_response(12345)


# ---------------------------------------------------------------------------
# _call_entity_llm
# ---------------------------------------------------------------------------
class TestCallEntityLlm:
    def test_anthropic_style_uses_correct_model(self):
        """Verify the fix: Anthropic path uses DEFAULT_ANTHROPIC_MODEL, not Gemini."""
        content_block = MagicMock()
        content_block.text = '{"companies": ["Apple"]}'

        response = MagicMock()
        response.content = [content_block]

        client = MagicMock()
        client.messages.create.return_value = response

        result = _call_entity_llm("test prompt", client)

        call_kwargs = client.messages.create.call_args
        assert call_kwargs.kwargs["model"] == DEFAULT_ANTHROPIC_MODEL
        assert result == '{"companies": ["Apple"]}'

    def test_anthropic_style_max_tokens_400(self):
        content_block = MagicMock()
        content_block.text = "{}"
        response = MagicMock()
        response.content = [content_block]
        client = MagicMock()
        client.messages.create.return_value = response

        _call_entity_llm("prompt", client)
        assert client.messages.create.call_args.kwargs["max_tokens"] == 400

    def test_langchain_style_invoke(self):
        resp = MagicMock()
        resp.content = '{"companies": []}'
        client = MagicMock(spec=["invoke"])
        client.invoke.return_value = resp

        result = _call_entity_llm("prompt", client)
        assert result == '{"companies": []}'

    def test_unsupported_client_raises(self):
        with pytest.raises(TypeError, match="must support"):
            _call_entity_llm("prompt", object())


# ---------------------------------------------------------------------------
# extract_entities
# ---------------------------------------------------------------------------
VALID_JSON = json.dumps({
    "companies": ["TSMC"],
    "countries": ["Taiwan"],
    "entities": [],
    "facilities": [],
    "locations": [],
    "materials": ["silicon"],
    "organizations": [],
    "products": [],
    "regulations": [],
    "regulatory_bodies": [],
    "risk_events": [],
    "relationships": ["TSMC SUPPLIES_TO Apple"],
})


def _langchain_client(return_value: str) -> MagicMock:
    resp = MagicMock()
    resp.content = return_value
    client = MagicMock(spec=["invoke"])
    client.invoke.return_value = resp
    return client


class TestExtractEntities:
    def test_happy_path_returns_all_keys(self):
        client = _langchain_client(VALID_JSON)
        result = extract_entities("Who does TSMC supply?", client)
        assert set(result.keys()) == set(DEFAULT_ENTITY_PAYLOAD.keys())

    def test_companies_and_materials_parsed(self):
        client = _langchain_client(VALID_JSON)
        result = extract_entities("TSMC supplies silicon", client)
        assert "TSMC" in result["companies"]
        assert "silicon" in result["materials"]

    def test_relationships_included(self):
        client = _langchain_client(VALID_JSON)
        result = extract_entities("TSMC supplies Apple", client)
        assert "TSMC SUPPLIES_TO Apple" in result["relationships"]

    def test_missing_keys_default_to_empty_list(self):
        partial = json.dumps({"companies": ["Acme"]})
        client = _langchain_client(partial)
        result = extract_entities("Acme query", client)
        assert result["materials"] == []
        assert result["relationships"] == []

    def test_fenced_json_parsed(self):
        fenced = f"```json\n{VALID_JSON}\n```"
        client = _langchain_client(fenced)
        result = extract_entities("query", client)
        assert result["companies"] == ["TSMC"]

    def test_retry_on_json_error_then_success(self):
        """First call returns garbage, second returns valid JSON — retry must succeed."""
        resp_bad = MagicMock()
        resp_bad.content = "not json at all"
        resp_good = MagicMock()
        resp_good.content = VALID_JSON

        client = MagicMock(spec=["invoke"])
        client.invoke.side_effect = [resp_bad, resp_good]

        with patch("retrieval.entity_extractor.time.sleep"):
            result = extract_entities("query", client)

        assert result["companies"] == ["TSMC"]
        assert client.invoke.call_count == 2

    def test_all_retries_exhausted_returns_empty_payload(self):
        resp = MagicMock()
        resp.content = "{{bad json}}"
        client = MagicMock(spec=["invoke"])
        client.invoke.return_value = resp

        with patch("retrieval.entity_extractor.time.sleep"), \
             patch("retrieval.entity_extractor._MAX_RETRIES", 1):
            result = extract_entities("query", client)

        assert result == dict(DEFAULT_ENTITY_PAYLOAD)

    def test_build_default_llm_raises_without_api_key(self):
        from retrieval.entity_extractor import build_default_entity_llm
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="GEMINI_API_KEY"):
                build_default_entity_llm()

    def test_no_client_uses_default_llm_factory(self):
        """extract_entities calls build_default_entity_llm when client=None."""
        fake_client = _langchain_client(VALID_JSON)
        with patch(
            "retrieval.entity_extractor.build_default_entity_llm",
            return_value=fake_client,
        ) as mock_factory:
            extract_entities("no client query")
        mock_factory.assert_called_once()
