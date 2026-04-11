"""Tests for retrieval.query_router."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from retrieval.query_router import (
    COUNTRY_INTENT_KEYWORDS,
    RISK_INTENT_KEYWORDS,
    _all_entity_names,
    _classify_intent_with_llm,
    _keyword_intent,
    route_query,
)
from tests.conftest import make_driver


# ---------------------------------------------------------------------------
# _keyword_intent
# ---------------------------------------------------------------------------
class TestKeywordIntent:
    @pytest.mark.parametrize("query", [
        "What is the risk exposure of TSMC?",
        "Which companies are exposed to China sanctions?",
        "Is Apple affected by tariffs?",
        "Show geopolitical disruption in chip supply",
        "What country does Samsung operate in?",
        "List locations of Foxconn facilities",
        "Geographic exposure of Tesla battery sourcing",
    ])
    def test_risk_and_country_keywords_yield_exposure(self, query):
        assert _keyword_intent(query) == "EXPOSURE_ANALYSIS"

    @pytest.mark.parametrize("query", [
        "Who supplies lithium to CATL?",
        "What materials does Samsung source?",
        "List tier-2 suppliers of Ford",
        "Where does Apple manufacture the iPhone?",
        "Which components does TSMC produce?",
    ])
    def test_supply_chain_keywords_yield_traversal(self, query):
        assert _keyword_intent(query) == "GRAPH_TRAVERSAL"

    def test_unrecognised_query_defaults_to_traversal(self):
        assert _keyword_intent("Tell me everything about TSMC") == "GRAPH_TRAVERSAL"

    def test_case_insensitive(self):
        assert _keyword_intent("RISK IN SUPPLY CHAIN") == "EXPOSURE_ANALYSIS"

    def test_no_duplicate_keywords(self):
        """Regression: RISK_INTENT_KEYWORDS must be defined exactly once."""
        assert len(RISK_INTENT_KEYWORDS) == len(set(RISK_INTENT_KEYWORDS))


# ---------------------------------------------------------------------------
# _all_entity_names
# ---------------------------------------------------------------------------
class TestAllEntityNames:
    def test_flattens_all_types(self):
        resolved = {
            "companies": ["TSMC", "Apple"],
            "countries": ["Taiwan"],
            "materials": ["silicon"],
        }
        names = _all_entity_names(resolved)
        assert set(names) == {"TSMC", "Apple", "Taiwan", "silicon"}

    def test_empty_lists_ignored(self):
        resolved = {"companies": [], "materials": []}
        assert _all_entity_names(resolved) == []

    def test_none_values_in_list_skipped(self):
        resolved = {"companies": ["TSMC", None, "", "Apple"]}
        names = _all_entity_names(resolved)
        assert None not in names
        assert "" not in names
        assert "TSMC" in names

    def test_non_list_values_skipped(self):
        """Non-list values (defensive) should not raise."""
        resolved = {"companies": ["TSMC"], "meta": "ignored"}
        names = _all_entity_names(resolved)
        assert names == ["TSMC"]

    def test_previous_bug_only_companies_would_miss_materials(self):
        """
        Regression for original bug where only companies were forwarded.
        All entity types must appear in the flat list.
        """
        resolved = {
            "companies": ["TSMC"],
            "materials": ["silicon"],
            "countries": ["Taiwan"],
            "facilities": ["Fab 18"],
        }
        names = _all_entity_names(resolved)
        assert "silicon" in names
        assert "Taiwan" in names
        assert "Fab 18" in names


# ---------------------------------------------------------------------------
# _classify_intent_with_llm
# ---------------------------------------------------------------------------
class TestClassifyIntentWithLlm:
    def _langchain_llm(self, response_text: str) -> MagicMock:
        resp = MagicMock()
        resp.content = response_text
        llm = MagicMock(spec=["invoke"])
        llm.invoke.return_value = resp
        return llm

    def _anthropic_llm(self, response_text: str) -> MagicMock:
        content_block = MagicMock()
        content_block.text = response_text
        resp = MagicMock()
        resp.content = [content_block]
        llm = MagicMock()
        del llm.invoke          # no invoke attr → Anthropic path
        llm.messages.create.return_value = resp
        return llm

    def test_langchain_exposure_classification(self):
        llm = self._langchain_llm("EXPOSURE_ANALYSIS")
        assert _classify_intent_with_llm("risk query", llm) == "EXPOSURE_ANALYSIS"

    def test_langchain_traversal_classification(self):
        llm = self._langchain_llm("GRAPH_TRAVERSAL")
        assert _classify_intent_with_llm("supplier query", llm) == "GRAPH_TRAVERSAL"

    def test_anthropic_exposure_classification(self):
        llm = self._anthropic_llm("EXPOSURE_ANALYSIS")
        assert _classify_intent_with_llm("sanction risk", llm) == "EXPOSURE_ANALYSIS"

    def test_anthropic_uses_correct_model_string(self):
        llm = self._anthropic_llm("GRAPH_TRAVERSAL")
        _classify_intent_with_llm("query", llm)
        call_kwargs = llm.messages.create.call_args.kwargs
        assert call_kwargs["model"] == "claude-sonnet-4-20250514"

    def test_unknown_llm_returns_none(self):
        llm = object()   # no invoke, no messages
        assert _classify_intent_with_llm("query", llm) is None

    def test_unexpected_llm_response_returns_none(self):
        llm = self._langchain_llm("I cannot decide")
        assert _classify_intent_with_llm("query", llm) is None

    def test_llm_exception_returns_none(self):
        llm = MagicMock(spec=["invoke"])
        llm.invoke.side_effect = RuntimeError("network error")
        assert _classify_intent_with_llm("query", llm) is None

    def test_lowercased_response_normalised(self):
        llm = self._langchain_llm("exposure_analysis")
        assert _classify_intent_with_llm("query", llm) == "EXPOSURE_ANALYSIS"


# ---------------------------------------------------------------------------
# route_query — intent-only mode (no driver)
# ---------------------------------------------------------------------------
class TestRouteQueryIntentOnly:
    def test_risk_query_returns_exposure(self):
        assert route_query("What is the geopolitical risk for Apple?") == "EXPOSURE_ANALYSIS"

    def test_supply_query_returns_traversal(self):
        assert route_query("List suppliers of TSMC") == "GRAPH_TRAVERSAL"

    def test_llm_overrides_keyword(self):
        """LLM returning GRAPH_TRAVERSAL on a keyword-risk query must win."""
        resp = MagicMock()
        resp.content = "GRAPH_TRAVERSAL"
        llm = MagicMock(spec=["invoke"])
        llm.invoke.return_value = resp
        # Query has risk keyword but LLM says GRAPH_TRAVERSAL
        result = route_query("risk assessment", llm=llm)
        assert result == "GRAPH_TRAVERSAL"

    def test_llm_failure_falls_back_to_keyword(self):
        llm = MagicMock(spec=["invoke"])
        llm.invoke.side_effect = RuntimeError("boom")
        result = route_query("country exposure of Tesla", llm=llm)
        assert result == "EXPOSURE_ANALYSIS"

    def test_no_resolved_returns_string(self):
        result = route_query("sanction risk", resolved=None, driver=None)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# route_query — live retrieval mode
# ---------------------------------------------------------------------------
class TestRouteQueryLiveRetrieval:
    _resolved = {
        "companies": ["TSMC"],
        "materials": ["silicon"],
        "countries": ["Taiwan"],
    }

    def test_exposure_intent_calls_fetch_country_exposure(self):
        driver, _ = make_driver(records=[{"path_summary": ["TSMC", "Taiwan"]}])
        with patch("retrieval.query_router.fetch_country_exposure") as mock_exposure, \
             patch("retrieval.query_router.fetch_hybrid_subgraph") as mock_hybrid:
            mock_exposure.return_value = [{"location_name": "Taiwan"}]
            route_query("sanction risk", resolved=self._resolved, driver=driver)
        mock_exposure.assert_called_once()
        mock_hybrid.assert_not_called()

    def test_traversal_intent_calls_fetch_hybrid_subgraph(self):
        driver, _ = make_driver(records=[])
        with patch("retrieval.query_router.fetch_country_exposure") as mock_exposure, \
             patch("retrieval.query_router.fetch_hybrid_subgraph") as mock_hybrid:
            mock_hybrid.return_value = []
            route_query("Who supplies lithium to CATL?", resolved=self._resolved, driver=driver)
        mock_hybrid.assert_called_once()
        mock_exposure.assert_not_called()

    def test_all_entity_types_forwarded_to_fetch(self):
        """Regression: previous code only passed companies to the fetch function."""
        driver, _ = make_driver(records=[])
        captured = {}

        def capture_hybrid(entity_names, **kwargs):
            captured["entity_names"] = entity_names
            return []

        with patch("retrieval.query_router.fetch_hybrid_subgraph", side_effect=capture_hybrid):
            route_query("material sourcing", resolved=self._resolved, driver=driver)

        assert "silicon" in captured["entity_names"]
        assert "Taiwan" in captured["entity_names"]
        assert "TSMC" in captured["entity_names"]

    def test_hybrid_receives_query_text(self):
        """fetch_hybrid_subgraph must receive the original query as query_text."""
        driver, _ = make_driver(records=[])
        captured = {}

        def capture(entity_names, query_text="", **kwargs):
            captured["query_text"] = query_text
            return []

        with patch("retrieval.query_router.fetch_hybrid_subgraph", side_effect=capture):
            route_query("source of lithium", resolved=self._resolved, driver=driver)

        assert captured["query_text"] == "source of lithium"

    def test_returns_fetch_result(self):
        expected = [{"anchor_name": "TSMC", "path_score": 42}]
        driver, _ = make_driver(records=[])
        with patch("retrieval.query_router.fetch_hybrid_subgraph", return_value=expected):
            result = route_query("lithium tier", resolved=self._resolved, driver=driver)
        assert result == expected
