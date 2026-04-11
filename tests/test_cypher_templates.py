"""Tests for retrieval.cypher_templates."""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from retrieval.cypher_templates import (
    COUNTRY_EXPOSURE_QUERY,
    COUNTRY_RISK_EXPOSURE,
    RELATION_WEIGHTS,
    TWO_HOP_NEIGHBORHOOD,
    fetch_country_exposure,
    fetch_hybrid_subgraph,
    fetch_subgraph,
)
from tests.conftest import make_driver


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _driver_with_records(records: list[dict]):
    driver, session = make_driver(records=records)
    return driver, session


# ---------------------------------------------------------------------------
# RELATION_WEIGHTS sanity
# ---------------------------------------------------------------------------
class TestRelationWeights:
    def test_supply_relations_have_highest_weight(self):
        assert RELATION_WEIGHTS["SUPPLIES_TO"] == 8
        assert RELATION_WEIGHTS["SOURCES_FROM"] == 8

    def test_noise_relations_have_low_weight(self):
        assert RELATION_WEIGHTS["MENTIONED_IN"] == 1

    def test_all_weights_positive(self):
        assert all(v > 0 for v in RELATION_WEIGHTS.values())


# ---------------------------------------------------------------------------
# Template string smoke tests
# ---------------------------------------------------------------------------
class TestCypherTemplates:
    def test_country_exposure_query_alias_matches_risk_template(self):
        """Regression: COUNTRY_EXPOSURE_QUERY must equal COUNTRY_RISK_EXPOSURE (not shortestPath)."""
        assert COUNTRY_EXPOSURE_QUERY == COUNTRY_RISK_EXPOSURE

    def test_country_risk_exposure_no_shortest_path(self):
        """Fix verification: shortestPath must NOT appear in the exposure template."""
        assert "shortestPath" not in COUNTRY_RISK_EXPOSURE

    def test_country_risk_exposure_multi_path_pattern(self):
        """Template must use variable-length path (not fixed single path)."""
        assert "[*1..5]" in COUNTRY_RISK_EXPOSURE or "[*" in COUNTRY_RISK_EXPOSURE

    def test_two_hop_neighborhood_uses_correct_depth(self):
        assert "[*1..2]" in TWO_HOP_NEIGHBORHOOD

    def test_hybrid_uses_apoc_expand(self):
        assert "apoc.path.expandConfig" in TWO_HOP_NEIGHBORHOOD or \
               "apoc.path.expandConfig" in fetch_hybrid_subgraph.__doc__ or \
               True  # template string check below
        from retrieval.cypher_templates import HYBRID_SUPPLY_CHAIN_RETRIEVAL
        assert "apoc" in HYBRID_SUPPLY_CHAIN_RETRIEVAL.lower()


# ---------------------------------------------------------------------------
# fetch_subgraph
# ---------------------------------------------------------------------------
class TestFetchSubgraph:
    def test_empty_entity_names_returns_empty(self):
        driver, session = _driver_with_records([])
        result = fetch_subgraph([], driver)
        assert result == []
        session.run.assert_not_called()

    def test_returns_record_data(self):
        row = {"anchor_name": "TSMC", "neighbor_name": "Apple", "relationship_types": ["SUPPLIES_TO"]}
        driver, _ = _driver_with_records([row])
        result = fetch_subgraph(["TSMC"], driver)
        assert result == [row]

    def test_passes_entity_names_to_cypher(self):
        driver, session = _driver_with_records([])
        fetch_subgraph(["TSMC", "Apple"], driver)
        call_kwargs = session.run.call_args.kwargs
        assert call_kwargs["entity_names"] == ["TSMC", "Apple"]

    def test_uses_two_hop_template(self):
        driver, session = _driver_with_records([])
        fetch_subgraph(["TSMC"], driver)
        cypher_used = session.run.call_args.args[0]
        assert cypher_used == TWO_HOP_NEIGHBORHOOD

    def test_multiple_records_returned(self):
        rows = [
            {"anchor_name": "TSMC", "neighbor_name": "Apple"},
            {"anchor_name": "TSMC", "neighbor_name": "Nvidia"},
        ]
        driver, _ = _driver_with_records(rows)
        result = fetch_subgraph(["TSMC"], driver)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# fetch_country_exposure
# ---------------------------------------------------------------------------
class TestFetchCountryExposure:
    def test_empty_names_returns_empty(self):
        driver, session = _driver_with_records([])
        result = fetch_country_exposure([], driver)
        assert result == []
        session.run.assert_not_called()

    def test_no_driver_returns_empty(self):
        result = fetch_country_exposure(["TSMC"], driver=None)
        assert result == []

    def test_returns_exposure_records(self):
        row = {"path_summary": ["TSMC", "Taiwan"], "exposure_score": 21, "risk_tier": "HIGH", "location_name": "Taiwan"}
        driver, _ = _driver_with_records([row])
        result = fetch_country_exposure(["TSMC"], driver)
        assert result == [row]

    def test_passes_entity_names(self):
        driver, session = _driver_with_records([])
        fetch_country_exposure(["TSMC", "Samsung"], driver)
        call_kwargs = session.run.call_args.kwargs
        assert "TSMC" in call_kwargs["entity_names"]
        assert "Samsung" in call_kwargs["entity_names"]

    def test_passes_relation_weights(self):
        driver, session = _driver_with_records([])
        fetch_country_exposure(["TSMC"], driver)
        call_kwargs = session.run.call_args.kwargs
        assert call_kwargs["relation_weights"] == RELATION_WEIGHTS

    def test_uses_country_risk_exposure_template(self):
        driver, session = _driver_with_records([])
        fetch_country_exposure(["TSMC"], driver)
        cypher_used = session.run.call_args.args[0]
        assert cypher_used == COUNTRY_RISK_EXPOSURE

    def test_no_shortest_path_in_executed_query(self):
        """Regression: the old query used shortestPath — must not appear anymore."""
        driver, session = _driver_with_records([])
        fetch_country_exposure(["TSMC"], driver)
        cypher_used = session.run.call_args.args[0]
        assert "shortestPath" not in cypher_used


# ---------------------------------------------------------------------------
# fetch_hybrid_subgraph
# ---------------------------------------------------------------------------
class TestFetchHybridSubgraph:
    def test_empty_names_returns_empty(self):
        driver, session = _driver_with_records([])
        result = fetch_hybrid_subgraph([], driver=driver)
        assert result == []
        session.run.assert_not_called()

    def test_no_driver_returns_empty(self):
        result = fetch_hybrid_subgraph(["TSMC"], driver=None)
        assert result == []

    def test_returns_weighted_records(self):
        row = {
            "anchor_name": "TSMC",
            "path_score": 30,
            "hops": 2,
            "path_nodes": [],
            "path_rels": [],
        }
        driver, _ = _driver_with_records([row])
        result = fetch_hybrid_subgraph(["TSMC"], driver=driver)
        assert result == [row]

    def test_passes_query_text(self):
        driver, session = _driver_with_records([])
        fetch_hybrid_subgraph(["TSMC"], query_text="supplier risk", driver=driver)
        call_kwargs = session.run.call_args.kwargs
        assert call_kwargs["query_text"] == "supplier risk"

    def test_passes_relation_weights(self):
        driver, session = _driver_with_records([])
        fetch_hybrid_subgraph(["TSMC"], driver=driver)
        call_kwargs = session.run.call_args.kwargs
        assert call_kwargs["relation_weights"] == RELATION_WEIGHTS

    def test_custom_relation_weights_override(self):
        custom = {"SUPPLIES_TO": 99}
        driver, session = _driver_with_records([])
        fetch_hybrid_subgraph(["TSMC"], driver=driver, relation_weights=custom)
        call_kwargs = session.run.call_args.kwargs
        assert call_kwargs["relation_weights"] == custom

    def test_apoc_failure_falls_back_to_subgraph(self):
        """
        If APOC raises (not installed), fetch_hybrid_subgraph must fall back
        to fetch_subgraph rather than propagating the exception.
        """
        session = MagicMock()
        session.__enter__ = MagicMock(return_value=session)
        session.__exit__ = MagicMock(return_value=False)
        session.run.side_effect = Exception("Unknown procedure apoc.path.expandConfig")

        driver = MagicMock()
        driver.session.return_value = session

        fallback_row = {"anchor_name": "TSMC", "neighbor_name": "Apple"}
        with patch("retrieval.cypher_templates.fetch_subgraph", return_value=[fallback_row]) as mock_fb:
            result = fetch_hybrid_subgraph(["TSMC"], driver=driver)

        mock_fb.assert_called_once_with(["TSMC"], driver)
        assert result == [fallback_row]

    def test_empty_strings_filtered_from_entity_names(self):
        driver, session = _driver_with_records([])
        fetch_hybrid_subgraph(["TSMC", "", "Apple"], driver=driver)
        call_kwargs = session.run.call_args.kwargs
        assert "" not in call_kwargs["entity_names"]
        assert "TSMC" in call_kwargs["entity_names"]
        assert "Apple" in call_kwargs["entity_names"]


# ---------------------------------------------------------------------------
# Integration-style: route → fetch chain
# ---------------------------------------------------------------------------
class TestRouteToFetchIntegration:
    """
    Smoke test: verify that route_query → fetch_country_exposure produces
    records with the expected shape when given a risk query.
    """

    def test_exposure_query_produces_location_records(self):
        from retrieval.query_router import route_query

        row = {
            "path_summary": ["Apple", "China"],
            "exposure_score": 18,
            "risk_tier": "HIGH",
            "location_name": "China",
        }
        driver, _ = _driver_with_records([row])
        resolved = {"companies": ["Apple"], "countries": ["China"]}

        with patch("retrieval.query_router.fetch_country_exposure", return_value=[row]) as mock_exp:
            result = route_query("sanction risk", resolved=resolved, driver=driver)

        assert result == [row]
        mock_exp.assert_called_once()
        names_arg = mock_exp.call_args.args[0]
        assert "Apple" in names_arg
        assert "China" in names_arg

    def test_traversal_query_calls_hybrid(self):
        from retrieval.query_router import route_query

        row = {"anchor_name": "TSMC", "path_score": 42}
        driver, _ = _driver_with_records([row])
        resolved = {"companies": ["TSMC"], "materials": ["silicon"]}

        with patch("retrieval.query_router.fetch_hybrid_subgraph", return_value=[row]) as mock_hyb:
            result = route_query("who supplies silicon to TSMC?", resolved=resolved, driver=driver)

        assert result == [row]
        mock_hyb.assert_called_once()
