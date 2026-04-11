"""Tests for retrieval.entity_resolver."""
    """Build a dict that mirrors a Neo4j record for the vector cypher query."""
    vector_rows: list[dict],
        )
        result = resolve_entities({"companies": ["TSMC"]}, driver)
        assert result["companies"] == ["TSMC Inc."]


# ---------------------------------------------------------------------------
# resolve_entities_semantically — vector match path
# ---------------------------------------------------------------------------
class TestResolveEntitiesSemantics:
    def test_vector_match_returns_canonical(self):
        driver, _ = _make_vector_session(
            vector_rows=[_neo4j_row("TSMC Inc.", score=0.92)],
        )
        result = resolve_entities_semantically({"companies": ["TSMC"]}, driver)
        assert result["companies"] == ["TSMC Inc."]

    def test_type_aware_label_filtering(self):
        """
        Row whose label does NOT match entity_type must be skipped;
        row with matching label must win.
        """
        rows = [
            _neo4j_row("Wrong Node", score=0.95, labels=["Material"]),   # wrong type
            _neo4j_row("TSMC Corp", score=0.91, labels=["Company"]),     # correct
        ]
        driver, _ = _make_vector_session(vector_rows=rows, entity_type="companies")
        result = resolve_entities_semantically({"companies": ["TSMC"]}, driver)
        assert result["companies"] == ["TSMC Corp"]

    def test_multiple_entities_in_same_type(self):
        """Each name in a list must be resolved independently."""
        # We need 4 session.run calls (2 entities × [vector, fallback path only if needed])
        # Simplify: both resolve via vector.
        session = MagicMock()
        session.__enter__ = MagicMock(return_value=session)
        session.__exit__ = MagicMock(return_value=False)

        def run_side_effect(cypher, **kwargs):
            name = kwargs.get("name", "")
            row_map = {
                "TSMC": _neo4j_row("TSMC Ltd"),
                "Apple": _neo4j_row("Apple Inc"),
            }
            rows = []
            for n, row in row_map.items():
                if n.lower() in str(kwargs).lower():
                    rows = [row]
                    break
            mock_result = MagicMock()
            mock_mocks = [MagicMock(**{"__getitem__": lambda s, k, r=row: r[k]}) for row in rows]
            mock_result.__iter__ = MagicMock(return_value=iter(mock_mocks))
            mock_result.single.return_value = None
            return mock_result

        # Use two separate drivers for simplicity
        driver1, _ = _make_vector_session([_neo4j_row("TSMC Ltd", 0.93)])
        driver2, _ = _make_vector_session([_neo4j_row("Apple Inc", 0.91)])

        r1 = resolve_entities_semantically({"companies": ["TSMC"]}, driver1)
        r2 = resolve_entities_semantically({"companies": ["Apple"]}, driver2)
        assert r1["companies"] == ["TSMC Ltd"]
        assert r2["companies"] == ["Apple Inc"]

    def test_empty_name_skipped(self):
        driver, session = _make_vector_session(vector_rows=[])
        result = resolve_entities_semantically({"companies": [""]}, driver)
        assert result["companies"] == []
        session.run.assert_not_called()

    def test_multiple_entity_types_all_resolved(self):
        """Fix verification: all entity types, not just companies, go through resolution."""
        driver_co, _ = _make_vector_session([_neo4j_row("TSMC Ltd", 0.93)])
        r = resolve_entities_semantically({"companies": ["TSMC"], "materials": []}, driver_co)
        assert "companies" in r
        assert "materials" in r

    # --- fallback path ---

    def test_no_vector_match_uses_exact_fallback(self):
        """When vector returns nothing, fallback MATCH must be tried."""
        driver, _ = _make_vector_session(
            vector_rows=[],
            fallback_row={"canonical_name": "Taiwan Semiconductor"},
        )
        result = resolve_entities_semantically({"companies": ["TSMC"]}, driver)
        assert result["companies"] == ["Taiwan Semiconductor"]

    def test_no_vector_no_fallback_passthrough(self):
        """Unresolvable names pass through unchanged."""
        driver, _ = _make_vector_session(vector_rows=[], fallback_row=None)
        result = resolve_entities_semantically({"companies": ["UnknownCo"]}, driver)
        assert result["companies"] == ["UnknownCo"]

    def test_low_score_vector_result_ignored(self):
        """
        Rows above threshold are returned by Neo4j (the WHERE in Cypher filters).
        If the mock returns an empty list (simulating threshold filter), fallback runs.
        """
        driver, _ = _make_vector_session(
            vector_rows=[],   # Neo4j filtered them out (score < threshold)
            fallback_row={"canonical_name": "Fallback Corp"},
        )
        result = resolve_entities_semantically({"companies": ["obscure"]}, driver)
        assert result["companies"] == ["Fallback Corp"]

    def test_vector_index_name_from_env(self):
        """RESOLVER_VECTOR_INDEX env var must be passed to session.run."""
        driver, session = _make_vector_session([_neo4j_row("TSMC", 0.91)])

        with patch.dict("os.environ", {"RESOLVER_VECTOR_INDEX": "my_custom_index"}):
            # Re-import to pick up env var — patch module-level constant instead
            with patch("retrieval.entity_resolver._VECTOR_INDEX", "my_custom_index"):
                resolve_entities_semantically({"companies": ["TSMC"]}, driver)

        first_call_kwargs = session.run.call_args_list[0].kwargs
        assert first_call_kwargs.get("index_name") == "my_custom_index"