
"""Tests for retrieval.entity_resolver (hybrid BM25+vector)."""
from unittest.mock import MagicMock, patch
import numpy as np
from retrieval.entity_resolver import resolve_entities

def _make_hybrid_session(fulltext_rows, vector_rows):
    class _Row:
        def __init__(self, d):
            self._d = d
        def __getitem__(self, k):
            return self._d[k]
    # The resolver calls session.run twice per entity: once for fulltext, once for vector
    ft_results = []
    v_results = []
    # If multiple entities, split the rows accordingly
    if isinstance(fulltext_rows, list) and fulltext_rows and isinstance(fulltext_rows[0], list):
        ft_results = [MagicMock(__iter__=MagicMock(return_value=iter([_Row(r) for r in rows]))) for rows in fulltext_rows]
    else:
        ft_results = [MagicMock(__iter__=MagicMock(return_value=iter([_Row(r) for r in fulltext_rows])))]
    if isinstance(vector_rows, list) and vector_rows and isinstance(vector_rows[0], list):
        v_results = [MagicMock(__iter__=MagicMock(return_value=iter([_Row(r) for r in rows]))) for rows in vector_rows]
    else:
        v_results = [MagicMock(__iter__=MagicMock(return_value=iter([_Row(r) for r in vector_rows])))]
    # Interleave fulltext and vector results for each entity
    side_effects = []
    for ft, v in zip(ft_results, v_results):
        side_effects.extend([ft, v])
    # If only one entity, just use the two
    if not side_effects:
        side_effects = ft_results + v_results
    session = MagicMock()
    session.__enter__ = MagicMock(return_value=session)
    session.__exit__ = MagicMock(return_value=False)
    session.run.side_effect = side_effects
    driver = MagicMock()
    driver.session.return_value = session
    return driver, session

class TestHybridEntityResolver:
    pass