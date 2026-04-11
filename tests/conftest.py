"""
conftest.py — shared fixtures and dependency stubs for GraphRAG retrieval tests.

Heavy external dependencies (sentence-transformers, thefuzz, langchain, neo4j)
are stubbed in sys.modules BEFORE any retrieval.* import so tests run offline
and without the packages installed.

Project layout assumed:
    project/
        retrieval/
            __init__.py
            entity_extractor.py
            entity_resolver.py
            query_router.py
            cypher_templates.py
        tests/
            conftest.py
            test_entity_extractor.py
            test_entity_resolver.py
            test_query_router.py
            test_cypher_templates.py

Run from the project root:
    pytest tests/ -v
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Stub heavy dependencies before any retrieval.* module is imported.
# This block executes at collection time, which is before any test file's
# top-level imports run.
# ---------------------------------------------------------------------------

def _stub_if_missing(name: str, mod: types.ModuleType) -> None:
    if name not in sys.modules:
        sys.modules[name] = mod


# --- sentence_transformers ---
_st_mod = types.ModuleType("sentence_transformers")
_FakeST = MagicMock(name="SentenceTransformer")
_fake_st_instance = MagicMock()
_fake_st_instance.encode.return_value = np.array(
    [[0.1, 0.2, 0.3, 0.4]], dtype=np.float32
)
_FakeST.return_value = _fake_st_instance
_st_mod.SentenceTransformer = _FakeST
_stub_if_missing("sentence_transformers", _st_mod)

# --- thefuzz ---
_thefuzz_mod = types.ModuleType("thefuzz")
_thefuzz_fuzz_mod = types.ModuleType("thefuzz.fuzz")
_thefuzz_fuzz_mod.token_sort_ratio = MagicMock(return_value=80)
_thefuzz_mod.fuzz = _thefuzz_fuzz_mod
_stub_if_missing("thefuzz", _thefuzz_mod)
_stub_if_missing("thefuzz.fuzz", _thefuzz_fuzz_mod)

# --- langchain_core ---
_lc_mod = types.ModuleType("langchain_core")
_lc_msg_mod = types.ModuleType("langchain_core.messages")


class _FakeHumanMessage:
    def __init__(self, content: str):
        self.content = content


_lc_msg_mod.HumanMessage = _FakeHumanMessage
_lc_mod.messages = _lc_msg_mod
_stub_if_missing("langchain_core", _lc_mod)
_stub_if_missing("langchain_core.messages", _lc_msg_mod)

# --- langchain_google_genai ---
_lg_mod = types.ModuleType("langchain_google_genai")
_lg_mod.ChatGoogleGenerativeAI = MagicMock(name="ChatGoogleGenerativeAI")
_stub_if_missing("langchain_google_genai", _lg_mod)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_session(records: list[dict] | None = None, single_result: dict | None = None):
    """Return a mock Neo4j session that yields the given record dicts."""
    session = MagicMock()

    record_mocks = [MagicMock(data=MagicMock(return_value=r)) for r in (records or [])]
    run_result = MagicMock()
    run_result.__iter__ = MagicMock(return_value=iter(record_mocks))

    single_mock = MagicMock()
    if single_result:
        single_mock.__getitem__ = MagicMock(
            side_effect=lambda k: single_result[k]
        )
    else:
        single_mock = None

    run_result.single.return_value = single_mock
    session.run.return_value = run_result
    return session


def make_driver(records: list[dict] | None = None, single_result: dict | None = None):
    """Build a mock Neo4j driver whose session yields *records*."""
    session = _make_session(records, single_result)
    session.__enter__ = MagicMock(return_value=session)
    session.__exit__ = MagicMock(return_value=False)

    driver = MagicMock()
    driver.session.return_value = session
    return driver, session


@pytest.fixture()
def empty_driver():
    driver, _ = make_driver(records=[])
    return driver


@pytest.fixture()
def neo4j_driver():
    """Driver fixture factory — call with keyword args matching make_driver."""
    return make_driver


@pytest.fixture(autouse=True)
def clear_embedding_cache():
    """
    Clear the lru_cache on _cached_embedding AND reset _model.encode's call
    count between tests, so encode call assertions are accurate per test.

    Root cause of failures: lru_cache.cache_clear() only evicts cached results;
    it does not reset MagicMock.call_count, so counts bled across tests.
    """
    from retrieval.entity_resolver import _cached_embedding, _model
    _cached_embedding.cache_clear()
    _model.encode.reset_mock()
    yield
    _cached_embedding.cache_clear()
    _model.encode.reset_mock()