"""Entity alias resolution for retrieval-time normalization."""

from __future__ import annotations

import logging
from pathlib import Path

ALIAS_MAP = {
    # Company aliases
    "contemporary amperex": "CATL",
    "contemporary amperex technology": "CATL",
    "lg chem energy": "LG Energy Solution",
    "lg energy": "LG Energy Solution",
    "panasonic energy": "Panasonic",
    # Country aliases
    "democratic republic of congo": "DRC",
    "republic of congo": "DRC",
    "dr congo": "DRC",
    "prc": "China",
    "people's republic of china": "China",
    # Material aliases
    "li": "lithium",
    "co": "cobalt",
    "ni": "nickel",
    "lithium carbonate": "lithium",
    "lithium hydroxide": "lithium",
    # Expanded for new entity types (examples, expand as needed)
    # Facility aliases
    "gigafactory nevada": "Gigafactory Nevada",
    # Location aliases
    "shanghai": "Shanghai",
    # Organization aliases
    "who": "World Health Organization",
    # Product aliases
    "model 3": "Tesla Model 3",
    # Regulation aliases
    "ira": "Inflation Reduction Act",
    # RegulatoryBody aliases
    "epa": "Environmental Protection Agency",
    # RiskEvent aliases
    "covid": "COVID-19 Pandemic",
}

_ALIAS_MISS_LOG = Path("logs/alias_misses.log")


def _get_alias_logger() -> logging.Logger:
    logger = logging.getLogger("retrieval.alias_miss")
    if logger.handlers:
        return logger

    _ALIAS_MISS_LOG.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(_ALIAS_MISS_LOG)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def resolve_entities(extracted: dict[str, list[str]]) -> dict[str, list[str]]:
    """Normalize extracted entities against known aliases without dropping unknowns."""
    logger = _get_alias_logger()
    resolved: dict[str, list[str]] = {}

    for entity_type, names in extracted.items():
        normalized_names = []
        for name in names:
            candidate = (name or "").strip()
            if not candidate:
                continue

            key = candidate.lower()
            canonical = ALIAS_MAP.get(key)
            if canonical is None:
                logger.info("alias_miss | type=%s | raw=%s", entity_type, candidate)
                canonical = candidate.title()

            normalized_names.append(canonical)

        resolved[entity_type] = normalized_names

    return resolved