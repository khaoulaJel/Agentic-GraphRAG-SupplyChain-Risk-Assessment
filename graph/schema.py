"""
Graph schema / ontology for EV Battery Supply Chain GraphRAG.

Defines 6 core node types, 8 relationship types, entity aliases for deduplication,
and geopolitical risk tiers.
"""

# ── Node types ────────────────────────────────────────────────────────────────
# Exactly 6 node types with properties

NODE_TYPES = {
    "Company": {
        "properties": ["name", "type", "country", "tier"],
        "description": "Supply chain companies (OEM, cell manufacturer, refiner, miner)",
        "constraints": {"type": ["oem", "cell_manufacturer", "refiner", "miner"], "tier": [1, 2, 3]}
    },
    "Material": {
        "properties": ["name", "category"],
        "description": "Materials and components (mineral, chemical, component)",
        "constraints": {"category": ["mineral", "chemical", "component"]}
    },
    "Country": {
        "properties": ["name", "risk_tier"],
        "description": "Geographic entities with geopolitical risk assessment",
        "constraints": {"risk_tier": ["high", "medium", "low"]}
    },
    "Facility": {
        "properties": ["name", "type", "location"],
        "description": "Production/extraction facilities (mine, refinery, gigafactory)",
        "constraints": {"type": ["mine", "refinery", "gigafactory"]}
    },
    "RegulatoryBody": {
        "properties": ["name", "jurisdiction"],
        "description": "Government or international regulatory entities"
    },
    "RiskEvent": {
        "properties": ["name", "type", "date", "severity"],
        "description": "Significant events affecting supply chain",
        "constraints": {"type": ["political", "natural", "financial", "regulatory"]}
    }
}

# ── Relationship types ────────────────────────────────────────────────────────
# Exactly 8 relationship types

RELATIONSHIP_TYPES = [
    "SUPPLIES_TO",      # Company → Company (direct supply)
    "SOURCES_FROM",     # Company → Material or Company (material sourcing)
    "PRODUCES",         # Company/Facility → Material (production)
    "LOCATED_IN",       # Company/Facility → Country (geographic location)
    "OWNS",             # Company → Company/Facility (ownership/investment)
    "ALTERNATIVE_TO",   # Material → Material or Company → Company (substitution)
    "REGULATES",        # RegulatoryBody → Material/Company (regulatory jurisdiction)
    "AFFECTS",          # RiskEvent → Country/Company/Material (impact)
]

# ── Entity aliases (for deduplication) ────────────────────────────────────────
# Map canonical names to common aliases

ALIASES = {
    "Tesla Inc.": ["Tesla", "TSLA"],
    "Contemporary Amperex Technology": ["CATL", "CAT", "Contemporary Amperex Technology Co."],
    "Panasonic Corporation": ["Panasonic", "Panasonic Energy"],
    "LG Energy Solution": ["LG", "LGES", "LG Chem"],
    "Lithium Carbonate": ["Lithium", "Li2CO3"],
    "Cobalt": ["Cobalt Oxide", "Cobalt Metal"],
    "Nickel": ["Nickel Oxide", "Nickel Metal"],
}

# ── Country / geopolitical risk tiers ─────────────────────────────────────────
# Categorized by supply chain vulnerability and political stability

COUNTRY_RISK = {
    "high": {
        "countries": ["Democratic Republic of Congo", "DRC", "Venezuela", "Iran", "North Korea"],
        "description": "Critical geopolitical/instability risk"
    },
    "medium": {
        "countries": ["Chile", "Peru", "Indonesia", "China", "Russia"],
        "description": "Moderate geopolitical/volatility risk"
    },
    "low": {
        "countries": ["Australia", "Canada", "USA", "Norway", "Sweden"],
        "description": "Stable supply chain partners"
    }
}
