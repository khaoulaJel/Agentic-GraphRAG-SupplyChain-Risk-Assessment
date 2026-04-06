# EV Battery Supply Chain — Ontology

> **Status**: ✅ **FINALIZED** (6 node types, 8 relationship types)

## Node Types (6)

| Label | Properties | Type Enum | Description |
|-------|-----------|-----------|-------------|
| **Company** | name, type, country, tier | oem, cell_manufacturer, refiner, miner | tier: 1/2/3 | Supply chain companies at varying supply tiers |
| **Material** | name, category | mineral, chemical, component | - | Raw materials and components in the supply chain |
| **Country** | name, risk_tier | high, medium, low | - | Geographic entities with geopolitical risk assessment |
| **Facility** | name, type, location | mine, refinery, gigafactory | - | Physical production/extraction locations |
| **RegulatoryBody** | name, jurisdiction | - | - | Government or international regulatory entities |
| **RiskEvent** | name, type, date, severity | political, natural, financial, regulatory | - | Events that affect supply chain (with date & severity) |

## Relationship Types (8)

| Source | Relation | Target | Description |
|--------|----------|--------|-------------|
| Company | **SUPPLIES_TO** | Company | Direct supply relationship |
| Company | **SOURCES_FROM** | Material, Company | Material sourcing / upstream sourcing |
| Company, Facility | **PRODUCES** | Material | Manufacturing/extraction output |
| Company, Facility | **LOCATED_IN** | Country | Geographic location/jurisdiction |
| Company | **OWNS** | Company, Facility | Ownership or investment relations |
| Material, Company | **ALTERNATIVE_TO** | Material, Company | Substitution/alternative sourcing options |
| RegulatoryBody | **REGULATES** | Material, Company | Regulatory jurisdiction |
| RiskEvent | **AFFECTS** | Country, Company, Material | Impact of risk events on entities |

## Risk Categorization

### Company Tiers (Supply Chain Position)
- **Tier 1**: OEMs (Tesla, major EV manufacturers)
- **Tier 2**: Cell manufacturers (CATL, Panasonic, LGES)
- **Tier 3**: Refiners, miners (upstream raw materials)

### Geopolitical Risk Tiers
- **High**: DRC, Venezuela, Iran, North Korea (critical instability)
- **Medium**: Chile, Peru, Indonesia, China, Russia (moderate volatility)
- **Low**: Australia, Canada, USA, Norway, Sweden (stable partners)

### Company Types
- **OEM**: Original equipment manufacturer (e.g., Tesla)
- **Cell Manufacturer**: Battery cell production (e.g., CATL)
- **Refiner**: Material processing (e.g., cobalt refineries)
- **Miner**: Raw extraction (e.g., lithium mines)

### Material Categories
- **Mineral**: Lithium, Cobalt, Nickel
- **Chemical**: Electrolyte, separator materials
- **Component**: Cathode, anode, pack assembly

## Data Sources

- ✅ Tesla 10K filing (2023)
- ✅ CATL sustainability report
- ✅ Panasonic annual report
- ✅ LG Energy Solution annual report
- ✅ BIS entity list (mining operations)
- ✅ USGS mineral reports (Lithium, Cobalt, Nickel)
- ✅ ITA semiconductor analysis
- ✅ Apple supplier data

## Key Queries

1. **Single-source dependency**: Find materials with ≤2 suppliers
2. **Geopolitical exposure**: Trace sourcing to high-risk countries
3. **Disruption propagation**: How does a facility outage cascade through tiers?
4. **Regulatory compliance**: What materials/companies fall under specific jurisdictions?
5. **Risk event impact**: Which companies are affected by geopolitical events?
