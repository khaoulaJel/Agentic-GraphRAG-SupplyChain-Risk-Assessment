// Schema constraints for EV Battery Supply Chain GraphRAG
// Run with: python run_schema.py

// Uniqueness constraints for primary entities
CREATE CONSTRAINT company_name  IF NOT EXISTS FOR (n:Company)   REQUIRE n.name IS UNIQUE;
CREATE CONSTRAINT material_name IF NOT EXISTS FOR (n:Material)  REQUIRE n.name IS UNIQUE;
CREATE CONSTRAINT country_name  IF NOT EXISTS FOR (n:Country)   REQUIRE n.name IS UNIQUE;
CREATE CONSTRAINT facility_name IF NOT EXISTS FOR (n:Facility)  REQUIRE n.name IS UNIQUE;
CREATE CONSTRAINT regbody_name  IF NOT EXISTS FOR (n:RegulatoryBody) REQUIRE n.name IS UNIQUE;
CREATE CONSTRAINT event_name    IF NOT EXISTS FOR (n:RiskEvent) REQUIRE n.name IS UNIQUE;

// Performance indexes for common queries
CREATE INDEX company_country IF NOT EXISTS FOR (n:Company) ON (n.country);
CREATE INDEX company_tier    IF NOT EXISTS FOR (n:Company) ON (n.tier);
CREATE INDEX company_type    IF NOT EXISTS FOR (n:Company) ON (n.type);
CREATE INDEX material_cat    IF NOT EXISTS FOR (n:Material) ON (n.category);
CREATE INDEX country_risk    IF NOT EXISTS FOR (n:Country) ON (n.risk_tier);
CREATE INDEX facility_type   IF NOT EXISTS FOR (n:Facility) ON (n.type);
CREATE INDEX event_type      IF NOT EXISTS FOR (n:RiskEvent) ON (n.type);
CREATE INDEX event_date      IF NOT EXISTS FOR (n:RiskEvent) ON (n.date);
