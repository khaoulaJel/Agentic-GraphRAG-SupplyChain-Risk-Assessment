// Schema constraints for EV Battery Supply Chain GraphRAG
// Run with: python run_schema.py

// Uniqueness constraints
CREATE CONSTRAINT company_name  IF NOT EXISTS FOR (n:Company)   REQUIRE n.name IS UNIQUE;
CREATE CONSTRAINT material_name IF NOT EXISTS FOR (n:Material)  REQUIRE n.name IS UNIQUE;
CREATE CONSTRAINT country_name  IF NOT EXISTS FOR (n:Country)   REQUIRE n.name IS UNIQUE;
CREATE CONSTRAINT facility_name IF NOT EXISTS FOR (n:Facility)  REQUIRE n.name IS UNIQUE;

// Performance indexes
CREATE INDEX company_country IF NOT EXISTS FOR (n:Company) ON (n.country);
CREATE INDEX company_tier    IF NOT EXISTS FOR (n:Company) ON (n.tier);
CREATE INDEX material_cat    IF NOT EXISTS FOR (n:Material) ON (n.category);
CREATE INDEX country_risk    IF NOT EXISTS FOR (n:Country) ON (n.risk_tier);
