"""
Convert BIS/OFAC entity list CSV to Knowledge Graph triples for Neo4j.
Entities, locations, and regulatory programs become nodes and relationships.
"""

import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd


def normalize_name(name: Optional[str]) -> Optional[str]:
    """Strip whitespace and convert to Title Case."""
    if not name or not isinstance(name, str):
        return None
    return name.strip().title()


def extract_country_from_address(address: str) -> Optional[str]:
    """Extract country code from address string (last 2 chars after comma)."""
    if not address or not isinstance(address, str):
        return None
    parts = address.split(",")
    if parts:
        last_part = parts[-1].strip()
        if len(last_part) == 2:
            return last_part.upper()
    return None


def csv_to_triples(df: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Convert entity list dataframe to (Subject, Predicate, Object) triples.
    
    Returns:
        List of triple dictionaries with subject, predicate, object, and properties.
    """
    triples = []

    for idx, row in df.iterrows():
        entity_name = normalize_name(row.get("name"))
        entity_type = normalize_name(row.get("type", "Entity"))
        entity_id = row.get("_id")

        if not entity_name:
            continue

        # 1. Create primary entity node triple
        triples.append(
            {
                "subject": entity_name,
                "predicate": "IS_A",
                "object": entity_type,
                "properties": {"entity_id": entity_id, "source": row.get("source", "")},
            }
        )

        # 2. Extract locations from addresses
        addresses = row.get("addresses")
        if addresses and isinstance(addresses, str):
            address_list = addresses.split(";")
            countries_seen = set()

            for address in address_list:
                country = extract_country_from_address(address)
                if country and country not in countries_seen:
                    countries_seen.add(country)
                    triples.append(
                        {
                            "subject": entity_name,
                            "predicate": "LOCATED_IN",
                            "object": country,
                            "properties": {"address": address.strip()},
                        }
                    )

        # 3. Regulatory programs
        programs = row.get("programs")
        if programs and isinstance(programs, str):
            for program in programs.split(";"):
                program = program.strip()
                if program:
                    triples.append(
                        {
                            "subject": entity_name,
                            "predicate": "SUBJECT_TO",
                            "object": program,
                            "properties": {"program_code": program},
                        }
                    )

        # 4. License requirement status
        license_req = row.get("license_requirement")
        if license_req and isinstance(license_req, str) and license_req.strip():
            triples.append(
                {
                    "subject": entity_name,
                    "predicate": "HAS_LICENSE_REQUIREMENT",
                    "object": license_req.strip(),
                    "properties": {"requirement_type": license_req},
                }
            )

        # 5. Add alternative names as aliases
        alt_names = row.get("alt_names")
        if alt_names and isinstance(alt_names, str):
            for alt_name in alt_names.split(";"):
                alt_name = normalize_name(alt_name)
                if alt_name and alt_name != entity_name:
                    triples.append(
                        {
                            "subject": entity_name,
                            "predicate": "ALSO_KNOWN_AS",
                            "object": alt_name,
                            "properties": {},
                        }
                    )

        # 6. Add dates if present
        start_date = row.get("start_date")
        if start_date and isinstance(start_date, str) and start_date.strip():
            triples.append(
                {
                    "subject": entity_name,
                    "predicate": "ACTIVE_FROM",
                    "object": start_date.strip(),
                    "properties": {"date_type": "activation"},
                }
            )

        end_date = row.get("end_date")
        if end_date and isinstance(end_date, str) and end_date.strip():
            triples.append(
                {
                    "subject": entity_name,
                    "predicate": "ACTIVE_UNTIL",
                    "object": end_date.strip(),
                    "properties": {"date_type": "deactivation"},
                }
            )

        # 7. Remarks / risk notes as properties
        remarks = row.get("remarks")
        if remarks and isinstance(remarks, str) and remarks.strip():
            triples.append(
                {
                    "subject": entity_name,
                    "predicate": "HAS_REMARK",
                    "object": remarks.strip()[:100],  # Truncate long remarks
                    "properties": {"remark_type": "regulatory_note"},
                }
            )

    return triples


def main() -> None:
    # Load CSV
    csv_path = Path("./data/raw/bis_entity_list_2024.csv")
    if not csv_path.exists():
        print(f"CSV not found at {csv_path}")
        return

    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path, dtype=str)
    print(f"Loaded {len(df)} rows")

    # Convert to triples
    print("Converting to triples...")
    triples = csv_to_triples(df)
    print(f"Generated {len(triples)} triples")

    # Save triples
    output_path = Path("./data/processed/entity_list_triples.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(triples, f, indent=2, ensure_ascii=False)

    print(f"Triples saved to {output_path}")

    # Show sample
    print("\n--- Sample Triples ---")
    for triple in triples[:5]:
        print(f"{triple['subject']} --[{triple['predicate']}]--> {triple['object']}")


if __name__ == "__main__":
    main()
