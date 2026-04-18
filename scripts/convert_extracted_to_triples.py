"""
Convert LLM-extracted entities/relationships to triples format and reload into Neo4j.

This script:
1. Reads all extracted JSON files from /data/processed
2. Converts relationships to triples (JSONL format)
3. Validates triples
4. Reloads into Neo4j with batch UNWIND
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict
from tqdm import tqdm

# Configuration
PROCESSED_DATA_DIR = Path("./data/processed")
EXTRACTION_FILE_PATTERN = "entities_*.json"
OUTPUT_TRIPLES_FILE = Path("./data/processed/triples_silver_layer_extracted.jsonl")

# Valid predicates (from original VALID_PREDICATES)
VALID_PREDICATES = {
    "SUPPLIES_TO", "SOURCES_FROM", "PRODUCES", "LOCATED_IN", "OWNS",
    "ALTERNATIVE_TO", "REGULATES", "AFFECTS", "POTENTIAL_RISK",
    "SUBJECT_TO", "ALSO_KNOWN_AS", "MEMBER_OF", "HAS_RISK",
    "FOUNDED_IN", "COLLABORATES_WITH", "SUPPORTS", "COMMUNICATES_WITH",
    "USES", "IMPLEMENTS", "EVALUATED", "PURSUING", "BUILDS_PARTNERSHIPS",
    "ENGAGES_IN_DIALOGUE", "RECEIVES_AWARD", "RECOGNIZED_BY",
    # Additional predicates from LLM extraction
    "LOCATED_IN", "OPERATES_IN", "MADE_FROM", "CONTAINS", "REQUIRES",
    "AFFECTED_BY", "POSES_RISK", "CREATES_IMPACT", "HAS_CONSTRAINT",
    "COLLABORATES_WITH", "ALTERNATIVE_TO", "MEMBER_OF", "PARTNER_WITH",
    "ACTIVE_FROM", "ACTIVE_UNTIL", "MENTIONED_IN", "REFERENCED_IN",
    "IS_A", "HAS_LICENSE_REQUIREMENT", "REGULATED_BY", "HAS_SUPPLIER",
}


def load_extraction_file(file_path: Path) -> Dict[str, Any]:
    """Load LLM-extracted JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠ Error loading {file_path}: {e}")
        return {"entities": [], "relationships": []}


def extract_and_convert_triples(extraction_files: List[Path]) -> List[Dict[str, Any]]:
    """
    Convert LLM-extracted relationships to triples format.
    Also converts entities to "IS_A" triples for entities without explicit relationships.
    
    Returns list of triples with:
    - subject, predicate, object (required)
    - evidence, properties (optional but recommended)
    """
    triples = []
    skipped_count = 0
    entity_triples = []
    
    for file_path in tqdm(extraction_files, desc="Processing extraction files"):
        data = load_extraction_file(file_path)
        relationships = data.get("relationships", [])
        entities = data.get("entities", [])
        
        # Track which entities are already in relationships (as subject or object)
        related_entities = set()
        
        # Process relationships
        for rel in relationships:
            subject = rel.get("subject", "").strip()
            predicate = rel.get("predicate", "").strip()
            obj = rel.get("object", "").strip()
            
            # Validate relationship
            if not subject or not predicate or not obj:
                skipped_count += 1
                continue
            
            # Track entities in relationships
            related_entities.add(subject)
            related_entities.add(obj)
            
            # Create triple with full metadata
            triple = {
                "subject": subject,
                "predicate": predicate,
                "object": obj,
                "evidence": rel.get("evidence", ""),
                "source": file_path.stem.replace("entities_", "").replace(".json", ""),
                "properties": {
                    "confidence": rel.get("confidence", "MEDIUM"),
                    "subject_type": rel.get("subject_type", "Entity"),
                    "object_type": rel.get("object_type", "Entity"),
                },
            }
            
            triples.append(triple)
        
        # Process orphaned entities (not in any relationship) as fallback
        # Create "IS_A" triples for type classification
        source_name = file_path.stem.replace("entities_", "").replace(".json", "")
        
        for entity in entities:
            entity_name = entity.get("name", "").strip()
            entity_type = entity.get("type", "Entity").strip()
            
            # Skip if already used in relationship or invalid
            if not entity_name or entity_name in related_entities:
                continue
            
            # Create IS_A triple for entity type
            entity_triple = {
                "subject": entity_name,
                "predicate": "IS_A",
                "object": entity_type,
                "evidence": entity.get("context", ""),
                "source": source_name,
                "properties": {
                    "confidence": entity.get("confidence", "MEDIUM"),
                    "subject_type": entity_type,
                    "object_type": "Classification",
                },
            }
            
            entity_triples.append(entity_triple)
    
    all_triples = triples + entity_triples
    
    print(f"\n✓ Extracted {len(triples)} relationship triples from {len(extraction_files)} files")
    print(f"✓ Extracted {len(entity_triples)} entity classification triples")
    print(f"✓ Total: {len(all_triples)} triples")
    print(f"⚠ Skipped {skipped_count} invalid/incomplete relationships")
    
    return all_triples


def validate_triples(triples: List[Dict[str, Any]]) -> tuple:
    """
    Validate triples and return statistics.
    
    Returns: (valid_triples, invalid_triples)
    """
    valid = []
    invalid = []
    
    for triple in triples:
        # Check required fields
        if not all(k in triple for k in ["subject", "predicate", "object"]):
            invalid.append(triple)
            continue
        
        # Check no empty strings
        if not triple["subject"] or not triple["predicate"] or not triple["object"]:
            invalid.append(triple)
            continue
        
        # Check predicate is reasonable
        if len(triple["predicate"]) > 50:
            invalid.append(triple)
            continue
        
        valid.append(triple)
    
    return valid, invalid


def save_triples_jsonl(triples: List[Dict[str, Any]], output_file: Path) -> int:
    """
    Save triples to JSONL format (one JSON per line).
    
    Returns: number of triples saved
    """
    os.makedirs(output_file.parent, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for triple in triples:
            f.write(json.dumps(triple) + '\n')
    
    print(f"✓ Saved {len(triples)} triples to {output_file}")
    return len(triples)


def print_statistics(triples: List[Dict[str, Any]]) -> None:
    """Print statistics about extracted triples."""
    predicates = defaultdict(int)
    sources = defaultdict(int)
    confidence_levels = defaultdict(int)
    
    for triple in triples:
        predicates[triple.get("predicate", "UNKNOWN")] += 1
        sources[triple.get("source", "UNKNOWN")] += 1
        confidence_levels[triple.get("properties", {}).get("confidence", "UNKNOWN")] += 1
    
    print("\n=== EXTRACTION STATISTICS ===\n")
    
    print("Predicates (relationship types):")
    for pred, count in sorted(predicates.items(), key=lambda x: x[1], reverse=True):
        print(f"  {pred}: {count}")
    
    print(f"\nSource files:")
    for source, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
        print(f"  {source}: {count}")
    
    print(f"\nConfidence levels:")
    for conf, count in sorted(confidence_levels.items(), key=lambda x: x[1], reverse=True):
        print(f"  {conf}: {count}")
    
    print(f"\n=== TOTAL ===")
    print(f"  Total triples: {len(triples)}")


def merge_with_existing(new_triples_file: Path, existing_triples_file: Path) -> int:
    """
    Merge new extracted triples with existing regulatory triples.
    
    Returns: total count
    """
    print(f"\n📂 Merging with existing triples...")
    
    # Read new triples
    new_triples = []
    if new_triples_file.exists():
        with open(new_triples_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    new_triples.append(json.loads(line.strip()))
                except:
                    pass
    
    # Read existing triples
    existing_triples = []
    if existing_triples_file.exists():
        with open(existing_triples_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    existing_triples.append(json.loads(line.strip()))
                except:
                    pass
    
    print(f"  New extracted triples: {len(new_triples)}")
    print(f"  Existing regulatory triples: {len(existing_triples)}")
    
    # Deduplicate by (subject, predicate, object)
    seen = set()
    merged = []
    
    for triple in existing_triples + new_triples:
        key = (triple["subject"], triple["predicate"], triple["object"])
        if key not in seen:
            seen.add(key)
            merged.append(triple)
    
    # Save merged
    with open(existing_triples_file, 'w', encoding='utf-8') as f:
        for triple in merged:
            f.write(json.dumps(triple) + '\n')
    
    print(f"✓ Merged into {existing_triples_file} with {len(merged)} unique triples")
    return len(merged)


def main():
    """Main pipeline: extract → convert → validate → save → merge."""
    
    print("=" * 70)
    print("EXTRACTION CONVERSION PIPELINE")
    print("=" * 70)
    
    # Step 1: Find extraction files
    print(f"\n📂 Scanning {PROCESSED_DATA_DIR} for extraction files...")
    extraction_files = sorted(PROCESSED_DATA_DIR.glob(EXTRACTION_FILE_PATTERN))
    
    if not extraction_files:
        print(f"✗ No extraction files found matching pattern: {EXTRACTION_FILE_PATTERN}")
        return
    
    print(f"✓ Found {len(extraction_files)} extraction files:")
    for f in extraction_files:
        print(f"  - {f.name}")
    
    # Step 2: Extract and convert relationships to triples
    print(f"\n🔄 Converting relationships to triples...")
    triples = extract_and_convert_triples(extraction_files)
    
    # Step 3: Validate triples
    print(f"\n✓ Validating {len(triples)} triples...")
    valid_triples, invalid_triples = validate_triples(triples)
    
    print(f"  ✓ Valid: {len(valid_triples)}")
    print(f"  ⚠ Invalid: {len(invalid_triples)}")
    
    if invalid_triples:
        print(f"  Sample invalid triple: {invalid_triples[0]}")
    
    # Step 4: Print statistics
    print_statistics(valid_triples)
    
    # Step 5: Save to JSONL
    print(f"\n💾 Saving to JSONL format...")
    output_count = save_triples_jsonl(valid_triples, OUTPUT_TRIPLES_FILE)
    
    # Step 6: Merge with existing regulatory triples
    print(f"\n🔗 Merging with regulatory triples...")
    entity_list_file = PROCESSED_DATA_DIR / "entity_list_triples.json"
    existing_file = PROCESSED_DATA_DIR / "triples_silver_layer.jsonl"
    
    # Skip regulatory data - only keep extracted triples
    # User requested: "Discard: entity_list_triples.json"
    # Graph should contain ONLY LLM-extracted data
    
    if entity_list_file.exists():
        print(f"⊘ Regulatory data (entity_list_triples.json) SKIPPED (extraction-only mode)")
        print(f"  → Keeping only {output_count} LLM-extracted triples")
    
    print(f"\n" + "=" * 70)
    print(f"✓ CONVERSION COMPLETE!")
    print(f"=" * 70)
    print(f"\nNext steps:")
    print(f"1. Review statistics above")
    print(f"2. Run: python load_graph_neo4j.py")
    print(f"3. Graph will be reloaded with extracted triples")
    print(f"\nOutput file: {OUTPUT_TRIPLES_FILE}")


if __name__ == "__main__":
    main()
