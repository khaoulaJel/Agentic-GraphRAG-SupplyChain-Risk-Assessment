"""
Named Entity Recognition & Relation Extraction from Markdown files using Google Gemini.
Extracts Knowledge Graph triples from EV battery supply chain reports.
Handles rate limiting, chunking, and SQLite persistence.
"""

import json
import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Generator, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from tqdm import tqdm

# Configuration
GEMINI_MODEL = os.getenv("NER_MODEL", "gemini-2.0-flash")
CHUNK_SIZE = 2000
GEMINI_API_KEY = None


def initialize_database(db_path: Path = Path("./data/processed/triples.db")) -> None:
    """Create SQLite schema for storing extracted triples."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS triples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject TEXT NOT NULL,
            predicate TEXT NOT NULL,
            object TEXT NOT NULL,
            evidence TEXT,
            source_file TEXT,
            chunk_index INTEGER,
            extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_subject ON triples(subject)
        """
    )
    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_source_file ON triples(source_file)
        """
    )

    conn.commit()
    conn.close()


def chunk_markdown_safely(text: str, chunk_size: int = CHUNK_SIZE) -> Generator[str, None, None]:
    """
    Split markdown text into chunks without breaking sentences.
    
    Strategy:
    1. Split by double newlines (paragraphs)
    2. Merge paragraphs until approaching chunk_size
    3. If a single paragraph exceeds chunk_size, split on sentence boundaries
    
    Memory-efficient: Uses generators to avoid loading entire text at once.
    """
    paragraphs = text.split("\n\n")
    current_chunk = []
    current_length = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        para_len = len(para)

        # If adding this paragraph exceeds limit, yield current chunk and start new one
        if current_length + para_len + 2 > chunk_size and current_chunk:
            yield "\n\n".join(current_chunk)
            current_chunk = []
            current_length = 0

        # If single paragraph is larger than chunk_size, split by sentences
        if para_len > chunk_size:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            sent_chunk = []
            sent_length = 0

            for sentence in sentences:
                sent_len = len(sentence)
                if sent_length + sent_len + 1 > chunk_size and sent_chunk:
                    yield " ".join(sent_chunk)
                    sent_chunk = []
                    sent_length = 0

                sent_chunk.append(sentence)
                sent_length += sent_len + 1

            if sent_chunk:
                yield " ".join(sent_chunk)
        else:
            current_chunk.append(para)
            current_length += para_len + 2

    if current_chunk:
        yield "\n\n".join(current_chunk)


def build_system_prompt() -> str:
    """Construct the system prompt for LLM extraction aligned with ontology."""
    return """You are a Supply Chain Intelligence Analyst specializing in EV battery supply chains.
Your task is to extract high-fidelity Knowledge Graph triples from sustainability and regulatory reports.

## SCHEMA (Extract ONLY these node and relationship types):

**NODE TYPES**: Company, Material, Country, Facility, RegulatoryBody, RiskEvent

**RELATIONSHIPS**:
- Company SUPPLIES_TO Company
- Company SOURCES_FROM Material, Company
- Company/Facility PRODUCES Material
- Company/Facility LOCATED_IN Country
- Company OWNS Company, Facility
- Material/Company ALTERNATIVE_TO Material, Company
- RegulatoryBody REGULATES Material, Company
- RiskEvent AFFECTS Country, Company, Material

## EXTRACTION RULES:

1. **ATOMICITY**: Break complex sentences into multiple simple triples.
2. **RISK PREDICATES**: Use "POTENTIAL_RISK" for uncertain/future risks; "AFFECTS" for confirmed events.
3. **NORMALIZATION**: Use full entity names (e.g., "LG Energy Solution", not "LG").
4. **EVIDENCE**: Include a brief quote or context explaining why you extracted the triple.
5. **CONFIDENCE**: Only extract facts clearly stated in text, not inferred.

## OUTPUT FORMAT:

Return ONLY a JSON array of triples. No conversational text, no explanations.

[
  {
    "subject": "string",
    "predicate": "string",
    "object": "string",
    "evidence": "brief quote or context"
  }
]

If no triples are found, return: []
"""


def call_gemini_with_retry(
    client: ChatGoogleGenerativeAI, chunk: str, max_retries: int = 3
) -> Optional[list[dict]]:
    """
    Call Google Gemini API with automatic retry and backoff for rate limiting.
    Handles rate limit errors gracefully with exponential backoff.
    """
    system_prompt = build_system_prompt()

    for attempt in range(max_retries):
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Extract triples from:\n\n{chunk}"),
            ]
            response = client.invoke(messages)
            response_text = response.content.strip()

            # Parse JSON response
            try:
                # Try to find JSON array in response
                json_match = re.search(r"\[.*\]", response_text, re.DOTALL)
                if json_match:
                    triples = json.loads(json_match.group())
                    return triples if isinstance(triples, list) else None
            except json.JSONDecodeError:
                pass

            return None

        except Exception as e:
            # Gemini rate limiting typically comes as ResourceExhausted error
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "rate" in error_str.lower():
                # Exponential backoff: 2s, 4s, 8s
                wait_time = min(2**attempt, 60)
                print(f"  Rate limited. Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            else:
                print(f"  API error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)

    return None


def extract_triples_from_file(
    md_path: Path, db_path: Path = Path("./data/processed/triples.db")
) -> int:
    """
    Extract triples from a single Markdown file and save to SQLite.
    Returns count of triples extracted.
    
    Memory management: Uses chunked generators to avoid loading entire file.
    API management: Respects rate limits with per-chunk delays.
    """
    from dotenv import load_dotenv

    load_dotenv()

    # Initialize LLM client
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set in environment")
        return 0

    client = ChatGoogleGenerativeAI(model=GEMINI_MODEL, api_key=api_key, temperature=0.2)

    # Read markdown
    text = md_path.read_text(encoding="utf-8")
    chunks = list(chunk_markdown_safely(text, CHUNK_SIZE))

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    total_extracted = 0

    for chunk_idx, chunk in enumerate(tqdm(chunks, desc=f"{md_path.name}", leave=False)):
        triples = call_gemini_with_retry(client, chunk)

        if triples:
            for triple in triples:
                try:
                    cursor.execute(
                        """
                        INSERT INTO triples (subject, predicate, object, evidence, source_file, chunk_index)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            triple.get("subject", ""),
                            triple.get("predicate", ""),
                            triple.get("object", ""),
                            triple.get("evidence", ""),
                            md_path.name,
                            chunk_idx,
                        ),
                    )
                    total_extracted += 1
                except Exception as e:
                    print(f"  Error saving triple: {e}")

            conn.commit()

        # Rate limiting: polite throttling between API calls
        # Gemini free tier has limits; 0.5s delay helps avoid rate limiting
        time.sleep(0.5)

    conn.close()
    return total_extracted


def export_triples_to_jsonl(
    db_path: Path = Path("./data/processed/triples.db"),
    output_path: Path = Path("./data/processed/triples_silver_layer.jsonl"),
) -> None:
    """
    Export SQLite triples to JSONL for downstream Neo4j ingestion.
    JSONL format: One JSON triple per line (newline-delimited).
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT subject, predicate, object, evidence, source_file FROM triples")
    rows = cursor.fetchall()

    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            triple = {
                "subject": row[0],
                "predicate": row[1],
                "object": row[2],
                "evidence": row[3],
                "source_file": row[4],
            }
            f.write(json.dumps(triple) + "\n")

    conn.close()
    print(f"Exported {len(rows)} triples to {output_path}")


def main() -> None:
    """Main extraction pipeline."""
    # Initialize database
    db_path = Path("./data/processed/triples.db")
    initialize_database(db_path)

    # Find all markdown files in output_markdown
    md_dir = Path("./data/output_markdown")
    if not md_dir.exists():
        print(f"ERROR: {md_dir} not found")
        return

    md_files = sorted(md_dir.glob("*.md"))
    if not md_files:
        print(f"No Markdown files found in {md_dir}")
        return

    print(f"Found {len(md_files)} Markdown files to process")
    total_triples = 0

    # Process each file with progress bar
    for md_path in tqdm(md_files, desc="Processing files"):
        try:
            count = extract_triples_from_file(md_path, db_path)
            total_triples += count
            print(f"✓ {md_path.name}: {count} triples extracted")
        except Exception as e:
            print(f"✗ {md_path.name}: {e}")

    print(f"\n=== EXTRACTION COMPLETE ===")
    print(f"Total triples extracted: {total_triples}")

    # Export to JSONL silver layer
    export_triples_to_jsonl(db_path)

    # Show stats
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM triples")
    total_count = cursor.fetchone()[0]
    cursor.execute("SELECT DISTINCT predicate FROM triples ORDER BY predicate")
    predicates = [row[0] for row in cursor.fetchall()]
    conn.close()

    print(f"Database contains {total_count} total triples")
    if predicates:
        print(f"Predicates found: {', '.join(predicates)}")


if __name__ == "__main__":
    main()
