"""
Local Named Entity Recognition using GLiNER on CPU.
Extracts supply chain entities from EV battery Markdown documents.
Uses sentence-windowing to respect token limits and capture all text.
"""

import csv
import os
import re
from pathlib import Path
from typing import Generator, Optional

# Disable HuggingFace symlinks (Windows permission fix)
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

import nltk
from gliner import GLiNER
from tqdm import tqdm

# Download punkt_tab tokenizer for NLTK (one-time operation)
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    print("Downloading NLTK punkt_tab tokenizer...")
    nltk.download("punkt_tab")

# Configuration
MODEL_NAME = "urchade/gliner_mediumv2.1"
ENTITY_LABELS = ["Company", "Material", "Location", "RiskFactor", "FinancialMetric"]
TOKEN_LIMIT = 512  # GLiNER typical token window
CSV_OUTPUT = Path("./data/processed/local_entities.csv")

# Entity normalization rules
NORMALIZATION_MAP = {
    r"\b(tesla|tsla)\b": "Tesla",
    r"\b(lg energy solution|lg energy)\b": "LG Energy Solution",
    r"\b(catl)\b": "CATL",
    r"\b(panasonic)\b": "Panasonic",
    r"\b(lges|lg electronics solutions?)\b": "LG Energy Solution",
    r"\b(saic|science applications?)\b": "Science Applications International",
    r"\b(drc)\b": "Democratic Republic of Congo",
    r"\b(us|u\.s\.)\b": "United States",
    r"\b(vr\s+china)\b": "Vietnam",
}


def load_gliner_model(model_name: str = MODEL_NAME, device: str = "cpu") -> GLiNER:
    """
    Load GLiNER model optimized for CPU inference.
    
    Args:
        model_name: HuggingFace model identifier
        device: 'cpu' for laptop, 'cuda' if GPU available
    
    Returns:
        Loaded GLiNER model instance
    """
    print(f"Loading {model_name} on {device}...")
    model = GLiNER.from_pretrained(model_name, local_files_only=False)
    model = model.to(device)
    print(f"✓ Model loaded on {device}")
    return model


def split_into_sentences(text: str) -> list[str]:
    """
    Split text into sentences while preserving context.
    Uses NLTK punkt_tab tokenizer for robust sentence splitting.
    """
    if not text:
        return []
    
    # Clean up markdown artifacts
    text = re.sub(r"<!--.*?-->", "", text)  # Remove HTML comments
    text = re.sub(r"\n{3,}", "\n\n", text)  # Normalize multiple newlines
    
    sentences = nltk.sent_tokenize(text)
    return [s.strip() for s in sentences if s.strip()]


def create_context_windows(
    sentences: list[str], token_limit: int = TOKEN_LIMIT, max_context: int = 3
) -> Generator[tuple[str, int, str], None, None]:
    """
    Create sliding windows of sentences to respect token limit.
    Each window includes up to max_context neighboring sentences for richer context.
    
    Yields:
        (window_text, start_idx, full_sentence_context)
    """
    for idx, sentence in enumerate(sentences):
        # Token estimation: rough rule of thumb is ~4 chars per token
        estimated_tokens = len(sentence) // 4
        
        if estimated_tokens > token_limit:
            # If single sentence exceeds limit, yield it anyway (GLiNER will truncate)
            yield sentence, idx, sentence
        else:
            # Build context window by adding neighbors
            window_sentences = [sentence]
            context_tokens = estimated_tokens
            
            # Add previous sentences
            for prev_idx in range(idx - 1, max(idx - max_context, -1), -1):
                prev_tokens = len(sentences[prev_idx]) // 4
                if context_tokens + prev_tokens < token_limit:
                    window_sentences.insert(0, sentences[prev_idx])
                    context_tokens += prev_tokens
                else:
                    break
            
            # Add following sentences
            for next_idx in range(idx + 1, min(idx + max_context + 1, len(sentences))):
                next_tokens = len(sentences[next_idx]) // 4
                if context_tokens + next_tokens < token_limit:
                    window_sentences.append(sentences[next_idx])
                    context_tokens += next_tokens
                else:
                    break
            
            window_text = " ".join(window_sentences)
            yield window_text, idx, sentence


def normalize_entity(text: str, label: str) -> str:
    """
    Normalize entity text using predefined rules and basic cleaning.
    Ensures consistency across similar entities.
    """
    text = text.strip()
    
    # Apply normalization rules (case-insensitive)
    for pattern, replacement in NORMALIZATION_MAP.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)
    
    # Title case for Company and Location
    if label in ["Company", "Location"]:
        text = text.title()
    
    return text


def extract_entities_from_file(
    model: GLiNER, md_path: Path, labels: list[str] = ENTITY_LABELS
) -> list[dict]:
    """
    Extract entities from a single Markdown file using GLiNER.
    
    Returns:
        List of dicts with keys: text, label, score, context, source_file
    """
    text = md_path.read_text(encoding="utf-8")
    sentences = split_into_sentences(text)
    
    entities = []
    seen = set()  # Track (text, label) pairs to avoid duplicates within file
    
    for window_text, sent_idx, full_sentence in create_context_windows(sentences):
        try:
            # GLiNER extraction
            results = model.predict_entities(window_text, labels, threshold=0.3)
            
            for entity in results:
                entity_text = normalize_entity(entity["text"], entity["label"])
                
                # Skip duplicates within the same file
                entity_key = (entity_text, entity["label"])
                if entity_key in seen:
                    continue
                seen.add(entity_key)
                
                entities.append(
                    {
                        "text": entity_text,
                        "label": entity["label"],
                        "score": round(entity["score"], 3),
                        "context": full_sentence[:200],  # Truncate context for CSV
                        "source_file": md_path.name,
                    }
                )
        except Exception as e:
            print(f"  Warning: Error processing sentence {sent_idx}: {e}")
    
    return entities


def extract_all_entities(
    md_dir: Path = Path("./data/output_markdown"),
    labels: list[str] = ENTITY_LABELS,
) -> list[dict]:
    """
    Extract entities from all Markdown files in a directory.
    Returns aggregated entity list.
    """
    if not md_dir.exists():
        raise FileNotFoundError(f"Directory not found: {md_dir}")
    
    md_files = sorted(md_dir.glob("*.md"))
    if not md_files:
        raise ValueError(f"No Markdown files found in {md_dir}")
    
    model = load_gliner_model()
    all_entities = []
    
    print(f"\nExtracting entities from {len(md_files)} files using GLiNER...")
    
    for md_path in tqdm(md_files, desc="Files"):
        try:
            file_entities = extract_entities_from_file(model, md_path, labels)
            all_entities.extend(file_entities)
        except Exception as e:
            print(f"✗ Error processing {md_path.name}: {e}")
    
    return all_entities


def save_entities_to_csv(
    entities: list[dict],
    output_path: Path = CSV_OUTPUT,
) -> None:
    """
    Save extracted entities to CSV for use as graph entities.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["text", "label", "score", "context", "source_file"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerows(entities)
    
    print(f"✓ Saved {len(entities)} unique entities to {output_path}")


def print_entity_statistics(entities: list[dict]) -> None:
    """Print summary statistics about extracted entities."""
    print("\n=== ENTITY EXTRACTION SUMMARY ===")
    print(f"Total entities: {len(entities)}")
    
    # Count by label
    by_label = {}
    for entity in entities:
        label = entity["label"]
        by_label[label] = by_label.get(label, 0) + 1
    
    print("\nEntities by type:")
    for label in sorted(by_label.keys()):
        count = by_label[label]
        print(f"  {label}: {count}")
    
    # Average confidence
    avg_score = sum(e["score"] for e in entities) / len(entities) if entities else 0
    print(f"\nAverage confidence score: {avg_score:.3f}")
    
    # Top entities by score
    top_entities = sorted(entities, key=lambda x: x["score"], reverse=True)[:5]
    print("\nTop-confidence entities:")
    for entity in top_entities:
        print(f"  {entity['text']} ({entity['label']}) - {entity['score']:.3f}")


def main() -> None:
    """Main pipeline for local entity extraction."""
    try:
        # Extract entities from all files
        entities = extract_all_entities()
        
        # Save to CSV
        save_entities_to_csv(entities)
        
        # Print stats
        print_entity_statistics(entities)
        
        print("\n✓ Local entity extraction complete!")
        print(f"  Output: {CSV_OUTPUT}")
        
    except Exception as e:
        print(f"✗ Pipeline error: {e}")
        raise


if __name__ == "__main__":
    main()
