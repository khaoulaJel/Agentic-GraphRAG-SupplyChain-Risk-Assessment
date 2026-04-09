from pathlib import Path
from typing import Iterator, Tuple

from pypdf import PdfReader, PdfWriter

from convert_pdfs_to_markdown import (
    build_docling_converter,
    extract_pdf_metadata,
    iter_pdf_files,
)


def split_pdf_into_chunks(pdf_path: Path, chunk_size: int = 10) -> Iterator[Tuple[Path, int, int]]:
    """
    Yield temp PDFs containing consecutive page ranges from the original PDF.
    Each chunk contains at most `chunk_size` pages.
    """
    reader = PdfReader(str(pdf_path))
    total_pages = len(reader.pages)

    for start_page in range(0, total_pages, chunk_size):
        end_page = min(start_page + chunk_size, total_pages)

        writer = PdfWriter()
        for page_num in range(start_page, end_page):
            writer.add_page(reader.pages[page_num])

        # Temp file for this chunk
        chunk_path = pdf_path.parent / f"{pdf_path.stem}_chunk_{start_page}_{end_page}.pdf"
        with open(chunk_path, "wb") as f:
            writer.write(f)

        yield chunk_path, start_page, end_page


def convert_pdf_chunked(pdf_path: Path, output_md_path: Path, chunk_size: int = 10) -> None:
    """
    Convert a large PDF by splitting into chunks, converting each, and merging results.
    """
    converter = build_docling_converter()
    title, pub_date = extract_pdf_metadata(pdf_path)

    chunks_md = []
    temp_files = []

    try:
        for chunk_path, start_page, end_page in split_pdf_into_chunks(pdf_path, chunk_size):
            temp_files.append(chunk_path)
            try:
                result = converter.convert(str(chunk_path))
                chunk_md = result.document.export_to_markdown()
                if chunk_md.strip():
                    chunks_md.append(chunk_md)
                print(f"  [OK] Pages {start_page}-{end_page}")
            except Exception as exc:
                print(f"  [WARN] Pages {start_page}-{end_page}: {exc}")

        # Build final markdown with metadata headers
        headers = [f"# {title}"]
        if pub_date:
            headers.append(f"## Publication Date: {pub_date}")

        final_md = "\n\n".join(headers) + "\n\n" + "\n\n".join(chunks_md)
        output_md_path.write_text(final_md, encoding="utf-8")
        print(f"[OK] {pdf_path.name} -> {output_md_path.name}")

    finally:
        # Clean up temporary chunk PDFs
        for temp_file in temp_files:
            try:
                temp_file.unlink()
            except Exception:
                pass


def run_chunked() -> None:
    input_dir = Path("./data/raw")
    output_dir = Path("./data/output_markdown")
    output_dir.mkdir(parents=True, exist_ok=True)

    processed = []
    failed = []

    for pdf_path in iter_pdf_files(input_dir):
        md_path = output_dir / f"{pdf_path.stem}.md"

        if md_path.exists():
            print(f"[SKIP] {pdf_path.name} already converted")
            continue

        try:
            print(f"Converting {pdf_path.name} in chunks...")
            convert_pdf_chunked(pdf_path, md_path, chunk_size=10)
            processed.append(pdf_path.name)
        except Exception as exc:
            print(f"[ERROR] {pdf_path.name}: {exc}")
            failed.append((pdf_path.name, str(exc)))

    print(f"\n--- SUMMARY ---")
    print(f"Processed: {len(processed)}")
    for name in processed:
        print(f"  ✓ {name}")
    if failed:
        print(f"Failed: {len(failed)}")
        for name, err in failed:
            print(f"  ✗ {name}: {err[:80]}...")


if __name__ == "__main__":
    run_chunked()
