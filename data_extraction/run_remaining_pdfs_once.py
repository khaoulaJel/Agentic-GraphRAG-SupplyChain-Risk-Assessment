from pathlib import Path

from convert_pdfs_to_markdown import (
    build_docling_converter,
    extract_pdf_metadata,
    iter_pdf_files,
)


def run_remaining_only() -> None:
    input_dir = Path("./data/raw")
    output_dir = Path("./data/output_markdown")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Reuse one converter instance for the full batch to avoid repeated model init overhead.
    converter = build_docling_converter()

    for pdf_path in iter_pdf_files(input_dir):
        md_path = output_dir / f"{pdf_path.stem}.md"

        # Skip already-converted PDFs so this one-off script only handles the remaining files.
        if md_path.exists():
            print(f"[SKIP] {pdf_path.name} already converted")
            continue

        try:
            title, pub_date = extract_pdf_metadata(pdf_path)
            result = converter.convert(str(pdf_path))
            body_md = result.document.export_to_markdown()

            headers = [f"# {title}"]
            if pub_date:
                headers.append(f"## Publication Date: {pub_date}")

            final_md = "\n\n".join(headers) + "\n\n" + body_md
            md_path.write_text(final_md, encoding="utf-8")
            print(f"[OK] {pdf_path.name} -> {md_path.name}")
        except Exception as exc:
            print(f"[ERROR] {pdf_path.name}: {exc}")


if __name__ == "__main__":
    run_remaining_only()
