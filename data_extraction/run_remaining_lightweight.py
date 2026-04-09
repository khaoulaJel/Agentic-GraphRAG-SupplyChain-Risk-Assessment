from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from convert_pdfs_to_markdown import extract_pdf_metadata, iter_pdf_files


def build_light_converter() -> DocumentConverter:
    """Lightweight converter: preserves table structure but disables OCR to save memory."""
    pdf_opts = PdfPipelineOptions()
    pdf_opts.do_table_structure = True
    # Disable OCR preprocessing for memory efficiency on large PDFs.
    pdf_opts.do_ocr = False

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts),
        }
    )


def run_remaining_lightweight() -> None:
    input_dir = Path("./data/raw")
    output_dir = Path("./data/output_markdown")
    output_dir.mkdir(parents=True, exist_ok=True)

    converter = build_light_converter()
    processed = []
    failed = []

    for pdf_path in iter_pdf_files(input_dir):
        md_path = output_dir / f"{pdf_path.stem}.md"

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
            print(f"  ✗ {name}: {err[:60]}...")


if __name__ == "__main__":
    run_remaining_lightweight()
