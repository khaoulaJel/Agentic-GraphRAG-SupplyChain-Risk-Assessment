from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional, Tuple

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

try:
    from pypdf import PdfReader
except Exception:  # Optional dependency for metadata extraction
    PdfReader = None


def iter_pdf_files(input_dir: Path) -> Iterator[Path]:
    # Lazy iteration processes one PDF at a time to avoid memory spikes.
    for pdf_path in sorted(input_dir.glob("*.pdf")):
        if pdf_path.is_file():
            yield pdf_path


def parse_pdf_date(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None

    m = re.search(r"(\d{4})(\d{2})?(\d{2})?", raw)
    if not m:
        return None

    year = int(m.group(1))
    month = int(m.group(2) or 1)
    day = int(m.group(3) or 1)

    try:
        return datetime(year, month, day).strftime("%Y-%m-%d")
    except ValueError:
        return str(year)


def extract_pdf_metadata(pdf_path: Path) -> Tuple[str, Optional[str]]:
    title = pdf_path.stem
    pub_date = None

    if PdfReader is None:
        return title, pub_date

    try:
        reader = PdfReader(str(pdf_path))
        meta = reader.metadata or {}
        meta_title = meta.get("/Title")
        if meta_title and str(meta_title).strip():
            title = str(meta_title).strip()

        pub_date = parse_pdf_date(meta.get("/CreationDate")) or parse_pdf_date(meta.get("/ModDate"))
    except Exception:
        # Metadata read failures should not stop conversion.
        pass

    return title, pub_date


def build_docling_converter() -> DocumentConverter:
    # Enable high-fidelity table extraction (TableFormer) for complex supply-chain tables.
    pdf_opts = PdfPipelineOptions()
    pdf_opts.do_table_structure = True

    if hasattr(pdf_opts, "table_structure_options"):
        tso = pdf_opts.table_structure_options
        try:
            from docling.datamodel.pipeline_options import TableFormerMode

            if hasattr(tso, "mode"):
                tso.mode = TableFormerMode.ACCURATE
        except Exception:
            pass

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts),
        }
    )


def process_folder(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    converter = build_docling_converter()

    for pdf_path in iter_pdf_files(input_dir):
        md_path = output_dir / f"{pdf_path.stem}.md"
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
            # Per-file exception handling keeps batch processing resilient.
            print(f"[ERROR] {pdf_path.name}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert SEC 10-K and sustainability PDFs to clean Markdown with docling."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("./data/raw"),
        help="Folder containing PDF documents.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("./data/output_markdown"),
        help="Folder where Markdown files will be written.",
    )
    args = parser.parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    process_folder(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
