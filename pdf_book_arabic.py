#!/usr/bin/env python3
"""Extract PDF pages and images to Markdown, translate with Ollama, rebuild an Arabic PDF.

Usage:
  python3 pdf_book_arabic.py book.pdf --output-dir book_ar
  python3 pdf_book_arabic.py book.pdf --output-dir book_ar --from-page 1 --to-page 20
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import re
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:
    import fitz  # PyMuPDF
except ImportError:
    raise SystemExit("Missing PyMuPDF. Install with: python3 -m pip install -r requirements.txt")

OLLAMA_URL = "http://localhost:11434/api/chat"


def extract_pdf(pdf_path: Path, output_dir: Path, start_page: int, end_page: int | None) -> Path:
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    last = len(doc) if end_page is None else min(end_page, len(doc))
    if start_page < 1 or start_page > last:
        raise SystemExit(f"Invalid page range: {start_page}..{last}")

    pages: list[str] = []
    image_counts: dict[int, int] = {}
    for page_no in range(start_page - 1, last):
        page = doc[page_no]
        blocks = page.get_text("dict").get("blocks", [])
        parts: list[str] = []
        # Blocks are in PDF reading order; this preserves text/image ordering as far as possible.
        for block in blocks:
            if block.get("type") == 0:
                lines = []
                for line in block.get("lines", []):
                    text = "".join(span.get("text", "") for span in line.get("spans", []))
                    if text.strip():
                        lines.append(text.rstrip())
                if lines:
                    parts.append("\n".join(lines))
            elif block.get("type") == 1 and block.get("image"):
                ext = block.get("ext", "png")
                if ext not in {"png", "jpg", "jpeg", "jxr", "jp2", "bmp", "tiff", "webp"}:
                    ext = "png"
                image_counts[page_no + 1] = image_counts.get(page_no + 1, 0) + 1
                name = f"page_{page_no + 1:04d}_image_{image_counts[page_no + 1]:02d}.{ext}"
                raw = block["image"]
                # Some formats are not directly supported by browsers/PDF renderers; convert to PNG.
                if ext in {"jxr", "jp2", "webp"}:
                    pix = fitz.Pixmap(raw)
                    raw = pix.tobytes("png")
                    name = Path(name).with_suffix(".png").name
                (image_dir / name).write_bytes(raw)
                parts.append(f"![صورة الصفحة {page_no + 1}](images/{name})")
        body = "\n\n".join(parts).strip()
        if not body:
            body = "[لا يوجد نص مستخرج من هذه الصفحة. قد تكون الصفحة ممسوحة ضوئيًا وتحتاج إلى OCR.]"
        pages.append(f"<!-- PAGE:{page_no + 1} -->\n\n{body}")

    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / "book.md"
    md_path.write_text("\n\n---\n\n".join(pages) + "\n", encoding="utf-8")
    doc.close()
    print(f"Extracted {last - start_page + 1} page(s) to {md_path}; images: {image_dir}")
    return md_path


def translate_page(text: str, model: str, ollama_url: str, retries: int = 3) -> str:
    prompt = (
        "Translate the following document content into clear, faithful Modern Standard Arabic. "
        "Return only the translation. Preserve Markdown syntax, headings, lists, tables, "
        "image links, URLs, HTML comments such as <!-- PAGE:1 -->, numbers, and code exactly. "
        "Do not add explanations.\n\nCONTENT:\n" + text
    )
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a careful document translator. Translate into Arabic and preserve all structure and non-translatable tokens."},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": 0.1},
    }).encode("utf-8")
    req = Request(ollama_url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    for attempt in range(retries):
        try:
            with urlopen(req, timeout=900) as response:
                data = json.loads(response.read().decode("utf-8"))
            result = data.get("message", {}).get("content", "").strip()
            if not result:
                raise RuntimeError("Ollama returned an empty translation")
            return result
        except (HTTPError, URLError, TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
            if attempt + 1 == retries:
                raise RuntimeError(f"Translation failed: {exc}") from exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("Translation failed")


def translate_markdown(source: Path, target: Path, model: str, ollama_url: str) -> None:
    content = source.read_text(encoding="utf-8")
    # Extract and translate each page independently to reduce model context pressure.
    chunks = re.split(r"(?=<!-- PAGE:\d+ -->)", content)
    translated: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        if not chunk.strip():
            continue
        print(f"Translating page {i}/{len(chunks)}...", flush=True)
        translated.append(translate_page(chunk.strip(), model, ollama_url))
    target.write_text("\n\n---\n\n".join(translated) + "\n", encoding="utf-8")
    print(f"Arabic Markdown saved to {target}")


def markdown_to_html(md: str) -> str:
    try:
        import markdown
    except ImportError:
        raise SystemExit("Missing Markdown package. Install with: python3 -m pip install -r requirements.txt")
    # Split page markers into separate printable sections; don't expose marker comments.
    chunks = re.split(r"<!-- PAGE:(\d+) -->", md)
    sections = []
    if chunks[0].strip():
        sections.append('<section class="page">' + markdown.markdown(chunks[0], extensions=["tables", "fenced_code", "sane_lists"]) + "</section>")
    for i in range(1, len(chunks), 2):
        page_text = chunks[i + 1] if i + 1 < len(chunks) else ""
        sections.append('<section class="page">' + markdown.markdown(page_text, extensions=["tables", "fenced_code", "sane_lists"]) + "</section>")
    return "\n".join(sections)


def build_pdf(md_path: Path, output_pdf: Path, page_size: str = "A4") -> None:
    try:
        from weasyprint import HTML
    except ImportError:
        raise SystemExit("Missing WeasyPrint. Install with: python3 -m pip install -r requirements.txt")
    md = md_path.read_text(encoding="utf-8")
    body = markdown_to_html(md)
    safe_size = page_size if page_size.upper() in {"A4", "A5", "LETTER"} else "A4"
    css = f"""
      @page {{ size: {safe_size}; margin: 18mm 17mm; }}
      html {{ direction: rtl; }}
      body {{ font-family: 'Noto Naskh Arabic', 'Noto Sans Arabic', 'DejaVu Sans', sans-serif;
             direction: rtl; text-align: right; line-height: 1.75; font-size: 11pt; color: #171717; }}
      .page {{ break-before: page; page-break-before: always; }}
      .page:first-child {{ break-before: auto; page-break-before: auto; }}
      p {{ margin: 0 0 0.8em; }} h1,h2,h3,h4 {{ line-height: 1.35; break-after: avoid; }}
      img {{ display: block; max-width: 100%; max-height: 220mm; height: auto; margin: 1em auto; }}
      table {{ width: 100%; border-collapse: collapse; direction: rtl; }}
      th,td {{ border: 1px solid #888; padding: 4px 6px; }}
      pre,code {{ direction: ltr; text-align: left; white-space: pre-wrap; }}
      blockquote {{ border-right: 3px solid #aaa; margin-right: 0; padding-right: 1em; color: #444; }}
    """
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    HTML(string=f"<!doctype html><html lang='ar' dir='rtl'><head><meta charset='utf-8'></head><body>{body}</body></html>",
         base_url=str(md_path.parent)).write_pdf(str(output_pdf), stylesheets=[__import__("weasyprint").CSS(string=css)])
    print(f"PDF saved to {output_pdf}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract a PDF book, translate it to Arabic with Ollama, and rebuild a PDF.")
    parser.add_argument("pdf", type=Path, help="Input PDF (must contain selectable text; scanned PDFs need OCR first)")
    parser.add_argument("--output-dir", type=Path, default=Path("translated_book"), help="Output folder")
    parser.add_argument("--model", default="translategemma:12b", help="Ollama model (default: translategemma:12b)")
    parser.add_argument("--ollama-url", default=OLLAMA_URL, help="Ollama chat API URL")
    parser.add_argument("--from-page", type=int, default=1, help="First page, 1-based")
    parser.add_argument("--to-page", type=int, default=None, help="Last page, inclusive")
    parser.add_argument("--page-size", choices=["A4", "A5", "LETTER"], default="A4")
    parser.add_argument("--skip-translation", action="store_true", help="Only extract and create untranslated PDF")
    args = parser.parse_args()
    if not args.pdf.is_file():
        raise SystemExit(f"Input PDF not found: {args.pdf}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_md = extract_pdf(args.pdf, args.output_dir, args.from_page, args.to_page)
    arabic_md = args.output_dir / "book_ar.md"
    if args.skip_translation:
        arabic_md.write_text(source_md.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        translate_markdown(source_md, arabic_md, args.model, args.ollama_url)
    build_pdf(arabic_md, args.output_dir / "book_ar.pdf", args.page_size)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)
