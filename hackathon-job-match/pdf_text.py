from __future__ import annotations

import io
import re
import zlib
from typing import Iterable


MAX_PDF_BYTES = 8_000_000


def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract text from a PDF resume.

    The app stays dependency-light for hackathon portability. If pypdf or
    PyPDF2 is installed, use it. Otherwise, fall back to a small parser that
    handles common text streams, including FlateDecode-compressed streams.
    """

    if len(pdf_bytes) > MAX_PDF_BYTES:
        raise ValueError("PDF is too large. Please upload a file under 8 MB.")
    if not pdf_bytes.startswith(b"%PDF"):
        raise ValueError("The uploaded file does not look like a PDF.")

    text = normalize_pdf_text(extract_with_pdf_library(pdf_bytes))
    if is_readable_resume_text(text):
        return text

    fallback_text = normalize_pdf_text(extract_with_stream_fallback(pdf_bytes))
    if is_readable_resume_text(fallback_text):
        return fallback_text

    raise ValueError(
        "Could not extract clean resume text from this PDF. If it is scanned, "
        "OCR it first; otherwise export it as text or paste the resume content."
    )


def extract_with_pdf_library(pdf_bytes: bytes) -> str:
    for module_name in ("pypdf", "PyPDF2"):
        try:
            module = __import__(module_name)
        except ImportError:
            continue

        try:
            reader = module.PdfReader(io.BytesIO(pdf_bytes))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception:
            continue
    return ""


def extract_with_stream_fallback(pdf_bytes: bytes) -> str:
    chunks = []
    for stream in iter_pdf_streams(pdf_bytes):
        decoded = maybe_decompress(stream)
        chunks.extend(text_from_pdf_commands(decoded))
    return "\n".join(chunks)


def iter_pdf_streams(pdf_bytes: bytes) -> Iterable[bytes]:
    for match in re.finditer(rb"stream\r?\n(.*?)\r?\nendstream", pdf_bytes, flags=re.S):
        yield match.group(1)


def maybe_decompress(stream: bytes) -> bytes:
    for wbits in (zlib.MAX_WBITS, -zlib.MAX_WBITS):
        try:
            return zlib.decompress(stream, wbits)
        except zlib.error:
            continue
    return stream


def text_from_pdf_commands(stream: bytes) -> list[str]:
    text = stream.decode("latin-1", errors="ignore")
    parts: list[str] = []

    for value in re.findall(r"\(((?:\\.|[^\\)])*)\)\s*Tj", text):
        parts.append(unescape_pdf_string(value))

    for array in re.findall(r"\[(.*?)\]\s*TJ", text, flags=re.S):
        strings = re.findall(r"\((?:\\.|[^\\)])*\)", array)
        if strings:
            parts.append("".join(unescape_pdf_string(item[1:-1]) for item in strings))

    for hex_value in re.findall(r"<([0-9A-Fa-f\s]+)>\s*Tj", text):
        cleaned = re.sub(r"\s+", "", hex_value)
        try:
            raw = bytes.fromhex(cleaned)
        except ValueError:
            continue
        parts.append(raw.decode("utf-16-be", errors="ignore") or raw.decode("latin-1", errors="ignore"))

    return [part for part in parts if part.strip()]


def unescape_pdf_string(value: str) -> str:
    replacements = {
        r"\(": "(",
        r"\)": ")",
        r"\\": "\\",
        r"\n": "\n",
        r"\r": "\n",
        r"\t": "\t",
        r"\b": "\b",
        r"\f": "\f",
    }
    for source, target in replacements.items():
        value = value.replace(source, target)
    value = re.sub(
        r"\\([0-7]{1,3})",
        lambda match: chr(int(match.group(1), 8)),
        value,
    )
    return value


def normalize_pdf_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"([A-Za-z])-\n([a-z])", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)

    lines = [line.strip() for line in text.split("\n")]
    lines = remove_fragment_blank_lines(lines)
    return compact_resume_lines(lines)


SECTION_HEADINGS = {
    "summary",
    "profile",
    "experience",
    "professional experience",
    "work experience",
    "education",
    "skills",
    "technical skills",
    "projects",
    "certifications",
    "certification",
    "publications",
    "awards",
    "leadership",
}


def remove_fragment_blank_lines(lines: list[str]) -> list[str]:
    compacted = []
    for index, line in enumerate(lines):
        if line:
            compacted.append(line)
            continue

        previous_line = next((item for item in reversed(lines[:index]) if item.strip()), "")
        next_line = next((item for item in lines[index + 1 :] if item.strip()), "")
        if is_fragment_line(previous_line) and is_fragment_line(next_line):
            continue
        if compacted and compacted[-1] == "":
            continue
        compacted.append("")
    return compacted


def compact_resume_lines(lines: list[str]) -> str:
    blocks: list[str] = []
    paragraph: list[str] = []

    def flush_paragraph() -> None:
        if not paragraph:
            return
        blocks.append(compact_paragraph(paragraph))
        paragraph.clear()

    for line in lines:
        if not line:
            flush_paragraph()
            continue
        if is_section_heading(line):
            flush_paragraph()
            blocks.append(line)
            continue
        paragraph.append(line)

    flush_paragraph()
    return "\n\n".join(block for block in blocks if block).strip()


def compact_paragraph(lines: list[str]) -> str:
    merged: list[str] = []
    for line in lines:
        if not merged:
            merged.append(line)
            continue
        if starts_bullet(line):
            merged.append(line)
            continue
        if starts_bullet(merged[-1]):
            merged.append(line)
            continue
        if merged[-1].endswith(":"):
            merged.append(line)
            continue
        merged[-1] = f"{merged[-1]} {line}"
    return "\n".join(merged)


def is_fragment_line(line: str) -> bool:
    line = line.strip()
    if not line or is_section_heading(line) or starts_bullet(line):
        return False
    return len(line) <= 45


def is_section_heading(line: str) -> bool:
    normalized = re.sub(r"[^a-z ]+", "", line.lower()).strip()
    return normalized in SECTION_HEADINGS


def starts_bullet(line: str) -> bool:
    return bool(re.match(r"^([•*-]|\d+[.)])\s+", line.strip()))


def is_readable_resume_text(text: str) -> bool:
    if len(text.strip()) < 40:
        return False

    printable = sum(1 for char in text if char.isprintable() or char in "\n\t")
    ascii_letters = sum(1 for char in text if char.isascii() and char.isalpha())
    replacement_chars = text.count("\ufffd")
    total = max(len(text), 1)

    if printable / total < 0.88:
        return False
    if replacement_chars > 3:
        return False
    if ascii_letters < 20:
        return False

    resume_signals = (
        "experience",
        "education",
        "skills",
        "projects",
        "work",
        "engineer",
        "research",
        "python",
        "university",
        "email",
    )
    normalized = text.lower()
    return any(signal in normalized for signal in resume_signals)
