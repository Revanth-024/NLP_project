"""Exercise 10: conservative cleaning, word windows and truthful provenance.

Markdown/PDF loading, offsets and statistics are starter infrastructure. Four
small bounded cores remain Student scaffolds. No embeddings,
retrieval, generation, Streamlit, OCR or model loading belongs in this module.
"""
from io import BytesIO
from pathlib import Path
import hashlib
import re

DEFAULT_CHUNK_SIZE = 100
DEFAULT_OVERLAP = 20
MAX_PDF_BYTES = 10 * 1024 * 1024
MAX_PDF_PAGES = 50


class DocumentExtractionError(ValueError):
    """Readable extraction failure without coupling loaders to the UI."""


def load_markdown_document(path, *, document_id=None, title=None):
    """Return the same document/pages structure as PDF loading.

    Markdown has one logical text unit, not a physical page: page_number=None.
    Paths supplied by the application can be repository-relative. UTF-8 and
    optional UTF-8 BOM are accepted; all headings/content are retained.
    """
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as error:
        raise DocumentExtractionError(f"Cannot read UTF-8 Markdown: {path}. {error}") from error
    first_heading = next((line[2:].strip() for line in text.splitlines() if line.startswith("# ")), path.stem)
    return {"document_id": document_id or path.stem, "title": title or first_heading,
            "source_path": path.as_posix(), "source_type": "markdown",
            "pages": [{"page_number": None, "text": text}]}


def extract_text_from_pdf(source, *, document_id=None, title=None, source_name=None):
    """Extract ordered page text from a path or in-memory PDF bytes.

    No upload persistence or temporary files. Preserve empty pages in mixed PDFs
    so later page numbers remain accurate. Entirely empty/image-only PDFs get
    explicit text-based-PDF guidance. Encrypted PDFs are rejected, not decrypted.
    Limits are workshop guardrails, not a general-purpose secure PDF sandbox.
    """
    try:
        if isinstance(source, (bytes, bytearray)):
            data = bytes(source)
            name = source_name or "uploaded.pdf"
            path_label = f"upload://{Path(name).name}"
            default_id = "UPLOAD-" + hashlib.sha256(data).hexdigest()[:16]
            default_title = Path(name).stem
        else:
            path = Path(source)
            if path.stat().st_size > MAX_PDF_BYTES:
                raise DocumentExtractionError("PDF exceeds the 10 MiB workshop limit.")
            data = path.read_bytes()
            path_label = path.as_posix()
            default_id, default_title = path.stem, path.stem
        if len(data) > MAX_PDF_BYTES:
            raise DocumentExtractionError("PDF exceeds the 10 MiB workshop limit.")
        from pypdf import PdfReader
        reader = PdfReader(BytesIO(data), strict=False)
        if reader.is_encrypted:
            raise DocumentExtractionError("Use an unencrypted text-based PDF for this lab.")
        if len(reader.pages) > MAX_PDF_PAGES:
            raise DocumentExtractionError("Use a PDF with at most 50 pages for this lab.")
        pages = [{"page_number": number, "text": page.extract_text() or ""}
                 for number, page in enumerate(reader.pages, start=1)]
        if not any(page["text"].strip() for page in pages):
            raise DocumentExtractionError("This workshop currently supports text-based PDFs. "
                                          "No extractable text was found. OCR is outside the workshop scope.")
        pdf_title = reader.metadata.title if reader.metadata else None
        return {"document_id": document_id or default_id, "title": title or pdf_title or default_title,
                "source_path": path_label, "source_type": "pdf", "pages": pages}
    except DocumentExtractionError:
        raise
    except Exception as error:
        raise DocumentExtractionError(f"Cannot extract this PDF. Use a valid, unencrypted text-based PDF. "
                                      f"Detail: {type(error).__name__}: {error}") from error


def clean_document_text(text):
    """Normalize whitespace only; preserve casing, punctuation, numbers/headings.

    Normalize CRLF/CR to LF, collapse horizontal whitespace per line, trim each
    line, retain single newlines and at most one blank line, trim document ends.
    No stopwords, stemming, lowercasing, dehyphenation or inferred header removal.
    """
    if not isinstance(text, str):
        raise TypeError("Document text must be a string.")
    # STUDENT TODO 10.1 — Clean Extracted Text
    # Difficulty: ★ Guided | Student scaffold.
    # Goal: normalize whitespace, not language. Expected: readable natural text.
    # Hint: normalize line endings, then horizontal spaces and blank lines.
    # BEGIN STUDENT CORE 10.1
    # Implement only this educational core; surrounding setup stays provided.
    raise NotImplementedError('STUDENT TODO 10.1: implement this exercise in the marked core; see docs/STUDENT_GUIDE.md.')
    # END STUDENT CORE 10.1


def _chunk_settings(chunk_size, overlap=0):
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer word count.")
    if isinstance(overlap, bool) or not isinstance(overlap, int) or not 0 <= overlap < chunk_size:
        raise ValueError("Use an integer overlap with 0 <= overlap < chunk_size.")


def _word_spans(text):
    if not isinstance(text, str):
        raise TypeError("Chunking requires a text string.")
    return [(match.start(), match.end()) for match in re.finditer(r"\S+", text)]


def _word_window(text, spans, start, size):
    # Starter offsets: preserve the exact substring, including internal newlines.
    end = min(start + size, len(spans))
    char_start, char_end = spans[start][0], spans[end - 1][1]
    return {"word_start": start, "word_end": end, "char_start": char_start, "char_end": char_end,
            "text": text[char_start:char_end]}


def split_text_into_chunks(text, chunk_size=DEFAULT_CHUNK_SIZE):
    """Non-overlapping word windows with final partial window retained.

    Words are whitespace-delimited non-whitespace spans (Markdown symbols count too), not
    linguistic/model tokens. Zero-based word/character ranges are end-exclusive
    within the supplied text. Exact substrings retain headings and paragraph
    structure inside each chunk. Empty text produces no chunks.
    """
    _chunk_settings(chunk_size)
    spans = _word_spans(text)
    # STUDENT TODO 10.2 — Split Text Into Chunks
    # Difficulty: ★★ Core | Student scaffold.
    # Goal: cover all words in size-word steps. Expected: ordered word windows.
    # Hint: starts are 0, size, 2*size; the starter helper records exact offsets.
    # BEGIN STUDENT CORE 10.2
    # Implement only this educational core; surrounding setup stays provided.
    raise NotImplementedError('STUDENT TODO 10.2: implement this exercise in the marked core; see docs/STUDENT_GUIDE.md.')
    # END STUDENT CORE 10.2


def create_overlapping_chunks(text, chunk_size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_OVERLAP):
    """Overlapping windows; step=size-overlap; stop when a window reaches the end.

    Do not emit another window containing only an already-covered tail. Zero
    overlap reuses TODO 10.2. Overlap retains boundary context but does not make
    windows sentence-aware, guarantee completeness, or cross physical PDF pages.
    """
    _chunk_settings(chunk_size, overlap)
    if overlap == 0:
        return split_text_into_chunks(text, chunk_size)
    spans = _word_spans(text)
    # STUDENT TODO 10.3 — Add Chunk Overlap
    # Difficulty: ★★ Core | Student scaffold.
    # Goal: retain shared boundary words. Formula: step = chunk_size - overlap.
    # Expected: progressing windows including final partial text.
    # Hint: stop after a window includes the last word; avoid redundant tails.
    # BEGIN STUDENT CORE 10.3
    # Implement only this educational core; surrounding setup stays provided.
    raise NotImplementedError('STUDENT TODO 10.3: implement this exercise in the marked core; see docs/STUDENT_GUIDE.md.')
    # END STUDENT CORE 10.3


def attach_chunk_metadata(chunks, document, *, page_number=None, start_index=0):
    """Add source identity with sequential document-wide indexes/IDs.

    IDs are document_id::chunk_NNN, deterministic for the same source/settings.
    They are positional, not content-addressed: re-chunking/editing can change
    what an ID means. Word/character offsets address the CLEANED page/text unit,
    not original PDF byte offsets. Markdown page_number must remain None.
    """
    for field in ("document_id", "title", "source_path", "source_type"):
        if not isinstance(document.get(field), str) or not document[field].strip():
            raise ValueError(f"Document metadata needs non-empty {field}.")
    if document["source_type"] not in ("markdown", "pdf"):
        raise ValueError("source_type must be markdown or pdf.")
    if isinstance(start_index, bool) or not isinstance(start_index, int) or start_index < 0:
        raise ValueError("start_index must be a non-negative integer.")
    if document["source_type"] == "markdown" and page_number is not None:
        raise ValueError("Markdown does not have physical page numbers.")
    if document["source_type"] == "pdf" and (isinstance(page_number, bool) or not isinstance(page_number, int) or page_number < 1):
        raise ValueError("PDF chunks require a positive physical page number.")
    # STUDENT TODO 10.4 — Attach Chunk Metadata
    # Difficulty: ★★ Core | Student scaffold.
    # Goal: keep source provenance. Expected: labeled chunks with stable IDs.
    # Hint: combine document identity, running index, page and window boundaries.
    # BEGIN STUDENT CORE 10.4
    # Implement only this educational core; surrounding setup stays provided.
    raise NotImplementedError('STUDENT TODO 10.4: implement this exercise in the marked core; see docs/STUDENT_GUIDE.md.')
    # END STUDENT CORE 10.4


def process_document(document, chunk_size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_OVERLAP):
    """Starter composition: clean and chunk each text/page unit separately.

    Overlap resets at a PDF page boundary. Empty pages create no chunks, but
    retain their numbered place in cleaned_pages. Chunk index is document-wide.
    """
    _chunk_settings(chunk_size, overlap)
    cleaned_pages, chunks = [], []
    for page in document["pages"]:
        text = clean_document_text(page["text"])
        cleaned_pages.append({"page_number": page["page_number"], "text": text})
        windows = create_overlapping_chunks(text, chunk_size, overlap)
        chunks.extend(attach_chunk_metadata(windows, document, page_number=page["page_number"], start_index=len(chunks)))
    return {"document": document, "cleaned_pages": cleaned_pages, "chunks": chunks,
            "settings": {"chunk_size": chunk_size, "overlap": overlap},
            "statistics": chunk_statistics(cleaned_pages, chunks, overlap)}


def chunk_statistics(cleaned_pages, chunks, overlap):
    """Starter counts; duplicated words count repeated covered word positions."""
    word_count = sum(len(_word_spans(page["text"])) for page in cleaned_pages)
    sizes = [chunk["word_end"] - chunk["word_start"] for chunk in chunks]
    return {"document_word_count": word_count, "chunk_count": len(chunks),
            "min_words": min(sizes, default=0), "max_words": max(sizes, default=0),
            "average_words": sum(sizes) / len(sizes) if sizes else 0.0,
            "overlap": overlap, "duplicated_word_positions": sum(sizes) - word_count}


def model_token_lengths(tokenizer, texts):
    """Starter analysis only: count untruncated tokens INCLUDING special tokens.

    No encoding to embeddings, model inference, truncation or retrieval. Counts
    use the tokenizer supplied by the caller; word counts are not token counts.
    """
    return [len(tokenizer(text, truncation=False, add_special_tokens=True, verbose=False)["input_ids"])
            for text in texts]
