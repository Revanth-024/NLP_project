"""Transparent lexical/semantic retrieval; shared ranking, no Streamlit or LLMs."""
from src.preprocessing import lowercase_text, remove_punctuation, tokenize_words
from src.classical_nlp import (
    build_vocabulary, calculate_df, calculate_idf, build_tfidf_matrix,
    build_query_tfidf_vector, cosine_similarity, _validate_vector,
)


def tokenize_for_search(text: str) -> list[str]:
    """Shared visible policy: lowercase, punctuation spaces, WordPunct tokens.

    Retain stopwords and word forms to isolate lexical weighting; no downloads.
    """
    return tokenize_words(remove_punctuation(lowercase_text(text)))


def score_documents(query_vector, document_vectors) -> list[float]:
    """Compare query with every document in original corpus order."""
    _validate_vector(query_vector)
    for vector in document_vectors:
        _validate_vector(vector, len(query_vector))
    # STUDENT TODO 4.2 — Query-Document Similarities
    # Difficulty: ★★ Core | Reference complete. Hint: reuse manual cosine.
    # BEGIN STUDENT CORE 4.2
    # Implement only this educational core; surrounding setup stays provided.
    raise NotImplementedError('STUDENT TODO 4.2: implement this exercise in the marked core; see docs/STUDENT_GUIDE.md.')
    # END STUDENT CORE 4.2


def rank_documents(documents: list[dict], scores: list[float], top_k: int = 3) -> list[dict]:
    """Return copied metadata plus scores; stable ties retain corpus order.

    Include zero-score rows transparently; a result is not proof of relevance.
    top_k may be zero or larger than the corpus, but cannot be negative.
    """
    _validate_vector(scores, len(documents))
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
        raise ValueError("top_k must be a non-negative integer.")
    # STUDENT TODO 4.3 — Rank Documents
    # Difficulty: ★★ Core | Reference complete. Hint: Python sorting is stable.
    # BEGIN STUDENT CORE 4.3
    # Implement only this educational core; surrounding setup stays provided.
    raise NotImplementedError('STUDENT TODO 4.3: implement this exercise in the marked core; see docs/STUDENT_GUIDE.md.')
    # END STUDENT CORE 4.3


def _validate_search_documents(documents: list[dict]) -> None:
    """Shared starter metadata checks; never use titles/IDs as searchable text."""
    if not isinstance(documents, list) or any(not isinstance(document, dict) or not isinstance(document.get("document_id"), str) or not document["document_id"] or not isinstance(document.get("text"), str) for document in documents):
        raise ValueError("Provide document dictionaries with non-empty document_id and string text.")
    if len({document["document_id"] for document in documents}) != len(documents):
        raise ValueError("Document IDs must be unique.")


def prepare_search_corpus(documents: list[dict]) -> dict:
    """Starter preparation returns inspectable tokens, vocabulary, DF, IDF/matrix.

    Required metadata: unique document_id and string text; optional title.
    Empty corpus is valid; no IDF calculation with N=0 is attempted.
    """
    _validate_search_documents(documents)
    tokens = [tokenize_for_search(document["text"]) for document in documents]
    vocabulary = build_vocabulary(tokens)
    df = calculate_df(tokens, vocabulary)
    idf = calculate_idf(df, len(documents)) if documents else []
    return {"documents": [dict(document) for document in documents], "tokens": tokens,
            "vocabulary": vocabulary, "df": df, "idf": idf,
            "matrix": build_tfidf_matrix(tokens, vocabulary, idf)}


def search_tfidf(query: str, corpus: dict, top_k: int = 3) -> dict:
    """Return query representations, scores and ranked results; no new query axes."""
    tokens = tokenize_for_search(query)
    vocabulary = corpus["vocabulary"]
    query_vector = build_query_tfidf_vector(tokens, vocabulary, corpus["idf"])
    scores = score_documents(query_vector, corpus["matrix"])
    results = rank_documents(corpus["documents"], scores, top_k)
    return {"tokens": tokens, "oov_terms": sorted(set(tokens) - set(vocabulary)),
            "query_vector": query_vector, "scores": scores, "results": results}


def encode_corpus_documents(model, documents: list[dict]):
    """Encode only document text in original order as an N × 384 array.

    Empty corpus yields (0,384). Blank document texts are rejected by the shared
    encoder. Metadata is preserved separately, never prepended to descriptions.
    Caching and model loading belong to starter UI/setup, outside this function.
    """
    from src.embeddings import encode_sentences
    _validate_search_documents(documents)
    # STUDENT TODO 7.1 — Encode Corpus Documents
    # Difficulty: ★★ Core | Student scaffold. Expected: one row per document.
    # Goal: reuse the sentence encoder on text fields. Hint: preserve row order.
    # BEGIN STUDENT CORE 7.1
    # Implement only this educational core; surrounding setup stays provided.
    raise NotImplementedError('STUDENT TODO 7.1: implement this exercise in the marked core; see docs/STUDENT_GUIDE.md.')
    # END STUDENT CORE 7.1


def encode_search_query(model, query: str) -> list[float]:
    """One complete query → one 384-dimensional list; blank queries fail clearly."""
    from src.embeddings import encode_sentences
    if not isinstance(query, str):
        raise TypeError("The search query must be a text string.")
    if not query.strip():
        raise ValueError("Enter a non-empty search query before encoding.")
    # STUDENT TODO 7.2 — Encode Search Query
    # Difficulty: ★ Guided | Student scaffold. Goal: use the corpus encoder.
    # Expected: 384 floats, not a (1,384) batch. Hint: take the first encoded row.
    # BEGIN STUDENT CORE 7.2
    # Implement only this educational core; surrounding setup stays provided.
    raise NotImplementedError('STUDENT TODO 7.2: implement this exercise in the marked core; see docs/STUDENT_GUIDE.md.')
    # END STUDENT CORE 7.2


def _semantic_matrix(document_embeddings, document_count=None):
    """Starter shape/finite checks; accept array or list-of-lists, never mutate."""
    import numpy as np
    from src.sentence_resources import EMBEDDING_DIMENSION
    try:
        matrix = np.asarray(document_embeddings, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("Provide a numeric document embedding matrix.") from error
    if matrix.ndim != 2 or matrix.shape[1] != EMBEDDING_DIMENSION:
        raise ValueError("Document embeddings must have shape (N,384), including (0,384).")
    if document_count is not None and len(matrix) != document_count:
        raise ValueError("Embedding rows must match document count and original order.")
    if not np.isfinite(matrix).all():
        raise ValueError("Document embeddings must be finite numbers.")
    return matrix


def calculate_semantic_similarities(query_vector, document_embeddings) -> list[float]:
    """Return one manual cosine per row, in original document order."""
    from src.sentence_resources import EMBEDDING_DIMENSION
    _validate_vector(query_vector, EMBEDDING_DIMENSION)
    matrix = _semantic_matrix(document_embeddings)
    # STUDENT TODO 7.3 — Calculate Semantic Similarities
    # Difficulty: ★★ Core | Student scaffold. Goal: query versus every row.
    # Expected: N cosine scores. Hint: reuse existing score_documents/manual cosine.
    # BEGIN STUDENT CORE 7.3
    # Implement only this educational core; surrounding setup stays provided.
    raise NotImplementedError('STUDENT TODO 7.3: implement this exercise in the marked core; see docs/STUDENT_GUIDE.md.')
    # END STUDENT CORE 7.3


def retrieve_semantic_results(documents: list[dict], scores: list[float], top_k: int = 5) -> list[dict]:
    """Shared result dictionaries; descending scores, original-order ties.

    K=0 returns []; oversized K returns all; negative/non-integer K is invalid.
    Even unrelated queries produce nearest results. Scores are not confidence.
    """
    _validate_search_documents(documents)
    # STUDENT TODO 7.4 — Retrieve Top-K Semantic Results
    # Difficulty: ★★ Core | Student scaffold. Goal: reuse deterministic ranking.
    # Expected: copied metadata plus score. Hint: the ranking operation is unchanged.
    # BEGIN STUDENT CORE 7.4
    # Implement only this educational core; surrounding setup stays provided.
    raise NotImplementedError('STUDENT TODO 7.4: implement this exercise in the marked core; see docs/STUDENT_GUIDE.md.')
    # END STUDENT CORE 7.4


def semantic_search(query: str, documents: list[dict], document_embeddings, model, top_k: int = 5) -> dict:
    """Starter composition exposes query vector, scores, sorted indices/results.

    Uses only query, document content and their representations. No benchmark
    expected IDs, metadata labels, vector store, chunking or answer generation.
    Caller supplies embeddings from the same encoder, in document order.
    """
    _validate_search_documents(documents)
    matrix = _semantic_matrix(document_embeddings, len(documents))
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
        raise ValueError("top_k must be a non-negative integer.")
    query_vector = encode_search_query(model, query)
    scores = calculate_semantic_similarities(query_vector, matrix)
    ranked = retrieve_semantic_results(documents, scores, len(documents))
    positions = {document["document_id"]: index for index, document in enumerate(documents)}
    return {"query_vector": query_vector, "scores": scores,
            "sorted_indices": [positions[result["document_id"]] for result in ranked],
            "results": ranked[:top_k]}


def load_canonical_documents(root):
    """Starter inventory: only the fourteen canonical Markdown sources.

    Read current content so UI cache keys can include source text and identity.
    No PDFs, question papers, benchmark expectations or fact answers are indexed.
    """
    import json
    from pathlib import Path
    from src.document_processor import load_markdown_document
    root = Path(root)
    manifest = json.loads((root / "data/document_manifest.json").read_text(encoding="utf-8"))
    return [load_markdown_document(root / "data" / entry["filename"],
            document_id=entry["document_id"], title=entry["title"])
            for entry in manifest["academic_documents"]]


def build_chunk_corpus(documents, chunk_size=100, overlap=20):
    """Starter composition reuses the existing processor, preserving order."""
    from src.document_processor import process_document
    return [chunk for document in documents
            for chunk in process_document(document, chunk_size, overlap)["chunks"]]


def _validate_chunks(chunks):
    """Starter provenance checks. Many chunks may share one document ID."""
    if not isinstance(chunks, list):
        raise ValueError("Provide a list of chunk dictionaries.")
    required = ["chunk_id", "document_id", "document_title", "source_path", "text"]
    for chunk in chunks:
        if not isinstance(chunk, dict) or any(not isinstance(chunk.get(k), str) or not chunk[k].strip() for k in required):
            raise ValueError("Each chunk needs non-empty text and source identity.")
        if chunk.get("source_type") not in ("markdown", "pdf"):
            raise ValueError("Each chunk needs a markdown/pdf source type.")
        page = chunk.get("page_number")
        if chunk["source_type"] == "markdown" and page is not None:
            raise ValueError("Markdown chunks must not invent physical page numbers.")
        if chunk["source_type"] == "pdf" and (isinstance(page, bool) or not isinstance(page, int) or page < 1):
            raise ValueError("PDF chunks need a physical page number.")
        if isinstance(chunk.get("chunk_index"), bool) or not isinstance(chunk.get("chunk_index"), int) or chunk["chunk_index"] < 0:
            raise ValueError("Chunk indexes must be non-negative integers.")
        for start, end in [("word_start", "word_end"), ("char_start", "char_end")]:
            if any(isinstance(chunk.get(k), bool) or not isinstance(chunk.get(k), int) for k in [start, end]) or not 0 <= chunk[start] < chunk[end]:
                raise ValueError("Chunks need increasing, non-negative word/character bounds.")
    if len({chunk["chunk_id"] for chunk in chunks}) != len(chunks):
        raise ValueError("Chunk IDs must be unique within the corpus.")


def encode_document_chunks(model, chunks):
    """Text-only batch encoding; row i corresponds exactly to chunk i.

    Empty corpus gives (0,384). Blank texts fail clearly. Model loading, caching
    and untruncated token-length preflight are starter responsibilities.
    """
    from src.embeddings import encode_sentences
    _validate_chunks(chunks)
    # STUDENT TODO 11.1 — Encode Document Chunks
    # Difficulty: ★★ Core | Student scaffold.
    # Goal: one row per passage. Expected: N × 384; never encode metadata.
    # Hint: extract text fields in order and reuse the batch sentence encoder.
    # BEGIN STUDENT CORE 11.1
    # Implement only this educational core; surrounding setup stays provided.
    raise NotImplementedError('STUDENT TODO 11.1: implement this exercise in the marked core; see docs/STUDENT_GUIDE.md.')
    # END STUDENT CORE 11.1


def encode_user_question(model, question):
    """Same model/vector space as chunks; reject blank questions."""
    # STUDENT TODO 11.2 — Encode the User Question
    # Difficulty: ★ Guided | Student scaffold.
    # Goal: one 384-dimensional vector. Hint: reuse the existing query encoder.
    # BEGIN STUDENT CORE 11.2
    # Implement only this educational core; surrounding setup stays provided.
    raise NotImplementedError('STUDENT TODO 11.2: implement this exercise in the marked core; see docs/STUDENT_GUIDE.md.')
    # END STUDENT CORE 11.2


def calculate_chunk_similarities(question_vector, chunk_embeddings):
    """One existing manual cosine per row; zero vectors safely score zero."""
    # STUDENT TODO 11.3 — Calculate Question-Chunk Similarities
    # Difficulty: ★★ Core | Student scaffold.
    # Goal: preserve row/score alignment. Hint: reuse semantic cosine scoring.
    # BEGIN STUDENT CORE 11.3
    # Implement only this educational core; surrounding setup stays provided.
    raise NotImplementedError('STUDENT TODO 11.3: implement this exercise in the marked core; see docs/STUDENT_GUIDE.md.')
    # END STUDENT CORE 11.3


def retrieve_top_k_chunks(chunks, scores, top_k=5):
    """Copied provenance plus score/rank; original-order ties, no deduplication.

    K=0 returns []; oversized K returns all. Negative/non-integer K fails.
    Ranking always finds nearest neighbors, even if none are relevant.
    """
    _validate_chunks(chunks)
    # STUDENT TODO 11.4 — Retrieve Top-K Evidence Chunks
    # Difficulty: ★★ Core | Student scaffold.
    # Goal: keep source metadata. Hint: shared ranking, then one-based ranks.
    # BEGIN STUDENT CORE 11.4
    # Implement only this educational core; surrounding setup stays provided.
    raise NotImplementedError('STUDENT TODO 11.4: implement this exercise in the marked core; see docs/STUDENT_GUIDE.md.')
    # END STUDENT CORE 11.4


def semantic_chunk_search(question, chunks, chunk_embeddings, model, top_k=5):
    """Starter orchestration: evidence only, no expected answers or generation.

    Caller guarantees same-model embeddings in chunk order. Counts/shapes are
    checked, but a supplied matrix cannot prove its semantic row identity.
    """
    _validate_chunks(chunks)
    matrix = _semantic_matrix(chunk_embeddings, len(chunks))
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
        raise ValueError("top_k must be a non-negative integer.")
    vector = encode_user_question(model, question)
    scores = calculate_chunk_similarities(vector, matrix)
    ranked = retrieve_top_k_chunks(chunks, scores, len(chunks))
    positions = {chunk["chunk_id"]: i for i, chunk in enumerate(chunks)}
    return {"question_vector": vector, "scores": scores,
            "sorted_indices": [positions[c["chunk_id"]] for c in ranked],
            "results": ranked[:top_k]}
